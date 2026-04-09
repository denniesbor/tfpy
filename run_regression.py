"""
Author: Dennies Bor
Role:   Entry point for running and saving all GIC regression model comparisons.
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import setup_logger, RESULTS_DIR
from scripts.data_loading import (
    load_tva_gic,
    load_tva_magnetometer,
    find_closest_magnetometers_to_gic,
    get_selected_sites,
)
from scripts.tf_models import model3_td, model3_fd, model4_fd, heyns_td, heyns_ensemble
from scripts.nn_models import model5a_cnn, model5b_gru

logger = setup_logger(name="tfgic.run_regression")

TRAIN_SLICE = slice(
    np.datetime64("2024-05-10T15:00:00"), np.datetime64("2024-05-11T06:00:00")
)
TEST_SLICE = slice(
    np.datetime64("2024-05-11T06:00:00"), np.datetime64("2024-05-11T18:00:00")
)

def _to_list(arr):
    """Convert a numpy array or list to a plain Python list."""
    return arr.tolist() if hasattr(arr, "tolist") else list(arr)


def run_all_models():
    """Run all models across selected sites; return results dict and GIC dataset."""
    logger.info("Loading data.")
    tva_gic  = load_tva_gic()
    tva_mag  = load_tva_magnetometer()
    sites    = get_selected_sites()
    site_rel = find_closest_magnetometers_to_gic(gic_data=tva_gic, mag_data=tva_mag)

    results = {}
    for site in sites:
        logger.info(f"Processing site: {site}")
        site_results = {}

        models = [
            ("3_td",      lambda: model3_td(site, tva_gic, tva_mag, site_rel, TRAIN_SLICE, TEST_SLICE)),
            ("3_fd",      lambda: model3_fd(site, tva_gic, tva_mag, site_rel, TRAIN_SLICE, TEST_SLICE)),
            ("4",         lambda: model4_fd(site, tva_gic, tva_mag, site_rel, TRAIN_SLICE, TEST_SLICE)),
            ("heyns_td",  lambda: heyns_td(site, tva_gic, tva_mag, site_rel, TRAIN_SLICE, TEST_SLICE, subsample_step=10)),
            ("heyns_ens", lambda: heyns_ensemble(site, tva_gic, tva_mag, site_rel, TRAIN_SLICE, TEST_SLICE)),
            ("5a",        lambda: model5a_cnn(site, tva_mag, tva_gic, site_rel, TRAIN_SLICE, TEST_SLICE, epochs=40)),
            ("5b",        lambda: model5b_gru(site, tva_mag, tva_gic, site_rel, TRAIN_SLICE, TEST_SLICE, epochs=40)),
        ]

        timings = {}
        for name, fn in models:
            try:
                t0 = time.time()
                site_results[name] = fn()
                timings[name] = time.time() - t0
            except Exception as e:
                logger.error(f"{site} / {name} failed: {e}", exc_info=True)

        for name, res in site_results.items():
            logger.info(f"  {name:<12} PE={res['pe']:.4f}  ({timings[name]:.1f}s)")

        results[site] = site_results

    return results, tva_gic


def _build_json_entry(model_name, res):
    """Serialise a single model result dict to a JSON-safe entry."""
    pred = res.get("pred", res.get("predictions", []))
    obs  = res.get("measured", res.get("observations", []))

    entry = {
        "pe":           float(res["pe"]),
        "predictions":  _to_list(pred),
        "observations": _to_list(obs),
    }

    if "test_times" in res:
        entry["test_times"] = [str(t) for t in _to_list(res["test_times"])]

    if "pred_lower" in res and "pred_upper" in res:
        entry["confidence_bounds"] = {
            "lower": _to_list(res["pred_lower"]),
            "upper": _to_list(res["pred_upper"]),
        }

    if model_name == "heyns_td":
        entry["ensemble_size"] = int(res["ensemble_size"])
        entry["alpha_median"]  = float(res["coeffs"][0])
        entry["beta_median"]   = float(res["coeffs"][1])
        if "alpha_iqr" in res:
            entry["coefficient_bounds"] = {
                "alpha_iqr": _to_list(res["alpha_iqr"]),
                "beta_iqr":  _to_list(res["beta_iqr"]),
            }
        if "sampling_info" in res:
            entry["sampling_info"] = res["sampling_info"]

    elif model_name == "heyns_ens":
        entry["n_windows"]         = int(res["n_windows"])
        entry["n_frequency_bands"] = int(res["n_frequency_bands"])
        if "ensemble_sizes" in res:
            entry["ensemble_sizes"] = {
                (f"{k[0]:.3f}-{k[1]:.3f}Hz" if isinstance(k, tuple) else str(k)): int(v)
                for k, v in res["ensemble_sizes"].items()
            }

    elif model_name in ("3_td", "3_fd", "4"):
        if "coeffs" in res and len(res["coeffs"]) == 2:
            entry["coefficients"] = {"alpha": float(res["coeffs"][0]), "beta": float(res["coeffs"][1])}
        if "transfer" in res:
            entry["n_frequency_bands"] = len(res["transfer"])

    for metric in ("rmse", "mae", "r2"):
        if metric in res:
            entry[metric] = float(res[metric])

    return entry


def save_results(results):
    """Persist results as pickle, JSON, CSV summary, and run metadata."""
    RESULTS_DIR.mkdir(exist_ok=True)

    # Raw pickle
    pkl_path = RESULTS_DIR / "regression_models_results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Saved pickle to {pkl_path}")

    # JSON
    json_results = {
        site: {name: _build_json_entry(name, res) for name, res in site_res.items()}
        for site, site_res in results.items()
    }
    json_path = RESULTS_DIR / "regression_models_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"Saved JSON to {json_path}")

    # PE summary CSV
    model_keys = ["3_td", "3_fd", "4", "heyns_td", "heyns_ens", "5a", "5b"]
    col_names  = {"3_td": "3_TD", "3_fd": "3_FD", "4": "4",
                  "heyns_td": "Heyns_TD", "heyns_ens": "Heyns_Ens",
                  "5a": "5a_CNN", "5b": "5b_GRU"}

    rows = []
    for site, site_res in results.items():
        row = {"Site": site}
        for k in model_keys:
            if k in site_res:
                row[col_names[k]] = site_res[k]["pe"]
        rows.append(row)

    df      = pd.DataFrame(rows)
    avg_row = {"Site": "Average", **{c: df[c].mean() for c in df.columns if c != "Site"}}
    df      = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    csv_path = RESULTS_DIR / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved summary CSV to {csv_path}")
    logger.info("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Metadata
    metadata = {
        "run_timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_period":   f"{TRAIN_SLICE.start} to {TRAIN_SLICE.stop}",
        "test_period":    f"{TEST_SLICE.start} to {TEST_SLICE.stop}",
        "sites":          list(results.keys()),
        "models":         model_keys,
        "confidence_methods": {
            "3_td": "bootstrap", "3_fd": "bootstrap", "4": "bootstrap",
            "heyns_td": "ensemble_iqr", "heyns_ens": "ensemble_spread",
            "5a": "model_ensemble", "5b": "model_ensemble",
        },
    }
    meta_path = RESULTS_DIR / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")

    return df


def main():
    """Run all models, save results, and log total elapsed time."""
    t0 = time.time()
    results, _ = run_all_models()
    save_results(results)
    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s ({elapsed / 60:.1f} min).")
    return results


if __name__ == "__main__":
    main()