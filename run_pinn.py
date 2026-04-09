"""
Author: Dennies Bor
Role:   Entry point for PINN training and temporal split evaluation.
"""

import time
import sys
import numpy as np
import geopandas as gpd

import bezpy

from config.settings import setup_logger, get_data_dir, MODELS_DIR, RESULTS_DIR
from scripts.data_loading import (
    load_tva_gic,
    load_tva_magnetometer,
    find_closest_magnetometers_to_gic,
    get_selected_sites,
    load_and_process_transmission_lines,
)
from scripts.pinn import (
    train_on_all_yards,
    train_temporal_split,
)
from utils.geo import prepare_gic_gdf, get_intersecting_transmission_lines
from utils.signal_processing import bxby_to_ExEy, fix_nans, clean_timeseries, patch_bezpy_tl

logger = setup_logger(name="tfgic.run_pinn")

TRAINING_YARDS = [
    "Bull Run", "Paradise", "Union",
    "Raccoon Mountain", "Shelby", "Widows Creek 2",
]

VOLTAGE_CACHE = get_data_dir() / "array_vs.npy"


def load_and_patch_data():
    """Load all datasets and apply bezpy patches."""
    patch_bezpy_tl()
    logger.info("Loading GIC and magnetometer data.")
    tva_gic  = load_tva_gic()
    tva_mag  = load_tva_magnetometer()
    tl_gdf   = load_and_process_transmission_lines()
    site_rel = find_closest_magnetometers_to_gic(gic_data=tva_gic, mag_data=tva_mag)
    return tva_gic, tva_mag, tl_gdf, site_rel


def process_geographic_data(site_rel, tl_gdf):
    """Build GIC GeoDataFrame and find intersecting transmission lines."""
    gic_gdf = prepare_gic_gdf(site_rel)
    return get_intersecting_transmission_lines(gic_gdf, tl_gdf, buffer_distance=150)


def extract_mt_sites_and_calculate_efield(site_rel, tva_mag):
    """Compute MT-site E-field predictions for all sites in the cluster mapping."""
    logger.info("Calculating E-field predictions.")
    mt_sites, site_xys = [], []

    for values in site_rel.values():
        for sm in values["mt_cluster"]:
            mt_sites.append(sm["mt_site"])
            site_xys.append((sm["mt_site"].latitude, sm["mt_site"].longitude))

    site_xys  = np.array(site_xys)
    first_mag = list(site_rel.values())[0]["magnetometer"]
    T         = len(tva_mag.sel(device=first_mag).Bx.values)
    E_pred    = np.zeros((T, len(mt_sites), 2))

    for i, mt in enumerate(mt_sites):
        for values in site_rel.values():
            for sm in values["mt_cluster"]:
                if sm["mt_site"] is mt:
                    mag  = sm["closest_mag"]
                    Ex, Ey = mt.convolve_fft(
                        tva_mag.sel(device=mag).Bx.values,
                        tva_mag.sel(device=mag).By.values,
                        dt=1.0,
                    )
                    n = min(T, len(Ex))
                    E_pred[:n, i, 0] = Ex[:n]
                    E_pred[:n, i, 1] = Ey[:n]
                    break

    return E_pred, site_xys, mt_sites


def process_transmission_lines(intersections_gdf, site_xys):
    """Set Delaunay and NN weights on transmission lines; flag invalid ones."""
    logger.info("Processing transmission lines.")
    tl = intersections_gdf.to_crs(epsg=4326).rename(columns={"geometry_left": "geometry"})
    tl.set_geometry("geometry", inplace=True)
    tl["obj"]    = tl.apply(bezpy.tl.TransmissionLine, axis=1)
    tl["length"] = tl.obj.apply(lambda x: x.length)

    E_test = np.ones((1, len(site_xys), 2))
    tl.obj.apply(lambda x: x.set_delaunay_weights(site_xys))
    tl.obj.apply(lambda x: x.set_nearest_sites(site_xys))

    def _calc(obj, how):
        try:
            return obj.calc_voltages(E_test, how=how)
        except Exception:
            return np.nan

    valid_d = ~np.isnan([_calc(o, "delaunay") for o in tl.obj])
    valid_n = ~np.isnan([_calc(o, "nn")       for o in tl.obj])

    logger.info(f"Delaunay valid: {valid_d.sum()}/{len(tl)}")
    logger.info(f"NN valid: {valid_n.sum()}/{len(tl)}")

    tl["method"] = "invalid"
    tl.loc[valid_d,              "method"] = "delaunay"
    tl.loc[~valid_d & valid_n,   "method"] = "nn"

    return tl, tl[tl["method"] != "invalid"]


def calculate_voltage_components(E_pred, trans_lines_gdf):
    """Load cached voltage array or compute and cache it."""
    if VOLTAGE_CACHE.exists():
        logger.info(f"Loading voltage data from {VOLTAGE_CACHE}.")
        return np.load(VOLTAGE_CACHE)

    logger.info("Computing voltage components.")
    array_vs = np.zeros((E_pred.shape[0], len(trans_lines_gdf), 2))

    for i, (_, row) in enumerate(trans_lines_gdf.iterrows()):
        if row["method"] == "invalid":
            array_vs[:, i, :] = np.nan
            continue
        try:
            Vx, Vy = row.obj.calc_voltage_components(E_pred, how=row["method"])
            array_vs[:, i, 0] = Vx
            array_vs[:, i, 1] = Vy
        except Exception as e:
            logger.error(f"Voltage error on line {i}: {e}")
            array_vs[:, i, :] = np.nan

    np.save(VOLTAGE_CACHE, array_vs)
    logger.info(f"Saved voltage data to {VOLTAGE_CACHE}.")
    return array_vs


def create_line_to_yard_mapping(trans_lines_gdf_valid):
    """Return array mapping each valid transmission line to its GIC yard."""
    line2yard = np.array([row["device"] for _, row in trans_lines_gdf_valid.iterrows()])
    for yard in TRAINING_YARDS:
        logger.info(f"  {yard}: {(line2yard == yard).sum()} lines")
    return line2yard


def clean_data_and_find_common_time(tva_gic, tva_mag):
    """Clean NaN gaps and return time masks aligned to the common time window."""
    logger.info("Cleaning magnetometer and GIC data.")
    for v in ("Bx", "By"):
        clean_timeseries(tva_mag, v, max_gap=60)
    for dev in tva_gic.device.values:
        clean_timeseries(tva_gic.sel(device=dev), "gic", max_gap=60)

    common_start = max(tva_gic.time.values[0],  tva_mag.time.values[0])
    common_end   = min(tva_gic.time.values[-1], tva_mag.time.values[-1])
    logger.info(f"Common time range: {common_start} to {common_end}.")

    gic_mask = (tva_gic.time >= common_start) & (tva_gic.time <= common_end)
    mag_mask = (tva_mag.time >= common_start) & (tva_mag.time <= common_end)
    n_gic, n_mag = gic_mask.sum().item(), mag_mask.sum().item()

    if n_gic != n_mag:
        logger.warning(f"Length mismatch after time filter — GIC: {n_gic}, Mag: {n_mag}.")

    return gic_mask, mag_mask, min(n_gic, n_mag)


def create_feature_tensor_builder(
    site_rel, tva_mag, tva_gic,
    gic_mask, mag_mask, common_length,
    line2yard, array_vs, trans_lines_gdf_valid,
):
    """Return a closure that builds (X, y, E_ref) for a given yard."""

    def build_feature_tensor(yard):
        info    = site_rel[yard]
        mag     = info["magnetometer"]
        Bx      = tva_mag.sel(device=mag).Bx.where(mag_mask, drop=True).values.astype("f4")
        By      = tva_mag.sel(device=mag).By.where(mag_mask, drop=True).values.astype("f4")
        y       = tva_gic.gic.sel(device=yard).where(gic_mask, drop=True).values.astype("f4")
        L       = min(common_length, len(Bx), len(y))
        Bx, By, y = Bx[:L], By[:L], y[:L]

        for arr, nm in ((Bx, "Bx"), (By, "By"), (y, "GIC")):
            if np.isnan(arr).any():
                fix_nans(arr)

        Ex_ref, Ey_ref = bxby_to_ExEy(info["mt_site"], Bx, By, dt=1.0)

        absB  = np.sqrt(Bx**2 + By**2)
        dBx   = np.gradient(Bx)
        dBy   = np.gradient(By)
        kern  = np.ones(30, "f4") / 30
        stdBx = np.sqrt(np.clip(np.convolve(Bx**2, kern, "same") - np.convolve(Bx, kern, "same") ** 2, 0, None))
        stdBy = np.sqrt(np.clip(np.convolve(By**2, kern, "same") - np.convolve(By, kern, "same") ** 2, 0, None))

        mask = np.array([yy == yard for yy in line2yard])
        if mask.any():
            yard_lines  = trans_lines_gdf_valid.loc[mask]
            voltages    = yard_lines["VOLTAGE"].fillna(138.0).values
            w_v         = voltages / 500.0
            Vx_w        = np.nansum(array_vs[:L, mask, 0] * w_v, axis=1)
            Vy_w        = np.nansum(array_vs[:L, mask, 1] * w_v, axis=1)
            avg_v, max_v = voltages.mean(), voltages.max()
            hv_count    = (voltages >= 345).sum()
            v_div       = voltages.std()
            n_lines     = mask.sum()
            L_tot       = yard_lines["length"].sum()
        else:
            Vx_w = Vy_w = np.zeros_like(Bx)
            avg_v = max_v = 138.0
            hv_count = n_lines = 0
            v_div = L_tot = 0.0

        X = np.column_stack([
            Bx, By, absB, dBx, dBy, stdBx, stdBy, Vx_w, Vy_w,
            np.full_like(Bx, avg_v,    "f4"),
            np.full_like(Bx, max_v,    "f4"),
            np.full_like(Bx, hv_count, "f4"),
            np.full_like(Bx, v_div,    "f4"),
            np.full_like(Bx, n_lines,  "f4"),
            np.full_like(Bx, L_tot,    "f4"),
        ])
        logger.debug(f"{yard}: {n_lines} lines, avg={avg_v:.0f}kV, max={max_v:.0f}kV, EHV={hv_count}")
        return X, y, np.column_stack([Ex_ref, Ey_ref])

    return build_feature_tensor


def run_temporal_training(build_feature_tensor_fn):
    """Train each yard on days 1-2 and evaluate on day 3."""
    logger.info("Starting temporal split training.")
    results = {}
    for yard in TRAINING_YARDS:
        try:
            res           = train_temporal_split(yard=yard, build_feature_tensor_fn=build_feature_tensor_fn, win=256, batch=1024, epochs=40)
            results[yard] = res
            logger.info(f"{yard}: PE={res['PE']:.3f}, RMSE={res['RMSE']:.3f}")
        except Exception as e:
            logger.error(f"{yard}: training failed — {e}")
            results[yard] = None
    return results


def print_training_summary(results):
    """Log a formatted PE/RMSE/Correlation table."""
    header = f"{'Site':15s} | {'PE':>8s} | {'RMSE':>8s} | {'Corr':>8s}"
    logger.info(header)
    pe_values = []
    for site, res in results.items():
        if res is None:
            logger.info(f"{site:15s} | {'failed':>8s}")
            continue
        logger.info(f"{site:15s} | {res['PE']:8.3f} | {res['RMSE']:8.3f} | {res['Correlation']:8.3f}")
        pe_values.append(res["PE"])
    logger.info(f"{'Average':15s} | {np.nanmean(pe_values):8.3f}")


def main():
    t0 = time.time()
    try:
        tva_gic, tva_mag, tl_gdf, site_rel = load_and_patch_data()
        intersections_gdf                   = process_geographic_data(site_rel, tl_gdf)
        E_pred, site_xys, _                 = extract_mt_sites_and_calculate_efield(site_rel, tva_mag)
        trans_lines_gdf, tl_valid           = process_transmission_lines(intersections_gdf, site_xys)
        array_vs                            = calculate_voltage_components(E_pred, trans_lines_gdf)
        line2yard                           = create_line_to_yard_mapping(tl_valid)
        gic_mask, mag_mask, common_length   = clean_data_and_find_common_time(tva_gic, tva_mag)

        build_ft = create_feature_tensor_builder(
            site_rel, tva_mag, tva_gic,
            gic_mask, mag_mask, common_length,
            line2yard, array_vs, tl_valid,
        )

        results = run_temporal_training(build_ft)
        print_training_summary(results)
        logger.info(f"Done in {time.time() - t0:.1f}s.")
        return results

    except Exception as e:
        logger.exception(f"PINN training failed: {e}")
        return None


if __name__ == "__main__":
    results = main()
    if not results:
        sys.exit(1)