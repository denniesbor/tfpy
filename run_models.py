#!/usr/bin/env python
"""
Run all regression-based GIC prediction models and save results for exploration:
- Model 3: Transfer function (time domain and frequency domain)
- Model 4: Enhanced transfer function (frequency domain)  
- Model 5a: CNN ensemble
- Model 5b: GRU ensemble
- Heyns TD: Time-domain ensemble method
- Heyns Ensemble: B-field transfer function ensemble method
"""

import numpy as np
import pandas as pd
import time
import os
from pathlib import Path

from tqdm import tqdm

from config.settings import setup_logger
logger = setup_logger(name="tfgic.regression")

from scripts.data_loading import (
    load_tva_gic,
    load_tva_magnetometer,
    find_closest_magnetometers_to_gic,
    get_selected_sites,
)

from scripts.tf_models import model3_td, model3_fd, model4_fd, heyns_td, heyns_ensemble
from scripts.nn_models import model5a_cnn, model5b_gru

# Define time periods - same for all models
train_slice = slice(
    np.datetime64("2024-05-10T15:00:00"), np.datetime64("2024-05-11T06:00:00")
)
test_slice = slice(
    np.datetime64("2024-05-11T06:00:00"), np.datetime64("2024-05-11T18:00:00")
)

def run_all_models():
    """Run all regression models on selected sites."""
    logger.info("Loading data...")
    tva_gic = load_tva_gic()
    tva_mag = load_tva_magnetometer()
    selected_sites = get_selected_sites()
    closest_mag_to_gic = find_closest_magnetometers_to_gic(
        gic_data=tva_gic, mag_data=tva_mag
    )
    
    results = {}
    for site in selected_sites:
        site_results = {}
        logger.info(f"Processing site: {site}")
        
        try:
            logger.info("  Running Model 3 (time domain)...")
            start_time = time.time()
            m3_td = model3_td(
                site, tva_gic, tva_mag, closest_mag_to_gic, train_slice, test_slice
            )
            m3_td_time = time.time() - start_time
            site_results["3_td"] = m3_td
            
            logger.info("  Running Model 3 (frequency domain)...")
            start_time = time.time()
            m3_fd = model3_fd(
                site, tva_gic, tva_mag, closest_mag_to_gic, train_slice, test_slice
            )
            m3_fd_time = time.time() - start_time
            site_results["3_fd"] = m3_fd
            
            logger.info("  Running Model 4...")
            start_time = time.time()
            m4 = model4_fd(
                site, tva_gic, tva_mag, closest_mag_to_gic, train_slice, test_slice
            )
            m4_time = time.time() - start_time
            site_results["4"] = m4
            
            logger.info("  Running Heyns TD (time domain ensemble)...")
            start_time = time.time()
            heyns_td_result = heyns_td(
                site, tva_gic, tva_mag, closest_mag_to_gic, train_slice, test_slice, subsample_step=10
            )
            heyns_td_time = time.time() - start_time
            site_results["heyns_td"] = heyns_td_result
            
            logger.info("  Running Heyns Ensemble (B-field TF ensemble)...")
            start_time = time.time()
            heyns_ens_result = heyns_ensemble(
                site, tva_gic, tva_mag, closest_mag_to_gic, train_slice, test_slice, 
            )
            heyns_ens_time = time.time() - start_time
            site_results["heyns_ens"] = heyns_ens_result
            
            logger.info("  Running Model 5a (CNN)...")
            start_time = time.time()
            m5a = model5a_cnn(
                site, tva_mag, tva_gic, closest_mag_to_gic, train_slice, test_slice, epochs=40
            )
            m5a_time = time.time() - start_time
            site_results["5a"] = m5a
            
            logger.info("  Running Model 5b (GRU)...")
            start_time = time.time()
            m5b = model5b_gru(
                site, tva_mag, tva_gic, closest_mag_to_gic, train_slice, test_slice, epochs=40
            )
            m5b_time = time.time() - start_time
            site_results["5b"] = m5b
            
            logger.info(f"Performance metrics for {site}:")
            logger.info(f"  Model 3 (TD):     PE={m3_td['pe']:.4f}  ({m3_td_time:.1f}s)")
            logger.info(f"  Model 3 (FD):     PE={m3_fd['pe']:.4f}  ({m3_fd_time:.1f}s)")
            logger.info(f"  Model 4:          PE={m4['pe']:.4f}  ({m4_time:.1f}s)")
            logger.info(f"  Heyns TD:         PE={heyns_td_result['pe']:.4f}  ({heyns_td_time:.1f}s, n={heyns_td_result['ensemble_size']})")
            logger.info(f"  Heyns Ensemble:   PE={heyns_ens_result['pe']:.4f}  ({heyns_ens_time:.1f}s, n_win={heyns_ens_result['n_windows']})")
            logger.info(f"  Model 5a (CNN):   PE={m5a['pe']:.4f}  ({m5a_time:.1f}s)")
            logger.info(f"  Model 5b (GRU):   PE={m5b['pe']:.4f}  ({m5b_time:.1f}s)")
            
            results[site] = site_results
            
        except Exception as e:
            logger.error(f"Error processing site {site}: {str(e)}", exc_info=True)
    
    return results, tva_gic


def save_results(results):
    """Save results in multiple formats for exploration."""
    logger.info("Saving results for later analysis")
    
    os.makedirs("results", exist_ok=True)
    
    import pickle
    pickle_path = Path("results") / "regression_models_results.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Saved raw results to {pickle_path}")
    
    import json
    json_results = {}
    for site, site_results in results.items():
        json_results[site] = {}
        for model_name, model_result in site_results.items():
            json_entry = {
                "pe": float(model_result["pe"]),
                "predictions": model_result.get("pred", model_result.get("predictions", [])).tolist() if hasattr(model_result.get("pred", model_result.get("predictions", [])), "tolist") else list(model_result.get("pred", model_result.get("predictions", []))),
                "observations": model_result.get("measured", model_result.get("observations", [])).tolist() if hasattr(model_result.get("measured", model_result.get("observations", [])), "tolist") else list(model_result.get("measured", model_result.get("observations", []))),
            }
            
            # Add test_times if available (all models should have this now)
            if "test_times" in model_result:
                test_times = model_result["test_times"]
                if hasattr(test_times, "tolist"):
                    json_entry["test_times"] = [str(t) for t in test_times.tolist()]
                else:
                    json_entry["test_times"] = [str(t) for t in test_times]
            
            # Add confidence bounds if available (all models should have this now)
            if "pred_lower" in model_result and "pred_upper" in model_result:
                pred_lower = model_result["pred_lower"]
                pred_upper = model_result["pred_upper"]
                json_entry["confidence_bounds"] = {
                    "lower": pred_lower.tolist() if hasattr(pred_lower, "tolist") else list(pred_lower),
                    "upper": pred_upper.tolist() if hasattr(pred_upper, "tolist") else list(pred_upper)
                }
            
            if model_name == "heyns_td":
                json_entry["ensemble_size"] = int(model_result["ensemble_size"])
                json_entry["alpha_median"] = float(model_result["coeffs"][0])
                json_entry["beta_median"] = float(model_result["coeffs"][1])
                if "alpha_iqr" in model_result:
                    json_entry["coefficient_bounds"] = {
                        "alpha_iqr": model_result["alpha_iqr"].tolist(),
                        "beta_iqr": model_result["beta_iqr"].tolist()
                    }
                if "sampling_info" in model_result:
                    json_entry["sampling_info"] = model_result["sampling_info"]
                    
            elif model_name == "heyns_ens":
                json_entry["n_windows"] = int(model_result["n_windows"])
                json_entry["n_frequency_bands"] = int(model_result["n_frequency_bands"])
                if "ensemble_sizes" in model_result:
                    ensemble_sizes = {}
                    for k, v in model_result["ensemble_sizes"].items():
                        if isinstance(k, tuple):
                            key_str = f"{k[0]:.3f}-{k[1]:.3f}Hz"
                        else:
                            key_str = str(k)
                        ensemble_sizes[key_str] = int(v)
                    json_entry["ensemble_sizes"] = ensemble_sizes
                    
            elif model_name in ["3_td", "3_fd", "4"]:
                if "coeffs" in model_result:
                    if len(model_result["coeffs"]) == 2:  # Time domain (3_td)
                        json_entry["coefficients"] = {
                            "alpha": float(model_result["coeffs"][0]),
                            "beta": float(model_result["coeffs"][1])
                        }
                if "transfer" in model_result:  # Frequency domain models
                    json_entry["n_frequency_bands"] = len(model_result["transfer"])
                    
            elif model_name in ["5a", "5b"]:
                if "model_info" in model_result:
                    json_entry["model_info"] = model_result["model_info"]
                if "training_history" in model_result:
                    # Save just final metrics from training
                    history = model_result["training_history"]
                    if isinstance(history, dict) and "loss" in history:
                        json_entry["final_training_loss"] = float(history["loss"][-1])
                        if "val_loss" in history:
                            json_entry["final_validation_loss"] = float(history["val_loss"][-1])
            
            if "rmse" in model_result:
                json_entry["rmse"] = float(model_result["rmse"])
            if "mae" in model_result:
                json_entry["mae"] = float(model_result["mae"])
            if "r2" in model_result:
                json_entry["r2"] = float(model_result["r2"])
                
            json_results[site][model_name] = json_entry
    
    json_path = Path("results") / "regression_models_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"Saved JSON results to {json_path}")
    
    summary_data = []
    for site in results:
        row = {'Site': site}
        site_results = results[site]
        
        if '3_td' in site_results:
            row['3_TD'] = site_results['3_td']['pe']
        if '3_fd' in site_results:
            row['3_FD'] = site_results['3_fd']['pe']
        if '4' in site_results:
            row['4'] = site_results['4']['pe']
        if 'heyns_td' in site_results:
            row['Heyns_TD'] = site_results['heyns_td']['pe']
        if 'heyns_ens' in site_results:
            row['Heyns_Ens'] = site_results['heyns_ens']['pe']
        if '5a' in site_results:
            row['5a_CNN'] = site_results['5a']['pe']
        if '5b' in site_results:
            row['5b_GRU'] = site_results['5b']['pe']
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    avg_row = {'Site': 'Average'}
    for col in df.columns:
        if col != 'Site':
            avg_row[col] = df[col].mean()
    
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    
    csv_path = Path("results") / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved summary table to {csv_path}")
    
    logger.info("\nModel Performance Summary:")
    logger.info("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Create enhanced metadata file with confidence bounds info
    metadata = {
        "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_period": f"{train_slice.start} to {train_slice.stop}",
        "test_period": f"{test_slice.start} to {test_slice.stop}",
        "sites_processed": list(results.keys()),
        "models_run": ["3_td", "3_fd", "4", "heyns_td", "heyns_ens", "5a_cnn", "5b_gru"],
        "total_sites": len(results),
        "features_included": [
            "predictions", 
            "observations", 
            "test_times", 
            "confidence_bounds",
            "prediction_efficiency_scores"
        ],
        "confidence_methods": {
            "3_td": "bootstrap_resampling",
            "3_fd": "bootstrap_resampling", 
            "4": "bootstrap_resampling",
            "heyns_td": "ensemble_iqr",
            "heyns_ens": "ensemble_spread",
            "5a_cnn": "model_ensemble",
            "5b_gru": "model_ensemble"
        },
        "notes": "Regression model comparison with confidence bounds and exact time alignment"
    }
    
    metadata_path = Path("results") / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved run metadata to {metadata_path}")
    
    logger.info("Results saved with confidence bounds - ready for exploration in Jupyter notebook!")
    
    return df

def main():
    """Main function to run all models and save results."""
    logger.info("Starting GIC model comparison (including Heyns ensemble methods)")
    
    start_time = time.time()
    results, tva_gic = run_all_models()
    
    df = save_results(results)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    logger.info("GIC model comparison completed - check results/ directory")
    
    return results

if __name__ == "__main__":
    main()