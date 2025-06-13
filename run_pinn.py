#!/usr/bin/env python
"""
Training Physics-Informed Neural Networks (PINN) for GIC prediction.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from math import ceil

from config.settings import setup_logger, get_data_dir

logger = setup_logger(name="logs/pinn")

from scripts.data_loading import (
    load_tva_gic,
    load_tva_magnetometer,
    find_closest_magnetometers_to_gic,
    get_selected_sites,
    load_and_process_transmission_lines
)

from utils.geo import (
    prepare_gic_gdf,
    get_intersecting_transmission_lines
)

from utils.signal_processing import (
    bxby_to_ExEy,
    fix_nans,
    clean_timeseries,
    patch_bezpy_tl
)

from scripts.pinn import (
    GICPINN,
    WindowDS,
    train_pinn,
    evaluate_model,
    leave_one_yard_out,
    train_on_all_yards,
    train_temporal_split
)

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)


def load_and_patch_data():
    """
    Load initial data and apply bezpy patch.
    
    """
    # Apply bezpy TransmissionLine patch
    logger.info("Patching bezpy TransmissionLine class")
    patch_bezpy_tl()
    
    logger.info("Loading GIC and magnetometer data")
    tva_gic = load_tva_gic()
    tva_mag = load_tva_magnetometer()
    selected_sites = get_selected_sites()
    tl_gdf = load_and_process_transmission_lines()
    
    logger.info("Finding relationships between GIC monitors and magnetometers")
    closest_mag_to_gic = find_closest_magnetometers_to_gic(
        gic_data=tva_gic, mag_data=tva_mag
    )
    
    return tva_gic, tva_mag, selected_sites, tl_gdf, closest_mag_to_gic


def process_geographic_data(closest_mag_to_gic, tl_gdf):
    """
    Process geographic data to find transmission line intersections.
    """
    logger.info("Creating GeoDataFrame for GIC sites")
    gic_gdf = prepare_gic_gdf(closest_mag_to_gic)
    
    logger.info("Finding transmission lines intersecting with GIC sites")
    intersections_gdf = get_intersecting_transmission_lines(
        gic_gdf, 
        tl_gdf, 
        buffer_distance=150
    )
    
    return intersections_gdf


def extract_mt_sites_and_calculate_efield(closest_mag_to_gic, tva_mag):
    """
    Extract MT sites and calculate E-field predictions.
    """
    logger.info("Calculating E-field predictions")
    mt_sites = []
    site_xys = []

    # Collect all unique MT sites from the clusters
    for site, values in closest_mag_to_gic.items():
        mt_cluster = values['mt_cluster']
        
        for site_mag in mt_cluster:
            mt = site_mag['mt_site']
            mt_sites.append(mt)
            site_xys.append((mt.latitude, mt.longitude))

    site_xys = np.array(site_xys)

    # Get the first magnetometer to determine time series length
    first_mag = list(closest_mag_to_gic.values())[0]['magnetometer']
    time_steps = len(tva_mag.sel(device=first_mag).Bx.values)

    E_pred = np.zeros((time_steps, len(mt_sites), 2))

    # Calculate E for each MT site using its closest magnetometer
    for i, mt in enumerate(mt_sites):
        # Find the cluster this MT site belongs to
        for site, values in closest_mag_to_gic.items():
            mt_cluster = values['mt_cluster']
            
            for site_mag in mt_cluster:
                if site_mag['mt_site'] == mt:  # Found the right cluster entry
                    closest_mag = site_mag['closest_mag']
                    
                    site_Bx = tva_mag.sel(device=closest_mag).Bx.values
                    site_By = tva_mag.sel(device=closest_mag).By.values
                    
                    # Calculate E-field using the MT site's transfer function
                    Ex, Ey = mt.convolve_fft(site_Bx, site_By, dt=1.0)
                    
                    # Store in E_pred (ensuring dimensions match)
                    for t in range(min(len(E_pred), len(Ex))):
                        E_pred[t, i, 0] = Ex[t]
                        E_pred[t, i, 1] = Ey[t]
                    
                    break  # Found the matching entry
    
    return E_pred, site_xys, mt_sites


def process_transmission_lines(intersections_gdf, site_xys):
    """
    Process transmission lines and find valid calculation methods.
    """
    logger.info("Processing transmission lines")
    import bezpy
    
    # Apply crs and prepare GeoDataFrame
    trans_lines_gdf = intersections_gdf.to_crs(epsg=4326)
    trans_lines_gdf.rename(columns={"geometry_left": "geometry"}, inplace=True)
    trans_lines_gdf.set_geometry("geometry", inplace=True)
    trans_lines_gdf["obj"] = trans_lines_gdf.apply(bezpy.tl.TransmissionLine, axis=1)
    trans_lines_gdf["length"] = trans_lines_gdf.obj.apply(lambda x: x.length)

    E_test = np.ones((1, len(site_xys), 2))

    trans_lines_gdf.obj.apply(lambda x: x.set_delaunay_weights(site_xys))
    arr_delaunay = np.zeros(shape=(1, len(trans_lines_gdf)))
    for i, tLine in enumerate(trans_lines_gdf.obj):
        try:
            arr_delaunay[:, i] = tLine.calc_voltages(E_test, how="delaunay")
        except:
            arr_delaunay[:, i] = np.nan

    trans_lines_gdf.obj.apply(lambda x: x.set_nearest_sites(site_xys))
    arr_nn = np.zeros(shape=(1, len(trans_lines_gdf)))
    for i, tLine in enumerate(trans_lines_gdf.obj):
        try:
            arr_nn[:, i] = tLine.calc_voltages(E_test, how="nn")
        except:
            arr_nn[:, i] = np.nan

    valid_delaunay = ~np.isnan(arr_delaunay[0, :])
    valid_nn = ~np.isnan(arr_nn[0, :])
    valid_either = valid_delaunay | valid_nn

    logger.info(f"Delaunay valid: {valid_delaunay.sum()}/{len(trans_lines_gdf)}")
    logger.info(f"NN valid: {valid_nn.sum()}/{len(trans_lines_gdf)}")
    logger.info(f"Either method valid: {valid_either.sum()}/{len(trans_lines_gdf)}")

    # Filter using best method
    trans_lines_gdf['method'] = 'invalid'
    trans_lines_gdf.loc[valid_delaunay, 'method'] = 'delaunay'
    trans_lines_gdf.loc[(~valid_delaunay) & valid_nn, 'method'] = 'nn'

    trans_lines_gdf_valid = trans_lines_gdf[trans_lines_gdf['method'] != 'invalid']
    
    return trans_lines_gdf, trans_lines_gdf_valid


def calculate_voltage_components(E_pred, trans_lines_gdf):
    """
    Calculate voltage components for transmission lines.
    """
    logger.info("Calculating voltage components")
    data_path = "data/array_vs.npy"
    
    if os.path.exists(data_path):
        logger.info(f"Loading voltage data from {data_path}")
        array_vs = np.load(data_path)
    else:
        logger.info("Calculating voltage components from scratch...")
        array_vs = np.zeros(shape=(E_pred.shape[0], trans_lines_gdf.shape[0], 2))
        
        for i, tLine in enumerate(trans_lines_gdf.obj):
            method = trans_lines_gdf.iloc[i]['method']
            if method == 'invalid':
                array_vs[:, i, 0] = np.nan
                array_vs[:, i, 1] = np.nan
                continue
                
            try:
                Vx, Vy = tLine.calc_voltage_components(E_pred, how=method)
                array_vs[:, i, 0] = Vx
                array_vs[:, i, 1] = Vy
            except Exception as e:
                logger.error(f"Error calculating voltage for line {i}: {str(e)}")
                array_vs[:, i, 0] = np.nan
                array_vs[:, i, 1] = np.nan
        
        np.save(data_path, array_vs)
        logger.info(f"Saved voltage data to {data_path}")
    
    return array_vs


def create_line_to_yard_mapping(trans_lines_gdf_valid):
    """
    Create mapping between transmission lines and yards (GIC sites).
    """
    logger.info("Creating line to yard mapping")
    yards = ["Bull Run", "Paradise", "Raccoon Mountain", "Shelby", "Union", "Widows Creek 2"]
    line2yard = []

    for idx, row in trans_lines_gdf_valid.iterrows():
        line2yard.append(row['device'])

    line2yard = np.array(line2yard)

    logger.info(f"Number of lines mapped to yards: {len(line2yard)}")
    yard_counts = {yard: sum(line2yard == yard) for yard in yards}
    logger.info("Lines per yard:")
    for yard, count in yard_counts.items():
        logger.info(f"  {yard}: {count}")
    
    return line2yard


def clean_data_and_find_common_time(tva_gic, tva_mag):
    """
    Clean data and find common time range.
    
    Parameters
    ----------
    tva_gic : xarray.Dataset
        GIC data
    tva_mag : xarray.Dataset
        Magnetometer data
        
    Returns
    -------
    tuple
        (gic_mask, mag_mask, common_length) - Masks and common data length
    """
    logger.info("Cleaning magnetometer and GIC data")
    for v in ["Bx", "By"]:
        clean_timeseries(tva_mag, v, max_gap=60)  # ≤60 s gaps only

    for dev in tva_gic.device.values:
        clean_timeseries(tva_gic.sel(device=dev), "gic", max_gap=60)
        
    # Find common time range
    gic_start = tva_gic.time.values[0]
    gic_end = tva_gic.time.values[-1]
    mag_start = tva_mag.time.values[0]
    mag_end = tva_mag.time.values[-1]

    # Use the later start time and earlier end time
    common_start = max(gic_start, mag_start)
    common_end = min(gic_end, mag_end)

    logger.info(f"Common time range: {common_start} to {common_end}")

    gic_mask = (tva_gic.time >= common_start) & (tva_gic.time <= common_end)
    mag_mask = (tva_mag.time >= common_start) & (tva_mag.time <= common_end)

    gic_length = gic_mask.sum().item()
    mag_length = mag_mask.sum().item()
    
    # Make sure the lengths match
    if gic_length != mag_length:
        logger.warning(f"Data lengths don't match after time filtering! GIC: {gic_length}, Mag: {mag_length}")
        common_length = min(gic_length, mag_length)
        logger.info(f"Will use the shorter length: {common_length}")
    else:
        common_length = gic_length
        logger.info(f"Data lengths match: {common_length}")
    
    return gic_mask, mag_mask, common_length


def create_feature_tensor_builder(closest_mag_to_gic, tva_mag, tva_gic, gic_mask, mag_mask, 
                                 common_length, line2yard, array_vs, trans_lines_gdf_valid):
    """
    Create a feature tensor builder function with voltage-weighted features.
    """
    def build_feature_tensor(yard):
        """Build voltage-weighted feature tensor for PINN model from yard data"""
        info = closest_mag_to_gic[yard]
        mag_name = info["magnetometer"]

        Bx = tva_mag.sel(device=mag_name).Bx.where(mag_mask, drop=True).values.astype("f4")
        By = tva_mag.sel(device=mag_name).By.where(mag_mask, drop=True).values.astype("f4")
        y = tva_gic.gic.sel(device=yard).where(gic_mask, drop=True).values.astype("f4")

        L = min(common_length, len(Bx), len(y))
        Bx, By, y = Bx[:L], By[:L], y[:L]
        
        for a, nm in [(Bx,'Bx'), (By,'By'), (y,'GIC')]:
            if np.isnan(a).any():
                logger.debug(f"{yard}: filling NaNs in {nm}")
                fix_nans(a)

        mt_site = info["mt_site"]
        Ex_ref, Ey_ref = bxby_to_ExEy(mt_site, Bx, By, dt=1.0)

        # Basic magnetic field features
        absB = np.sqrt(Bx**2 + By**2)
        dBx, dBy = np.gradient(Bx), np.gradient(By)
        kern = np.ones(30, "f4") / 30
        stdBx = np.sqrt(np.clip(np.convolve(Bx**2, kern, "same") -
                                np.convolve(Bx, kern, "same")**2, 0, None))
        stdBy = np.sqrt(np.clip(np.convolve(By**2, kern, "same") -
                                np.convolve(By, kern, "same")**2, 0, None))

        mask = np.array([yy == yard for yy in line2yard])
        
        # VOLTAGE-WEIGHTED FEATURES - This is the key improvement!
        if mask.any():
            # Get voltage ratings for this yard's lines
            yard_lines = trans_lines_gdf_valid.loc[mask]
            voltage_ratings = yard_lines['VOLTAGE'].fillna(138.0).values  # Default 138kV if missing
            
            Vx_raw = array_vs[:L, mask, 0]  # Shape: (time, n_lines)
            Vy_raw = array_vs[:L, mask, 1]
            
            # Voltage-weighted sum (higher voltage lines contribute more)
            voltage_weights = voltage_ratings / 500.0  # Normalize to 500kV
            Vx_weighted = np.nansum(Vx_raw * voltage_weights, axis=1)
            Vy_weighted = np.nansum(Vy_raw * voltage_weights, axis=1)
            
            # System characteristics
            avg_voltage = voltage_ratings.mean()
            max_voltage = voltage_ratings.max()
            hv_line_count = (voltage_ratings >= 345).sum()  # EHV lines (345kV+)
            voltage_diversity = voltage_ratings.std()  # How mixed the voltage levels are
            
            # Line characteristics
            n_lines = mask.sum()
            L_tot = yard_lines["length"].sum()
            
        else:
            Vx_weighted = np.zeros_like(Bx)
            Vy_weighted = np.zeros_like(By)
            avg_voltage = 138.0  # Default
            max_voltage = 138.0
            hv_line_count = 0
            voltage_diversity = 0.0
            n_lines = 0
            L_tot = 0.0

        # Enhanced feature set (15 features instead of 11)
        X = np.column_stack([
                # Magnetic field features (7)
                Bx, By, absB, dBx, dBy, stdBx, stdBy,
                
                # Voltage-weighted transmission line features (4)
                Vx_weighted, Vy_weighted,  # Voltage-weighted voltages
                np.full_like(Bx, avg_voltage, "f4"),    # System avg voltage (repeated for each time step)
                np.full_like(Bx, max_voltage, "f4"),    # System max voltage (repeated for each time step)
                
                # System topology features (4)
                np.full_like(Bx, hv_line_count, "f4"),      # Number of EHV lines
                np.full_like(Bx, voltage_diversity, "f4"),   # Voltage level diversity
                np.full_like(Bx, n_lines, "f4"),            # Total line count
                np.full_like(Bx, L_tot, "f4")               # Total line length
            ])
        E_ref = np.column_stack([Ex_ref, Ey_ref])
        
        logger.debug(f"{yard}: {n_lines} lines, avg_V={avg_voltage:.0f}kV, "
                    f"max_V={max_voltage:.0f}kV, EHV_lines={hv_line_count}")
        
        return X, y, E_ref
    
    return build_feature_tensor


def run_training(build_feature_tensor_fn):
    """
    Run the training process.
    
    Parameters
    ----------
    build_feature_tensor_fn : function
        Function to build feature tensors
        
    Returns
    -------
    dict
        Training results
    """
    logger.info("Starting leave-one-yard-out cross-validation")
    training_yards = ["Bull Run", "Paradise", "Union", "Raccoon Mountain", "Shelby", "Widows Creek 2"]
    
    results = train_on_all_yards(
        yards=training_yards,
        build_feature_tensor_fn=build_feature_tensor_fn,
        win=256,
        batch=1024,
        epochs=40
    )
    
    return results


def run_temporal_training(build_feature_tensor_fn):
    """
    Run temporal split training - train on quiet periods, test on storm peak.
    
    Parameters
    ----------
    build_feature_tensor_fn : function
        Function to build feature tensors
        
    Returns
    -------
    dict
        Training results for each yard
    """
    logger.info("Starting temporal split training")
    training_yards = ["Bull Run", "Paradise", "Union", "Raccoon Mountain", "Shelby", "Widows Creek 2"]
    
    results = {}
    
    # Loop through each yard individually
    for yard in training_yards:
        logger.info(f"Training {yard} with temporal split...")
        
        try:
            # Train on this single yard with temporal split
            result = train_temporal_split(
                yard=yard,
                build_feature_tensor_fn=build_feature_tensor_fn,
                win=256,
                batch=1024,
                epochs=40
            )
            
            results[yard] = result
            logger.info(f"✓ {yard}: PE={result['PE']:.3f}, RMSE={result['RMSE']:.3f}")
            
        except Exception as e:
            logger.error(f"✗ {yard}: Error - {str(e)}")
            results[yard] = None
    
    return results



def print_training_summary(results):
    """
    Print a summary of training results.
    
    Parameters
    ----------
    results : dict
        Training results
    """
    logger.info("\nEvaluation Summary")
    logger.info("-" * 80)
    logger.info(f"{'Site':15s} | {'PE':>8s} | {'RMSE':>8s} | {'Corr':>8s}")
    logger.info("-" * 80)
    
    pe_values = []
    for site, metrics in results.items():
        pe = metrics['PE']
        rmse = metrics['RMSE']
        corr = metrics['Correlation']
        pe_values.append(pe)
        logger.info(f"{site:15s} | {pe:8.3f} | {rmse:8.3f} | {corr:8.3f}")
    
    mean_pe = np.nanmean(pe_values)
    logger.info("-" * 80)
    logger.info(f"{'Average':15s} | {mean_pe:8.3f} | {'-':8s} | {'-':8s}")


def main():
    """Main function to run the complete PINN training process."""
    start_time = time.time()
    
    try:
        # Step 1-3: Load data and apply patching
        tva_gic, tva_mag, selected_sites, tl_gdf, closest_mag_to_gic = load_and_patch_data()
        
        # Step 4-5: Process geographic data
        intersections_gdf = process_geographic_data(closest_mag_to_gic, tl_gdf)
        
        # Step 6: Calculate E-field prediction
        E_pred, site_xys, mt_sites = extract_mt_sites_and_calculate_efield(closest_mag_to_gic, tva_mag)
        
        # Step 7: Process transmission lines
        trans_lines_gdf, trans_lines_gdf_valid = process_transmission_lines(intersections_gdf, site_xys)
        
        # Step 8: Calculate voltage components
        array_vs = calculate_voltage_components(E_pred, trans_lines_gdf)
        
        # Step 9: Create line to yard mapping
        line2yard = create_line_to_yard_mapping(trans_lines_gdf_valid)
        
        # Step 10: Clean data and find common time range
        gic_mask, mag_mask, common_length = clean_data_and_find_common_time(tva_gic, tva_mag)
        
        # Step 11: Create feature tensor builder
        build_feature_tensor = create_feature_tensor_builder(
            closest_mag_to_gic, tva_mag, tva_gic, gic_mask, mag_mask,
            common_length, line2yard, array_vs, trans_lines_gdf_valid
        )
        
        # Step 12: Run training
        # results = run_training(build_feature_tensor)
        results = run_temporal_training(build_feature_tensor)
        
        # Step 13: Print training summary
        print_training_summary(results)
        
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
        
        return results
    
    except Exception as e:
        logger.exception(f"Error during PINN training: {str(e)}")
        return None


if __name__ == "__main__":
    try:
        results = main()
        if results:
            logger.info("PINN training completed successfully")
        else:
            logger.error("PINN training failed")
            sys.exit(1)
    except Exception as e:
        logger.exception(f"Unhandled error during execution: {str(e)}")
        sys.exit(1)