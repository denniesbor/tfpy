"""
Data loading functions for GIC analysis.
"""

import os
import gc
import geopandas as gpd
import bezpy
from pathlib import Path
import sys
import pickle
import numpy as np
import xarray as xr
from pathlib import Path
from config.settings import get_data_dir, setup_logger
from utils.geo import haversine_dist

logger = setup_logger(name="data_loader")


def load_mt_sites(filename=None):
    """Load MT sites from pickle file."""
    if filename is None:
        filename = get_data_dir() / "mt_pickle.pkl"

    if os.path.exists(filename):
        with open(filename, "rb") as pkl:
            return pickle.load(pkl)
    else:
        error_msg = f"MT sites pickle file {filename} not found."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    
def load_transmission_lines(filename=None):
    if filename is None:
        filename = get_data_dir() / "trans_lines_within_FERC_filtered.pkl"
    if os.path.exists(filename):
        with open(filename, "rb") as pkl:
            return pickle.load(pkl)
    else:
        error_msg = f"Transmission lines pickle file {filename} not found."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    

def load_and_process_transmission_lines(transmission_lines_path=None):
    """
    Load and process transmission line data, filtering for extra-high voltage
    (EHV) lines and associating them with FERC regions.
    """
    
    if transmission_lines_path is None:
        transmission_lines_path = get_data_dir() / "TL" / "Electric__Power_Transmission_Lines.shp"
    if not os.path.exists(transmission_lines_path):
        error_msg = f"Transmission lines shapefile {transmission_lines_path} not found."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
     
    gdf = gpd.read_file(transmission_lines_path).to_crs("EPSG:4326")
    gdf.rename(columns={"ID": "line_id"}, inplace=True)
    gdf = gdf.reset_index(drop=True).explode(index_parts=True).reset_index(level=1)
    gdf["line_id"] = gdf.apply(
        lambda row: (
            f"{row['line_id']}_{row['level_1']}"
            if row["level_1"] > 0
            else row["line_id"]
        ),
        axis=1,
    )
    
    # Filter for EHV lines first
    gdf = gdf[gdf["VOLTAGE"] >= 100].drop(columns=["level_1"])
    
    # Standardize voltages right after filtering
    line_voltage_ratings = {
        345: 345, 230: 230, 450: 500, 500: 500, 765: 765,
        250: 230, 400: 345, 232: 230, 1000: 765, 220: 230,
        273: 230, 218: 230, 236: 230, 287: 345, 238: 230, 200: 230,
    }
    
    gdf["VOLTAGE"] = gdf["VOLTAGE"].map(line_voltage_ratings).fillna(gdf["VOLTAGE"])
    
    gdf["length"] = gdf.apply(lambda row: bezpy.tl.TransmissionLine(row).length, axis=1)
    gc.collect()

    return gdf


def load_tva_gic(filename=None):
    """Load TVA GIC data."""
    if filename is None:
        filename = get_data_dir() / "tva_gic.nc"
    return xr.open_dataset(filename)


def load_tva_magnetometer(filename=None):
    """Load TVA magnetometer data."""
    if filename is None:
        filename = get_data_dir() / "tva_magnetometer_data.nc"
    return xr.open_dataset(filename)


def find_closest_mt_sites_to_magnetometers(mt_sites=None, mag_data=None):
    """Find closest MT site for each magnetometer."""
    if mt_sites is None:
        mt_sites = load_mt_sites()
    if mag_data is None:
        mag_data = load_tva_magnetometer()

    closest_sites = {}
    for device in mag_data.device.values:
        mag_lat = mag_data.sel(device=device).latitude.item()
        mag_lon = mag_data.sel(device=device).longitude.item()

        distances = [
            (i, site, haversine_dist(mag_lat, mag_lon, site.latitude, site.longitude))
            for i, site in enumerate(mt_sites)
        ]

        closest_idx, closest_site, min_dist = min(distances, key=lambda x: x[2])

        closest_sites[device] = {
            "site": closest_site,
            "site_idx": closest_idx,
            "distance": min_dist,
            "lat": mag_lat,
            "lon": mag_lon,
        }

        logger.info(
            f"Magnetometer {device}: Closest MT site is {closest_site.name} at {min_dist:.2f} km"
        )

    return closest_sites


def find_closest_magnetometers_to_mt_sites(
    mt_sites=None,
    mag_data=None,
):
    """
    For every MT site return its nearest TVA magnetometer.

    Returns
    -------
    dict
        {
          mt_name : {
              "magnetometer" : <str>,      # magnetometer device code
              "distance_km"  : <float>,    # centre-to-centre distance
              "mt_lat"       : <float>,
              "mt_lon"       : <float>,
              "mag_lat"      : <float>,
              "mag_lon"      : <float>,
          },
          ...
        }
    """
    if mt_sites is None:
        mt_sites = load_mt_sites()                # from data_loader
    if mag_data is None:
        mag_data = load_tva_magnetometer()

    mapping = {}
    for site in mt_sites:
        mt_lat, mt_lon = site.latitude, site.longitude

        dists = [
            (
                mag,
                haversine_dist(
                    mt_lat,
                    mt_lon,
                    mag_data.sel(device=mag).latitude.item(),
                    mag_data.sel(device=mag).longitude.item(),
                ),
            )
            for mag in mag_data.device.values
        ]
        closest_mag, min_dist = min(dists, key=lambda x: x[1])

        mapping[site.name] = {
            "magnetometer": closest_mag,
            "distance_km": min_dist,
            "mt_lat": mt_lat,
            "mt_lon": mt_lon,
            "mag_lat": mag_data.sel(device=closest_mag).latitude.item(),
            "mag_lon": mag_data.sel(device=closest_mag).longitude.item(),
        }

        logger.info(
            f"MT site {site.name}: nearest magnetometer {closest_mag} "
            f"({min_dist:.2f} km)."
        )

    return mapping


def find_closest_magnetometers_to_gic(
    gic_data=None,
    mag_data=None,
    closest_mag_sites=None,
    mt_sites=None,
    radius_km: float = 40.0,
):
    """
    For every GIC monitoring device, return:
        • nearest magnetometer  (name + distance)
        • primary MT site linked to that magnetometer
        • list of *all* MT sites within `radius_km` of the GIC yard  (mt_cluster)
    """
    # --- load defaults ----------------------------------------------------- #
    if gic_data is None:
        gic_data = load_tva_gic()
    if mag_data is None:
        mag_data = load_tva_magnetometer()
    if closest_mag_sites is None:
        closest_mag_sites = find_closest_mt_sites_to_magnetometers(mag_data=mag_data)
    if mt_sites is None:
        mt_sites = load_mt_sites()

    # ---------------------------------------------------------------------- #
    closest_to_gic = {}
    for device in gic_data.device.values:
        gic_lat = gic_data.sel(device=device).latitude.item()
        gic_lon = gic_data.sel(device=device).longitude.item()

        # ---------- nearest magnetometer ----------------------------------- #
        dists_mag = [
            (
                mag,
                haversine_dist(
                    gic_lat,
                    gic_lon,
                    mag_data.sel(device=mag).latitude.item(),
                    mag_data.sel(device=mag).longitude.item(),
                ),
            )
            for mag in mag_data.device.values
        ]
        closest_mag, min_dist = min(dists_mag, key=lambda x: x[1])
        closest_mt = closest_mag_sites[closest_mag]["site"]

        # ---------- MT cluster inside radius_km ---------------------------- #
        mt_cluster = [
            site
            for site in mt_sites
            if haversine_dist(gic_lat, gic_lon, site.latitude, site.longitude) <= radius_km
        ]
        if not mt_cluster:                       # ensure at least one site
            mt_cluster = [closest_mt]
            
        mt_cluster_with_mags = [
            {"mt_site": site, "closest_mag": find_closest_magnetometers_to_mt_sites(mt_sites=[site], mag_data=mag_data)[site.name]["magnetometer"]} 
            for site in mt_cluster
        ]

        # ---------- record ------------------------------------------------- #
        closest_to_gic[device] = {
            "magnetometer": closest_mag,
            "mt_site": closest_mt,
            "mt_cluster": mt_cluster_with_mags,
            "distance_to_mag": min_dist,
            "lat": gic_lat,
            "lon": gic_lon,
        }

        logger.info(
            f"GIC {device}: nearest magnetometer {closest_mag} "
            f"({min_dist:.2f} km), MT sites within {radius_km} km: "
            f"{', '.join(site.name for site in mt_cluster)}"
        )

    return closest_to_gic


def get_selected_sites():
    """Return list of selected sites for analysis."""

    # This will vary based on future analysis and working with NERC data or other datasets
    return ["Bull Run", "Paradise", "Union", "Raccoon Mountain"]