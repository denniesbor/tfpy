"""
Author: Dennies Bor
Role:   Entry point for generating all tfgic figures.
"""

import time

from config.settings import setup_logger
from scripts.data_loading import (
    load_tva_gic,
    load_tva_magnetometer,
    find_closest_magnetometers_to_gic,
    get_selected_sites,
    load_and_process_transmission_lines,
)
from viz.plot_raw_data import plot_raw_data
from viz.plot_sites import plot_sites
from viz.plot_predictions import plot_all_sites

logger = setup_logger(name="tfgic.run_viz")


def load_data():
    """Load all datasets needed for visualisation."""
    logger.info("Loading data.")
    tva_gic  = load_tva_gic()
    tva_mag  = load_tva_magnetometer()
    sites    = get_selected_sites()
    tl_gdf   = load_and_process_transmission_lines()
    site_rel = find_closest_magnetometers_to_gic(gic_data=tva_gic, mag_data=tva_mag)
    return tva_gic, tva_mag, sites, tl_gdf, site_rel


def main():
    t0 = time.time()

    tva_gic, tva_mag, sites, tl_gdf, site_rel = load_data()

    logger.info("Plotting raw B/E/GIC data.")
    plot_raw_data(tva_gic, tva_mag, site_rel)

    logger.info("Plotting site map.")
    plot_sites(sites, tva_gic, tva_mag, site_rel, tl_gdf)

    logger.info("Plotting model predictions.")
    plot_all_sites(sites, tva_gic, site_rel)

    logger.info(f"All figures saved in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()