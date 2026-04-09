"""
Author: Dennies Bor
Role:   Three-panel B/E/GIC plot for selected TVA sites.
"""

import numpy as np
import matplotlib.pyplot as plt

from config.settings import setup_logger
from utils.geo import haversine_dist
from viz.plot_utils import setup_matplotlib, save_figure, extract_site_data

logger = setup_logger(name="tfgic.plot_raw_data")

START_TIME = np.datetime64("2024-05-10T15:00:00")
END_TIME   = np.datetime64("2024-05-11T18:00:00")
SITES      = ["Bull Run", "Paradise", "Union", "Raccoon Mountain"]


def plot_single_site(axes, time_axis, Bx, By, Ex, Ey, gic_data, site, mt_site, closest_mag_to_gic):
    """Draw B, E and GIC panels for one site onto three provided axes."""
    ax1, ax2, ax3 = axes
    info = closest_mag_to_gic[site]

    B_mag = np.sqrt(Bx**2 + By**2)
    E_mag = np.sqrt(Ex**2 + Ey**2)

    ax1.plot(time_axis, B_mag, "b-", alpha=0.9, linewidth=0.8)
    ax1.set_ylabel("|B| (nT)", fontsize=9)

    ax2.plot(time_axis, E_mag, "r-", alpha=0.9, linewidth=0.8)
    ax2.set_ylabel("|E| (V/km)", fontsize=9)

    ax3.plot(gic_data.time, gic_data.values, "k-", alpha=0.9, linewidth=0.8)
    ax3.set_ylabel("GIC (A)", fontsize=9)
    ax3.set_xlabel("Time", fontsize=9)

    mag_dist = info["distance_to_mag"]
    mt_dist  = haversine_dist(info["lat"], info["lon"], mt_site.latitude, mt_site.longitude)

    ax1.text(0.02, 0.85, f"Mag: {mag_dist:.1f} km to {info['magnetometer']}", transform=ax1.transAxes, fontsize=8)
    ax2.text(0.02, 0.85, f"MT: {mt_site.name} {mt_dist:.1f} km",             transform=ax2.transAxes, fontsize=8)
    ax3.text(0.02, 0.85, f"GIC: {site}",                                      transform=ax3.transAxes, fontsize=8)
    ax2.text(1.02, 0.5,  site, transform=ax2.transAxes, rotation=90, va="center", fontsize=10)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, bottom=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    ax3.tick_params(labelsize=8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)


def plot_raw_data(tva_gic, tva_mag, closest_mag_to_gic):
    """Produce and save the multi-site B/E/GIC overview figure."""
    setup_matplotlib()
    fig, axes = plt.subplots(len(SITES) * 3, 1, figsize=(8, 9), sharex=True)

    for i, site in enumerate(SITES):
        logger.info(f"Plotting raw data for {site}.")
        time_axis, Bx, By, Ex, Ey, gic_data, mt_site = extract_site_data(
            site, START_TIME, END_TIME, tva_gic, tva_mag, closest_mag_to_gic
        )
        plot_single_site(
            (axes[i * 3], axes[i * 3 + 1], axes[i * 3 + 2]),
            time_axis, Bx, By, Ex, Ey, gic_data, site, mt_site, closest_mag_to_gic,
        )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    save_figure(fig, "raw_data")
    plt.close(fig)
    logger.info("Saved raw_data figure.")
    

if __name__ == "__main__":
    from scripts.data_loading import (
        load_tva_gic,
        load_tva_magnetometer,
        find_closest_magnetometers_to_gic,
    )

    tva_gic  = load_tva_gic()
    tva_mag  = load_tva_magnetometer()
    site_rel = find_closest_magnetometers_to_gic(gic_data=tva_gic, mag_data=tva_mag)

    plot_raw_data(tva_gic, tva_mag, site_rel)