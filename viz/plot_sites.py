"""
Author: Dennies Bor
Role:   Map of TVA GIC monitors, magnetometers, MT stations and transmission lines.
"""

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from config.settings import setup_logger
from utils.geo import prepare_gic_gdf, get_intersecting_transmission_lines
from viz.plot_utils import setup_matplotlib, save_figure, add_checkered_border

logger = setup_logger(name="tfgic.plot_sites")

EXTENT       = [-90, -82, 34, 39]
VOLTAGE_COLS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def plot_sites(selected_sites, tva_gic, tva_mag, closest_mag_to_gic, tl_gdf):
    """Produce and save the GIC site map with transmission lines."""
    setup_matplotlib()

    site_rel_filt    = {k: v for k, v in closest_mag_to_gic.items() if k in selected_sites}
    gic_gdf          = prepare_gic_gdf(site_rel_filt)
    intersections_gdf = get_intersecting_transmission_lines(gic_gdf, tl_gdf, buffer_distance=150)

    if intersections_gdf.crs.to_epsg() != 4326:
        intersections_gdf = intersections_gdf.to_crs("EPSG:4326")

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES,    linewidth=0.8, edgecolor="gray")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    add_checkered_border(ax)

    voltages      = sorted(v for v in intersections_gdf["VOLTAGE"].unique() if pd.notna(v))
    voltage_colors = dict(zip(voltages, VOLTAGE_COLS[:len(voltages)]))

    for v in voltages:
        intersections_gdf[intersections_gdf["VOLTAGE"] == v].plot(
            ax=ax, color=voltage_colors[v], linewidth=1.5, alpha=0.7,
            transform=ccrs.PlateCarree(),
        )

    for site in selected_sites:
        info       = closest_mag_to_gic[site]
        mag        = tva_mag.sel(device=info["magnetometer"])
        mt         = info["mt_site"]

        ax.scatter(info["lon"],    info["lat"],    color="red",   marker="s", s=120, edgecolor="k", linewidth=1, zorder=5)
        ax.scatter(float(mag.longitude), float(mag.latitude), color="blue",  marker="o", s=100, edgecolor="k", linewidth=1, zorder=5)
        ax.scatter(mt.longitude,   mt.latitude,    color="green", marker="^", s=100, edgecolor="k", linewidth=1, zorder=5)

        ax.text(info["lon"],   info["lat"] + 0.3, f"{site} Station", ha="center", fontsize=10, fontweight="bold")
        ax.text(mt.longitude + 0.6, mt.latitude - 0.08, f"{mt.name} MT",  ha="center", fontsize=9)

    legend_elements = [
        plt.scatter([], [], color="red",   marker="s", s=120, edgecolor="k", label="GIC Monitor"),
        plt.scatter([], [], color="blue",  marker="o", s=100, edgecolor="k", label="Magnetometer"),
        plt.scatter([], [], color="green", marker="^", s=100, edgecolor="k", label="MT Station"),
        *[plt.plot([], [], color=voltage_colors[v], linewidth=2, label=f"{v} kV")[0] for v in voltages],
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=False, fontsize=9)

    plt.tight_layout()
    save_figure(fig, "sites")
    plt.close(fig)
    logger.info("Saved sites figure.")
    

if __name__ == "__main__":
    from scripts.data_loading import (
        load_tva_gic,
        load_tva_magnetometer,
        find_closest_magnetometers_to_gic,
        get_selected_sites,
        load_and_process_transmission_lines,
    )

    tva_gic  = load_tva_gic()
    tva_mag  = load_tva_magnetometer()
    sites    = get_selected_sites()
    tl_gdf   = load_and_process_transmission_lines()
    site_rel = find_closest_magnetometers_to_gic(gic_data=tva_gic, mag_data=tva_mag)

    plot_sites(sites, tva_gic, tva_mag, site_rel, tl_gdf)