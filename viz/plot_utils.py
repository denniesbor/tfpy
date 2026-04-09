"""
Author: Dennies Bor
Role:   Shared plotting utilities for the tfgic visualisation module.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import pandas as pd

from config.settings import FIGURES_DIR
from utils.signal_processing import tukey_window, taper
from utils.geo import haversine_dist


def setup_matplotlib():
    """Apply project-wide matplotlib style."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "serif"]


def save_figure(fig, name):
    """Save figure as both PNG and PDF to FIGURES_DIR."""
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")


def add_checkered_border(ax, n=25, bar=0.01, lw=0.3):
    """Draw a thin checkered border around a cartopy axes."""
    step = 1 / n
    trans = ax.transAxes
    for i in range(n):
        c = "k" if i % 2 else "w"
        ax.add_patch(
            patches.Rectangle(
                (i * step, 1 - bar), step, bar, fc=c, ec="k", lw=lw, transform=trans
            )
        )
        ax.add_patch(
            patches.Rectangle(
                (i * step, 0), step, bar, fc=c, ec="k", lw=lw, transform=trans
            )
        )
        ax.add_patch(
            patches.Rectangle(
                (0, i * step), bar, step, fc=c, ec="k", lw=lw, transform=trans
            )
        )
        ax.add_patch(
            patches.Rectangle(
                (1 - bar, i * step), bar, step, fc=c, ec="k", lw=lw, transform=trans
            )
        )


def extract_site_data(site, start_time, end_time, tva_gic, tva_mag, closest_mag_to_gic):
    """Extract B-field, E-field and GIC time series for a single site."""
    info = closest_mag_to_gic[site]
    mt_site = info["mt_site"]
    mag_data = tva_mag.sel(
        device=info["magnetometer"], time=slice(start_time, end_time)
    )
    gic_data = tva_gic.gic.sel(device=site, time=slice(start_time, end_time))

    Bx, By = mag_data.Bx.values, mag_data.By.values
    w = tukey_window(len(Bx), 0.05)
    Ex, Ey = mt_site.convolve_fft(taper(Bx, w), taper(By, w), dt=1.0)

    return mag_data.time, Bx, By, Ex, Ey, gic_data, mt_site


def align_model_results(site_results, model_colors):
    """Return model predictions/observations aligned to the common test time window."""
    time_ranges = {}
    for name in model_colors:
        if name in site_results:
            times = pd.to_datetime(
                [int(t) for t in site_results[name]["test_times"]], unit="ns"
            )
            time_ranges[name] = times

    common_start = max(t[0] for t in time_ranges.values())
    common_end = min(t[-1] for t in time_ranges.values())

    aligned = {}
    for name, times in time_ranges.items():
        s = times.searchsorted(common_start)
        e = times.searchsorted(common_end, side="right")
        aligned[name] = {
            "times": times[s:e],
            "observations": np.array(site_results[name]["observations"])[s:e],
            "predictions": np.array(site_results[name]["predictions"])[s:e],
            "pe": site_results[name]["pe"],
        }

    return aligned, common_start, common_end


def find_common_time_window(selected_sites, full_results, model_colors):
    """Find the common test time window across all sites and models."""
    starts, ends = [], []
    for site in selected_sites:
        if site not in full_results:
            continue
        for name in model_colors:
            if name in full_results[site]:
                times = pd.to_datetime(
                    [int(t) for t in full_results[site][name]["test_times"]], unit="ns"
                )
                starts.append(times[0])
                ends.append(times[-1])
    return max(starts), min(ends)
