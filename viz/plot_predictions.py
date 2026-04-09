"""
Author: Dennies Bor
Role:   Stacked model prediction plots for all TVA GIC sites.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config.settings import setup_logger, RESULTS_DIR
from viz.plot_utils import (
    setup_matplotlib,
    save_figure,
    find_common_time_window,
    align_model_results,
)

logger = setup_logger(name="tfgic.plot_predictions")

MODEL_COLORS = {
    "3_td": "red",
    "3_fd": "blue",
    "4": "green",
    "heyns_td": "orange",
    "heyns_ens": "purple",
    "5a": "cyan",
    "5b": "magenta",
}

MODEL_DISPLAY_NAMES = {
    "3_td": "Model 1",
    "3_fd": "Model 2",
    "4": "Model 3",
    "heyns_td": "Heyns TD",
    "heyns_ens": "Heyns FD",
    "5a": "Model 5A",
    "5b": "Model 5B",
}

TRAIN_START = np.datetime64("2024-05-10T15:00:00")


def _plot_single_site(
    fig, gs_row, site, full_results, tva_gic, common_start, common_end, bottom=False
):
    """Draw training + stacked prediction panels for one site."""
    n_models = len(MODEL_COLORS)

    gs_site = fig.add_gridspec(
        n_models,
        2,
        width_ratios=[1, 1],
        hspace=0,
        wspace=0,
        top=0.98 - gs_row * 0.2,
        bottom=0.98 - (gs_row + 1) * 0.2,
    )

    # Left: training panel
    ax_train = fig.add_subplot(gs_site[:, 0])
    train_slice = slice(TRAIN_START, common_start)
    train_gic = tva_gic.gic.sel(device=site, time=train_slice)

    ax_train.axvspan(train_slice.start, train_slice.stop, alpha=0.2, color="grey")
    ax_train.plot(train_gic.time, train_gic.values, "grey", alpha=0.8, linewidth=0.8)
    ax_train.axvline(
        train_slice.stop, color="red", linestyle=":", alpha=0.7, linewidth=1.0
    )
    ax_train.set_ylabel("GIC (A)", fontsize=9)
    ax_train.set_xlim(train_slice.start, train_slice.stop)
    ax_train.spines["top"].set_visible(False)
    ax_train.spines["right"].set_visible(False)
    ax_train.tick_params(right=False, top=False, labelsize=8)

    if bottom:
        ax_train.set_xlabel("Time", fontsize=10)
        ax_train.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
        ax_train.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%m/%d"))
    else:
        ax_train.set_xticklabels([])

    # Right: stacked model panels
    aligned, _, _ = align_model_results(full_results[site], MODEL_COLORS)

    for i, (name, color) in enumerate(MODEL_COLORS.items()):
        if name not in aligned:
            continue

        ax = fig.add_subplot(gs_site[i, 1])
        d = aligned[name]

        ax.plot(d["times"], d["observations"], "k-", linewidth=0.6, alpha=0.7)
        ax.plot(d["times"], d["predictions"], color, linewidth=0.6, alpha=0.8)

        label = MODEL_DISPLAY_NAMES.get(name, name.upper())
        ax.text(
            1.07,
            0.5,
            f"{label} (PE: {d['pe']:.2f})",
            transform=ax.transAxes,
            fontsize=9,
            color="k",
            va="center",
        )

        ax.set_xlim(common_start, common_end)
        ax.set_yticklabels([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False, right=False, top=False, labelsize=8)

        is_last = i == n_models - 1
        if is_last and bottom:
            ax.set_xlabel("Time", fontsize=10)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%m/%d"))
        else:
            ax.set_xticklabels([])
            ax.tick_params(bottom=False)

    ax.text(-0.8, 0.5, site, transform=ax.transAxes, va="center", fontsize=10)


def plot_all_sites(selected_sites, tva_gic, closest_mag_to_gic):
    """Produce and save the stacked multi-site prediction comparison figure."""
    setup_matplotlib()

    with open(RESULTS_DIR / "regression_models_results.json") as f:
        full_results = json.load(f)

    common_start, common_end = find_common_time_window(
        selected_sites, full_results, MODEL_COLORS
    )
    logger.info(f"Common test window: {common_start} to {common_end}.")

    fig = plt.figure(figsize=(8, 12))

    for i, site in enumerate(selected_sites):
        if site not in full_results:
            logger.warning(f"{site} not in results, skipping.")
            continue
        _plot_single_site(
            fig,
            i,
            site,
            full_results,
            tva_gic,
            common_start,
            common_end,
            bottom=(site == selected_sites[-1]),
        )

    save_figure(fig, "preds_h_w")
    plt.close(fig)
    logger.info("Saved preds_h_w figure.")


if __name__ == "__main__":
    from scripts.data_loading import (
        load_tva_gic,
        load_tva_magnetometer,
        find_closest_magnetometers_to_gic,
        get_selected_sites,
    )

    tva_gic = load_tva_gic()
    tva_mag = load_tva_magnetometer()
    sites = get_selected_sites()
    site_rel = find_closest_magnetometers_to_gic(gic_data=tva_gic, mag_data=tva_mag)

    plot_all_sites(sites, tva_gic, site_rel)
