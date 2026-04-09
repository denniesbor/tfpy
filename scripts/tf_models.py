"""
Author: Dennies Bor
Role:   Transfer function models for GIC prediction.
        Implements time-domain and frequency-domain approaches
        following Weigel (2019) and Heyns & Gaunt (2020).
"""

import numpy as np
from tqdm import tqdm

from config.settings import setup_logger
from utils.signal_processing import (
    tukey_window,
    taper,
    zero_pad,
    edge_blend,
    robust_complex_lsq,
)

logger = setup_logger(name="tfgic.transfer_functions")


def _load_train_test(
    site, gic_data, mag_data, site_relations, train_slice, test_slice, taper_pct
):
    """Extract and taper train/test B-field and GIC arrays for a site."""
    mag = site_relations[site]["magnetometer"]

    bx_tr = mag_data.sel(device=mag, time=train_slice).Bx.values
    by_tr = mag_data.sel(device=mag, time=train_slice).By.values
    g_tr = gic_data.gic.sel(device=site, time=train_slice).values
    n = min(len(g_tr), len(bx_tr), len(by_tr))
    w = tukey_window(n, taper_pct)

    bx_ts = mag_data.sel(device=mag, time=test_slice).Bx.values
    by_ts = mag_data.sel(device=mag, time=test_slice).By.values
    g_ts = gic_data.gic.sel(device=site, time=test_slice).values
    m = min(len(g_ts), len(bx_ts), len(by_ts))
    w_ts = tukey_window(m, taper_pct)

    return (
        taper(g_tr[:n], w),
        taper(bx_tr[:n], w),
        taper(by_tr[:n], w),
        n,
        g_ts[:m],
        taper(bx_ts[:m], w_ts),
        taper(by_ts[:m], w_ts),
        m,
    )


def _pe(obs, pred):
    """Prediction efficiency (1 - MSE/variance)."""
    return 1 - np.mean((obs - pred) ** 2) / np.var(obs)


def _apply_tf(tf, freqs, bx_f, by_f):
    """Apply a list of (f_lo, f_hi, zx, zy) bands to produce a prediction spectrum."""
    g_pred_f = np.zeros_like(bx_f, dtype=complex)
    for f_lo, f_hi, zx, zy in tf:
        msk = (freqs >= f_lo) & (freqs < f_hi)
        if msk.any():
            g_pred_f[msk] = zx * bx_f[msk] + zy * by_f[msk]
    return g_pred_f


def _bootstrap_bounds(preds_list, fallback, q=(25, 75)):
    """Return IQR bounds from a list of prediction arrays, or a ±10% fallback."""
    if preds_list:
        arr = np.array(preds_list)
        return np.percentile(arr, q[0], axis=0), np.percentile(arr, q[1], axis=0)
    return fallback * 0.9, fallback * 1.1


def model3_td(
    site,
    gic_data,
    mag_data,
    site_relations,
    train_slice,
    test_slice,
    pad_factor=1.0,
    taper_pct=0.05,
    n_bootstrap=100,
):
    """Time-domain GIC model using MT-derived E-fields with bootstrap uncertainty."""
    mt = site_relations[site]["mt_site"]

    g_tr, bx_tr, by_tr, n, g_ts, bx_ts, by_ts, m = _load_train_test(
        site, gic_data, mag_data, site_relations, train_slice, test_slice, taper_pct
    )

    Ex_tr, Ey_tr = mt.convolve_fft(bx_tr, by_tr, dt=1.0)
    a, b = np.linalg.lstsq(np.column_stack([Ex_tr, Ey_tr]), g_tr, rcond=None)[0]

    # Bootstrap IQR on coefficients
    np.random.seed(42)
    boot_coeffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            ab = np.linalg.lstsq(
                np.column_stack([Ex_tr[idx], Ey_tr[idx]]), g_tr[idx], rcond=None
            )[0]
            boot_coeffs.append(ab)
        except Exception:
            continue
    boot_coeffs = np.array(boot_coeffs)
    a_lo, a_hi = np.percentile(boot_coeffs[:, 0], [25, 75])
    b_lo, b_hi = np.percentile(boot_coeffs[:, 1], [25, 75])

    Ex_ts, Ey_ts = mt.convolve_fft(bx_ts, by_ts, dt=1.0)
    pred = edge_blend(a * Ex_ts + b * Ey_ts)
    pred_lower = edge_blend(a_lo * Ex_ts + b_lo * Ey_ts)
    pred_upper = edge_blend(a_hi * Ex_ts + b_hi * Ey_ts)

    return {
        "pred": pred,
        "pred_lower": pred_lower,
        "pred_upper": pred_upper,
        "pe": _pe(g_ts, pred),
        "coeffs": (a, b),
        "measured": g_ts,
        "test_times": gic_data.gic.sel(device=site, time=test_slice).time.values[:m],
    }


def model3_fd(
    site,
    gic_data,
    mag_data,
    site_relations,
    train_slice,
    test_slice,
    n_eval=28,
    taper_pct=0.05,
    n_bootstrap=50,
):
    """Frequency-domain GIC model using MT-derived E-fields."""
    mt = site_relations[site]["mt_site"]

    g_tr, bx_tr, by_tr, n, g_ts, bx_ts, by_ts, m = _load_train_test(
        site, gic_data, mag_data, site_relations, train_slice, test_slice, taper_pct
    )

    freqs, Ex_f, Ey_f = mt.convolve_fft_freq(bx_tr, by_tr, dt=1.0)
    N = 2 * (len(freqs) - 1)
    g_f = np.fft.rfft(g_tr, n=N)

    eval_f = np.logspace(np.log10(freqs[1]), np.log10(freqs[-1]), n_eval)
    bands = list(zip(eval_f[:-1], eval_f[1:]))

    def _fit_bands(Ex_f, Ey_f, g_f, freqs):
        tf = []
        for f_lo, f_hi in bands:
            msk = (freqs >= f_lo) & (freqs < f_hi)
            if msk.sum() < 5:
                continue
            A = np.vstack(
                [
                    np.column_stack([Ex_f[msk].real, Ey_f[msk].real]),
                    np.column_stack([Ex_f[msk].imag, Ey_f[msk].imag]),
                ]
            )
            b_vec = np.hstack([g_f[msk].real, g_f[msk].imag])
            a, bc = robust_complex_lsq(A, b_vec)
            tf.append((f_lo, f_hi, a + 0j, bc + 0j))
        return tf

    tf = _fit_bands(Ex_f, Ey_f, g_f, freqs)

    # Bootstrap transfer functions
    np.random.seed(42)
    boot_tfs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            fr_b, Ex_b, Ey_b = mt.convolve_fft_freq(bx_tr[idx], by_tr[idx], dt=1.0)
            boot_tfs.append(_fit_bands(Ex_b, Ey_b, np.fft.rfft(g_tr[idx], n=N), fr_b))
        except Exception:
            continue

    freqs2, Ex_f2, Ey_f2 = mt.convolve_fft_freq(bx_ts, by_ts, dt=1.0)
    N2 = 2 * (len(freqs2) - 1)
    g_pred = edge_blend(
        np.real(np.fft.irfft(_apply_tf(tf, freqs2, Ex_f2, Ey_f2), n=N2))[:m]
    )

    boot_preds = []
    for tf_b in boot_tfs[:20]:
        boot_preds.append(
            edge_blend(
                np.real(np.fft.irfft(_apply_tf(tf_b, freqs2, Ex_f2, Ey_f2), n=N2))[:m]
            )
        )
    pred_lower, pred_upper = _bootstrap_bounds(boot_preds, g_pred)

    return {
        "pred": g_pred,
        "pred_lower": pred_lower,
        "pred_upper": pred_upper,
        "pe": _pe(g_ts, g_pred),
        "transfer": tf,
        "measured": g_ts,
        "test_times": gic_data.gic.sel(device=site, time=test_slice).time.values[:m],
    }


def model4_fd(
    site,
    gic_data,
    mag_data,
    site_relations,
    train_slice,
    test_slice,
    n_eval=28,
    pad_factor=1.5,
    taper_pct=0.05,
    ridge=1e-4,
    n_bootstrap=50,
):
    """Frequency-domain GIC model estimated directly from B-fields."""
    g_tr, bx_tr, by_tr, n, g_ts, bx_ts, by_ts, m = _load_train_test(
        site, gic_data, mag_data, site_relations, train_slice, test_slice, taper_pct
    )

    bx_pad, p = zero_pad(bx_tr, pad_factor)
    by_pad, _ = zero_pad(by_tr, pad_factor)
    gic_pad, _ = zero_pad(g_tr, pad_factor)
    freqs = np.fft.rfftfreq(len(bx_pad), 1.0)
    bx_f = np.fft.rfft(bx_pad)
    by_f = np.fft.rfft(by_pad)
    gic_f = np.fft.rfft(gic_pad)

    eval_f = np.logspace(np.log10(freqs[1]), np.log10(freqs[-1]), n_eval)
    bands = list(zip(eval_f[:-1], eval_f[1:]))

    def _fit_b_bands(bx_f, by_f, gic_f, freqs):
        tf = []
        for f_lo, f_hi in bands:
            msk = (freqs >= f_lo) & (freqs < f_hi)
            if msk.sum() < 5:
                continue
            bx_b, by_b, g_b = bx_f[msk], by_f[msk], gic_f[msk]
            A = np.vstack(
                [
                    np.column_stack([bx_b.real, -bx_b.imag, by_b.real, -by_b.imag]),
                    np.column_stack([bx_b.imag, bx_b.real, by_b.imag, by_b.real]),
                ]
            )
            coef = robust_complex_lsq(
                A, np.hstack([g_b.real, g_b.imag]), δ=1.345, max_iter=20, tol=1e-6
            )
            tf.append((f_lo, f_hi, coef[0] + 1j * coef[1], coef[2] + 1j * coef[3]))
        return tf

    tf = _fit_b_bands(bx_f, by_f, gic_f, freqs)

    # Bootstrap
    np.random.seed(42)
    boot_tfs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            bx_p, _ = zero_pad(bx_tr[idx], pad_factor)
            by_p, _ = zero_pad(by_tr[idx], pad_factor)
            gi_p, _ = zero_pad(g_tr[idx], pad_factor)
            fr_b = np.fft.rfftfreq(len(bx_p), 1.0)
            boot_tfs.append(
                _fit_b_bands(
                    np.fft.rfft(bx_p), np.fft.rfft(by_p), np.fft.rfft(gi_p), fr_b
                )
            )
        except Exception:
            continue

    bx_pad_ts, p_ts = zero_pad(bx_ts, pad_factor)
    by_pad_ts, _ = zero_pad(by_ts, pad_factor)
    freqs_ts = np.fft.rfftfreq(len(bx_pad_ts), 1.0)
    bx_f_ts = np.fft.rfft(bx_pad_ts)
    by_f_ts = np.fft.rfft(by_pad_ts)

    pred = edge_blend(
        np.real(np.fft.irfft(_apply_tf(tf, freqs_ts, bx_f_ts, by_f_ts)))[
            p_ts : p_ts + m
        ]
    )

    boot_preds = [
        edge_blend(
            np.real(np.fft.irfft(_apply_tf(tb, freqs_ts, bx_f_ts, by_f_ts)))[
                p_ts : p_ts + m
            ]
        )
        for tb in boot_tfs[:20]
    ]
    pred_lower, pred_upper = _bootstrap_bounds(boot_preds, pred)

    return {
        "pred": pred,
        "pred_lower": pred_lower,
        "pred_upper": pred_upper,
        "pe": _pe(g_ts, pred),
        "transfer": tf,
        "measured": g_ts,
        "test_times": gic_data.gic.sel(device=site, time=test_slice).time.values[:m],
    }


def heyns_td(
    site,
    gic_data,
    mag_data,
    site_relations,
    train_slice,
    test_slice,
    taper_pct=0.05,
    min_gic_thresh=0.1,
    subsample_step=60,
    max_pairs=None,
):
    """Heyns time-domain ensemble with temporal subsampling."""
    mt = site_relations[site]["mt_site"]

    g_tr, bx_tr, by_tr, n, g_ts, bx_ts, by_ts, m = _load_train_test(
        site, gic_data, mag_data, site_relations, train_slice, test_slice, taper_pct
    )

    Ex_tr, Ey_tr = mt.convolve_fft(bx_tr, by_tr, dt=1.0)

    sub_idx = np.arange(0, n, subsample_step)
    n_sub = len(sub_idx)
    n_pairs = n_sub * (n_sub - 1) // 2
    logger.info(f"{site}: {n_sub} subsampled points, {n_pairs:,} pairs.")

    estimates = []
    np.random.seed(42)

    def _solve_pair(i, j):
        if abs(g_tr[i]) < min_gic_thresh and abs(g_tr[j]) < min_gic_thresh:
            return None
        A = np.array([[Ex_tr[i], Ey_tr[i]], [Ex_tr[j], Ey_tr[j]]])
        if abs(np.linalg.det(A)) <= 1e-10:
            return None
        try:
            a, b = np.linalg.solve(A, np.array([g_tr[i], g_tr[j]]))
            return (a, b) if abs(a) < 1000 and abs(b) < 1000 else None
        except np.linalg.LinAlgError:
            return None

    if max_pairs and n_pairs > max_pairs:
        for _ in tqdm(range(max_pairs), desc=f"{site} pairs"):
            i_i, j_i = np.random.choice(n_sub, 2, replace=False)
            res = _solve_pair(sub_idx[i_i], sub_idx[j_i])
            if res:
                estimates.append(res)
    else:
        for i_i in tqdm(range(n_sub), desc=f"{site} pairs"):
            for j_i in range(i_i + 1, n_sub):
                res = _solve_pair(sub_idx[i_i], sub_idx[j_i])
                if res:
                    estimates.append(res)

    if not estimates:
        raise ValueError(f"{site}: no valid parameter estimates found.")

    estimates = np.array(estimates)
    logger.info(f"{site}: {len(estimates):,} valid estimates.")

    a_med, b_med = np.median(estimates[:, 0]), np.median(estimates[:, 1])
    a_mean, b_mean = np.mean(estimates[:, 0]), np.mean(estimates[:, 1])
    a_iqr = np.percentile(estimates[:, 0], [25, 75])
    b_iqr = np.percentile(estimates[:, 1], [25, 75])

    Ex_ts, Ey_ts = mt.convolve_fft(bx_ts, by_ts, dt=1.0)
    pred = edge_blend(a_med * Ex_ts + b_med * Ey_ts)
    pred_mean = edge_blend(a_mean * Ex_ts + b_mean * Ey_ts)
    pred_lower = edge_blend(a_iqr[0] * Ex_ts + b_iqr[0] * Ey_ts)
    pred_upper = edge_blend(a_iqr[1] * Ex_ts + b_iqr[1] * Ey_ts)

    pe_med = _pe(g_ts, pred)
    pe_mean = _pe(g_ts, pred_mean)
    logger.info(f"{site}: PE_median={pe_med:.4f}, PE_mean={pe_mean:.4f}")

    return {
        "pred": pred,
        "pred_mean": pred_mean,
        "pred_lower": pred_lower,
        "pred_upper": pred_upper,
        "pe": pe_med,
        "pe_mean": pe_mean,
        "coeffs": (a_med, b_med),
        "coeffs_mean": (a_mean, b_mean),
        "ensemble_size": len(estimates),
        "alpha_ensemble": estimates[:, 0],
        "beta_ensemble": estimates[:, 1],
        "alpha_iqr": a_iqr,
        "beta_iqr": b_iqr,
        "measured": g_ts,
        "test_times": gic_data.gic.sel(device=site, time=test_slice).time.values[:m],
        "sampling_info": {
            "subsample_step": subsample_step,
            "subsampled_points": n_sub,
            "total_pairs": n_pairs,
            "valid_estimates": len(estimates),
        },
    }


def heyns_ensemble(
    site,
    gic_data,
    mag_data,
    site_relations,
    train_slice,
    test_slice,
    window_hours=8,
    shift_hours=0.5,
    n_eval=28,
    taper_pct=0.05,
    pad_factor=1.5,
):
    """Heyns B-field to GIC ensemble transfer function with sliding windows."""
    g_tr, bx_tr, by_tr, n, g_ts, bx_ts, by_ts, m = _load_train_test(
        site, gic_data, mag_data, site_relations, train_slice, test_slice, taper_pct
    )

    dt_s = float(gic_data.time.sel(time=train_slice).diff("time")[0].dt.total_seconds())
    win_pts = min(int(window_hours * 3600 / dt_s), n)
    shift_pts = int(shift_hours * 3600 / dt_s) or win_pts // 4
    starts = list(range(0, n - win_pts + 1, shift_pts))
    logger.info(
        f"{site}: {len(starts)} windows, win={win_pts} pts, shift={shift_pts} pts."
    )

    tf_ensemble = {}

    for start in tqdm(starts, desc=f"{site} windows"):
        sl = slice(start, start + win_pts)
        g_w = taper(g_tr[sl], tukey_window(win_pts, taper_pct))
        bx_w = taper(bx_tr[sl], tukey_window(win_pts, taper_pct))
        by_w = taper(by_tr[sl], tukey_window(win_pts, taper_pct))

        bx_pad, _ = zero_pad(bx_w, pad_factor)
        by_pad, _ = zero_pad(by_w, pad_factor)
        gi_pad, _ = zero_pad(g_w, pad_factor)
        freqs = np.fft.rfftfreq(len(bx_pad), dt_s)
        bx_f = np.fft.rfft(bx_pad)
        by_f = np.fft.rfft(by_pad)
        gic_f = np.fft.rfft(gi_pad)

        eval_f = np.logspace(np.log10(freqs[1]), np.log10(freqs[-1]), n_eval)
        for f_lo, f_hi in zip(eval_f[:-1], eval_f[1:]):
            msk = (freqs >= f_lo) & (freqs < f_hi)
            if msk.sum() < 5:
                continue
            bx_b, by_b, g_b = bx_f[msk], by_f[msk], gic_f[msk]
            A = np.vstack(
                [
                    np.column_stack([bx_b.real, -bx_b.imag, by_b.real, -by_b.imag]),
                    np.column_stack([bx_b.imag, bx_b.real, by_b.imag, by_b.real]),
                ]
            )
            try:
                coef = robust_complex_lsq(A, np.hstack([g_b.real, g_b.imag]))
                tf_ensemble.setdefault((f_lo, f_hi), []).append(
                    (coef[0] + 1j * coef[1], coef[2] + 1j * coef[3])
                )
            except Exception:
                continue

    tf_median, tf_bounds = [], []
    for (f_lo, f_hi), ests in tf_ensemble.items():
        if len(ests) < 2:
            continue
        zx_v = np.array([e[0] for e in ests])
        zy_v = np.array([e[1] for e in ests])
        tf_median.append(
            (
                f_lo,
                f_hi,
                np.median(zx_v.real) + 1j * np.median(zx_v.imag),
                np.median(zy_v.real) + 1j * np.median(zy_v.imag),
            )
        )
        tf_bounds.append((f_lo, f_hi, np.std(np.abs(zx_v)), np.std(np.abs(zy_v))))

    if not tf_median:
        raise ValueError(f"{site}: no valid transfer function estimates.")

    logger.info(f"{site}: {len(tf_median)} valid frequency bands.")

    bx_pad_ts, p_ts = zero_pad(bx_ts, pad_factor)
    by_pad_ts, _ = zero_pad(by_ts, pad_factor)
    freqs_ts = np.fft.rfftfreq(len(bx_pad_ts), dt_s)
    bx_f_ts = np.fft.rfft(bx_pad_ts)
    by_f_ts = np.fft.rfft(by_pad_ts)

    pred = edge_blend(
        np.real(np.fft.irfft(_apply_tf(tf_median, freqs_ts, bx_f_ts, by_f_ts)))[
            p_ts : p_ts + m
        ]
    )

    # Sample from ensemble for uncertainty bounds
    boot_preds = []
    for (f_lo, f_hi), ests in tf_ensemble.items():
        if len(ests) < 10:
            continue
        for idx in np.random.choice(len(ests), min(20, len(ests)), replace=False):
            zx, zy = ests[idx]
            g_f_s = np.zeros_like(bx_f_ts, dtype=complex)
            msk = (freqs_ts >= f_lo) & (freqs_ts < f_hi)
            g_f_s[msk] = zx * bx_f_ts[msk] + zy * by_f_ts[msk]
            boot_preds.append(edge_blend(np.real(np.fft.irfft(g_f_s))[p_ts : p_ts + m]))

    pred_lower, pred_upper = _bootstrap_bounds(boot_preds, pred, q=(25, 75))
    pe = _pe(g_ts, pred)
    logger.info(f"{site}: PE={pe:.4f}")

    return {
        "pred": pred,
        "pred_lower": pred_lower,
        "pred_upper": pred_upper,
        "pe": pe,
        "transfer_median": tf_median,
        "transfer_bounds": tf_bounds,
        "ensemble_sizes": {k: len(v) for k, v in tf_ensemble.items()},
        "n_windows": len(starts),
        "n_frequency_bands": len(tf_median),
        "measured": g_ts,
        "test_times": gic_data.gic.sel(device=site, time=test_slice).time.values[:m],
    }
