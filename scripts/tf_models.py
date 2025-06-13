"""
Transfer function models for GIC prediction.

This module implements various transfer function models for predicting
GIC from magnetic field measurements, including time-domain and
frequency-domain approaches as described in Weigel and Cilliers (2019) 
and Gaunt and Heyns (2020).

References:
https://www.sciencedirect.com/science/article/abs/pii/S0378779620303503
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019SW002234
"""

import numpy as np
from utils.signal_processing import (
    tukey_window,
    taper,
    zero_pad,
    edge_blend,
    robust_complex_lsq,
)
from config.settings import setup_logger
from tqdm import tqdm

logger = setup_logger(name="tfgic.transfer_functions")

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
    """Time-domain model for GIC prediction using MT-derived E-fields."""
    mag = site_relations[site]["magnetometer"]
    mt = site_relations[site]["mt_site"]

    gic_tr = gic_data.gic.sel(device=site, time=train_slice).values
    bx_tr = mag_data.sel(device=mag, time=train_slice).Bx.values
    by_tr = mag_data.sel(device=mag, time=train_slice).By.values
    n = min(len(gic_tr), len(bx_tr), len(by_tr))
    w = tukey_window(n, taper_pct)

    gic_tr = taper(gic_tr[:n], w)
    Ex_tr, Ey_tr = mt.convolve_fft(taper(bx_tr[:n], w), taper(by_tr[:n], w), dt=1.0)

    a, b = np.linalg.lstsq(np.column_stack([Ex_tr, Ey_tr]), gic_tr, rcond=None)[0]
    
    np.random.seed(42)
    bootstrap_coeffs = []
    for i in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        Ex_boot = Ex_tr[indices]
        Ey_boot = Ey_tr[indices]
        gic_boot = gic_tr[indices]
        try:
            a_boot, b_boot = np.linalg.lstsq(np.column_stack([Ex_boot, Ey_boot]), gic_boot, rcond=None)[0]
            bootstrap_coeffs.append((a_boot, b_boot))
        except:
            continue
    
    bootstrap_coeffs = np.array(bootstrap_coeffs)
    a_lower, a_upper = np.percentile(bootstrap_coeffs[:, 0], [25, 75])
    b_lower, b_upper = np.percentile(bootstrap_coeffs[:, 1], [25, 75])

    gic_ts = gic_data.gic.sel(device=site, time=test_slice).values
    bx_ts = mag_data.sel(device=mag, time=test_slice).Bx.values
    by_ts = mag_data.sel(device=mag, time=test_slice).By.values
    m = min(len(gic_ts), len(bx_ts), len(by_ts))
    w_ts = tukey_window(m, taper_pct)

    Ex_ts, Ey_ts = mt.convolve_fft(
        taper(bx_ts[:m], w_ts), taper(by_ts[:m], w_ts), dt=1.0
    )
    
    pred = edge_blend(a * Ex_ts + b * Ey_ts)
    pred_lower = edge_blend(a_lower * Ex_ts + b_lower * Ey_ts)
    pred_upper = edge_blend(a_upper * Ex_ts + b_upper * Ey_ts)

    test_times = gic_data.gic.sel(device=site, time=test_slice).time.values[:m]

    res = gic_ts[:m] - pred
    pe = 1 - np.mean(res**2) / np.var(gic_ts[:m])

    return {
        "pred": pred, 
        "pred_lower": pred_lower,
        "pred_upper": pred_upper,
        "pe": pe, 
        "coeffs": (a, b), 
        "measured": gic_ts[:m],
        "test_times": test_times
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
    """Frequency-domain model for GIC prediction using MT-derived E-fields."""
    mag = site_relations[site]["magnetometer"]
    mt = site_relations[site]["mt_site"]

    g = gic_data.gic.sel(device=site, time=train_slice).values
    bx = mag_data.sel(device=mag, time=train_slice).Bx.values
    by = mag_data.sel(device=mag, time=train_slice).By.values
    n = min(len(g), len(bx), len(by))
    w = tukey_window(n, taper_pct)
    g, bx, by = taper(g[:n], w), taper(bx[:n], w), taper(by[:n], w)

    freqs, Ex_f, Ey_f = mt.convolve_fft_freq(bx, by, dt=1.0)
    N = 2 * (len(freqs) - 1)
    g_f = np.fft.rfft(g, n=N)

    f_min, f_max = freqs[1], freqs[-1]
    eval_f = np.logspace(np.log10(f_min), np.log10(f_max), n_eval)
    bands = list(zip(eval_f[:-1], eval_f[1:]))

    tf = []
    for f_lo, f_hi in bands:
        msk = (freqs >= f_lo) & (freqs < f_hi)
        if msk.sum() < 5:
            continue
        Ex_b, Ey_b, g_b = Ex_f[msk], Ey_f[msk], g_f[msk]
        A = np.vstack([
            np.column_stack([Ex_b.real, Ey_b.real]),
            np.column_stack([Ex_b.imag, Ey_b.imag]),
        ])
        b = np.hstack([g_b.real, g_b.imag])
        a, bcoef = robust_complex_lsq(A, b)
        tf.append((f_lo, f_hi, a + 0j, bcoef + 0j))

    np.random.seed(42)
    bootstrap_tfs = []
    for i in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        g_boot = g[indices]
        bx_boot = bx[indices]
        by_boot = by[indices]
        
        w_boot = tukey_window(len(g_boot), taper_pct)
        g_boot, bx_boot, by_boot = taper(g_boot, w_boot), taper(bx_boot, w_boot), taper(by_boot, w_boot)
        
        try:
            freqs_boot, Ex_f_boot, Ey_f_boot = mt.convolve_fft_freq(bx_boot, by_boot, dt=1.0)
            g_f_boot = np.fft.rfft(g_boot, n=N)
            
            tf_boot = []
            for f_lo, f_hi in bands:
                msk = (freqs_boot >= f_lo) & (freqs_boot < f_hi)
                if msk.sum() < 5:
                    continue
                Ex_b, Ey_b, g_b = Ex_f_boot[msk], Ey_f_boot[msk], g_f_boot[msk]
                A = np.vstack([
                    np.column_stack([Ex_b.real, Ey_b.real]),
                    np.column_stack([Ex_b.imag, Ey_b.imag]),
                ])
                b = np.hstack([g_b.real, g_b.imag])
                a, bcoef = robust_complex_lsq(A, b)
                tf_boot.append((f_lo, f_hi, a + 0j, bcoef + 0j))
            bootstrap_tfs.append(tf_boot)
        except:
            continue

    g2 = gic_data.gic.sel(device=site, time=test_slice).values
    bx = mag_data.sel(device=mag, time=test_slice).Bx.values
    by = mag_data.sel(device=mag, time=test_slice).By.values
    m = min(len(g2), len(bx), len(by))
    w2 = tukey_window(m, taper_pct)
    bx, by = taper(bx[:m], w2), taper(by[:m], w2)

    freqs2, Ex_f2, Ey_f2 = mt.convolve_fft_freq(bx, by, dt=1.0)
    N2 = 2 * (len(freqs2) - 1)
    
    g_pred_f = np.zeros_like(Ex_f2, dtype=complex)
    for f_lo, f_hi, a, bcoef in tf:
        msk = (freqs2 >= f_lo) & (freqs2 < f_hi)
        g_pred_f[msk] = a * Ex_f2[msk] + bcoef * Ey_f2[msk]
    g_pred = edge_blend(np.real(np.fft.irfft(g_pred_f, n=N2))[:m])
    
    bootstrap_preds = []
    for tf_boot in bootstrap_tfs[:20]:  # Use subset for speed
        g_pred_f_boot = np.zeros_like(Ex_f2, dtype=complex)
        for f_lo, f_hi, a, bcoef in tf_boot:
            msk = (freqs2 >= f_lo) & (freqs2 < f_hi)
            if msk.sum() > 0:
                g_pred_f_boot[msk] = a * Ex_f2[msk] + bcoef * Ey_f2[msk]
        pred_boot = edge_blend(np.real(np.fft.irfft(g_pred_f_boot, n=N2))[:m])
        bootstrap_preds.append(pred_boot)
    
    if bootstrap_preds:
        bootstrap_preds = np.array(bootstrap_preds)
        pred_lower = np.percentile(bootstrap_preds, 25, axis=0)
        pred_upper = np.percentile(bootstrap_preds, 75, axis=0)
    else:
        pred_lower = g_pred * 0.9
        pred_upper = g_pred * 1.1
    
    test_times = gic_data.gic.sel(device=site, time=test_slice).time.values[:m]
    
    res = g2[:m] - g_pred
    pe = 1 - np.mean(res**2) / np.var(g2[:m])

    return {
        "pred": g_pred, 
        "pred_lower": pred_lower,
        "pred_upper": pred_upper,
        "pe": pe, 
        "transfer": tf, 
        "measured": g2[:m],
        "test_times": test_times
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
    """Enhanced frequency-domain model for GIC prediction directly from B-fields."""
    mag = site_relations[site]["magnetometer"]

    gic_tr = gic_data.gic.sel(device=site, time=train_slice).values
    bx_tr = mag_data.sel(device=mag, time=train_slice).Bx.values
    by_tr = mag_data.sel(device=mag, time=train_slice).By.values
    n = min(len(gic_tr), len(bx_tr), len(by_tr))
    w = tukey_window(n, taper_pct)

    gic_tr = taper(gic_tr[:n], w)
    bx_tr = taper(bx_tr[:n], w)
    by_tr = taper(by_tr[:n], w)

    bx_pad, p = zero_pad(bx_tr, pad_factor)
    by_pad, _ = zero_pad(by_tr, pad_factor)
    gic_pad, _ = zero_pad(gic_tr, pad_factor)

    dt = 1.0
    freqs = np.fft.rfftfreq(len(bx_pad), dt)
    bx_f = np.fft.rfft(bx_pad)
    by_f = np.fft.rfft(by_pad)
    gic_f = np.fft.rfft(gic_pad)

    f_min, f_max = freqs[1], freqs[-1]
    eval_f = np.logspace(np.log10(f_min), np.log10(f_max), n_eval)
    bands = list(zip(eval_f[:-1], eval_f[1:]))

    tf = []
    for f_lo, f_hi in bands:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        if mask.sum() < 5:
            continue

        bx_b, by_b, g_b = bx_f[mask], by_f[mask], gic_f[mask]

        A = np.vstack([
            np.column_stack([bx_b.real, -bx_b.imag, by_b.real, -by_b.imag]),
            np.column_stack([bx_b.imag, bx_b.real, by_b.imag, by_b.real]),
        ])
        b = np.hstack([g_b.real, g_b.imag])

        coef = robust_complex_lsq(A, b, δ=1.345, max_iter=20, tol=1e-6)
        zx, zy = coef[0] + 1j * coef[1], coef[2] + 1j * coef[3]
        tf.append((f_lo, f_hi, zx, zy))

    np.random.seed(42)
    bootstrap_tfs = []
    for i in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        gic_boot = gic_tr[indices]
        bx_boot = bx_tr[indices]
        by_boot = by_tr[indices]
        
        try:
            bx_pad_boot, p_boot = zero_pad(bx_boot, pad_factor)
            by_pad_boot, _ = zero_pad(by_boot, pad_factor)
            gic_pad_boot, _ = zero_pad(gic_boot, pad_factor)
            
            bx_f_boot = np.fft.rfft(bx_pad_boot)
            by_f_boot = np.fft.rfft(by_pad_boot)
            gic_f_boot = np.fft.rfft(gic_pad_boot)
            
            tf_boot = []
            for f_lo, f_hi in bands:
                mask = (freqs >= f_lo) & (freqs < f_hi)
                if mask.sum() < 5:
                    continue
                    
                bx_b, by_b, g_b = bx_f_boot[mask], by_f_boot[mask], gic_f_boot[mask]
                
                A = np.vstack([
                    np.column_stack([bx_b.real, -bx_b.imag, by_b.real, -by_b.imag]),
                    np.column_stack([bx_b.imag, bx_b.real, by_b.imag, by_b.real]),
                ])
                b = np.hstack([g_b.real, g_b.imag])
                
                coef = robust_complex_lsq(A, b, δ=1.345, max_iter=20, tol=1e-6)
                zx, zy = coef[0] + 1j * coef[1], coef[2] + 1j * coef[3]
                tf_boot.append((f_lo, f_hi, zx, zy))
            
            bootstrap_tfs.append(tf_boot)
        except:
            continue

    gic_ts = gic_data.gic.sel(device=site, time=test_slice).values
    bx_ts = mag_data.sel(device=mag, time=test_slice).Bx.values
    by_ts = mag_data.sel(device=mag, time=test_slice).By.values
    m = min(len(gic_ts), len(bx_ts), len(by_ts))
    w_ts = tukey_window(m, taper_pct)

    bx_pad, p_ts = zero_pad(taper(bx_ts[:m], w_ts), pad_factor)
    by_pad, _ = zero_pad(taper(by_ts[:m], w_ts), pad_factor)
    bx_f = np.fft.rfft(bx_pad)
    by_f = np.fft.rfft(by_pad)
    freqs_ts = np.fft.rfftfreq(len(bx_pad), dt)

    g_pred_f = np.zeros_like(bx_f, dtype=complex)
    for f_lo, f_hi, zx, zy in tf:
        msk = (freqs_ts >= f_lo) & (freqs_ts < f_hi)
        g_pred_f[msk] = zx * bx_f[msk] + zy * by_f[msk]

    g_pred_pad = np.real(np.fft.irfft(g_pred_f))
    pred = edge_blend(g_pred_pad[p_ts : p_ts + m])

    bootstrap_preds = []
    for tf_boot in bootstrap_tfs[:20]:  # Use subset for speed
        g_pred_f_boot = np.zeros_like(bx_f, dtype=complex)
        for f_lo, f_hi, zx, zy in tf_boot:
            msk = (freqs_ts >= f_lo) & (freqs_ts < f_hi)
            if msk.sum() > 0:
                g_pred_f_boot[msk] = zx * bx_f[msk] + zy * by_f[msk]
        
        g_pred_pad_boot = np.real(np.fft.irfft(g_pred_f_boot))
        pred_boot = edge_blend(g_pred_pad_boot[p_ts : p_ts + m])
        bootstrap_preds.append(pred_boot)
    
    if bootstrap_preds:
        bootstrap_preds = np.array(bootstrap_preds)
        pred_lower = np.percentile(bootstrap_preds, 25, axis=0)
        pred_upper = np.percentile(bootstrap_preds, 75, axis=0)
    else:
        pred_lower = pred * 0.9
        pred_upper = pred * 1.1

    test_times = gic_data.gic.sel(device=site, time=test_slice).time.values[:m]

    res = gic_ts[:m] - pred
    pe = 1 - np.mean(res**2) / np.var(gic_ts[:m])

    return {
        "pred": pred, 
        "pred_lower": pred_lower,
        "pred_upper": pred_upper,
        "pe": pe, 
        "transfer": tf, 
        "measured": gic_ts[:m],
        "test_times": test_times
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
    """Heyns time-domain ensemble method with temporal subsampling."""
    mag = site_relations[site]["magnetometer"]
    mt = site_relations[site]["mt_site"]

    logger.info(f"[{site}] Starting Heyns TD with temporal subsampling (every {subsample_step}s)...")
    
    gic_tr = gic_data.gic.sel(device=site, time=train_slice).values
    bx_tr = mag_data.sel(device=mag, time=train_slice).Bx.values
    by_tr = mag_data.sel(device=mag, time=train_slice).By.values
    n = min(len(gic_tr), len(bx_tr), len(by_tr))
    
    logger.info(f"[{site}] Training data length: {n} points")
    
    w = tukey_window(n, taper_pct)
    gic_tr = taper(gic_tr[:n], w)
    Ex_tr, Ey_tr = mt.convolve_fft(taper(bx_tr[:n], w), taper(by_tr[:n], w), dt=1.0)

    subsample_indices = np.arange(0, n, subsample_step)
    n_sub = len(subsample_indices)
    
    logger.info(f"[{site}] Subsampled to {n_sub} points (every {subsample_step}s)")
    
    total_pairs = n_sub * (n_sub - 1) // 2
    logger.info(f"[{site}] Total pairs from subsampled data: {total_pairs:,}")
    
    ensemble_estimates = []
    
    if max_pairs is not None and total_pairs > max_pairs:
        logger.info(f"[{site}] Randomly sampling {max_pairs:,} pairs...")
        np.random.seed(42)
        
        for _ in tqdm(range(max_pairs), desc=f"[{site}] Processing pairs"):
            i_idx, j_idx = np.random.choice(n_sub, 2, replace=False)
            if i_idx == j_idx:
                continue
                
            i, j = subsample_indices[i_idx], subsample_indices[j_idx]
            
            # Skip very small GIC pairs
            if abs(gic_tr[i]) < min_gic_thresh and abs(gic_tr[j]) < min_gic_thresh:
                continue
                
            # Solve 2x2 system: GIC[i,j] = α*Ex[i,j] + β*Ey[i,j]
            A = np.array([[Ex_tr[i], Ey_tr[i]], 
                         [Ex_tr[j], Ey_tr[j]]])
            b = np.array([gic_tr[i], gic_tr[j]])
            
            # Check for numerical stability
            if abs(np.linalg.det(A)) > 1e-10:
                try:
                    alpha, beta = np.linalg.solve(A, b)
                    # Basic outlier rejection
                    if abs(alpha) < 1000 and abs(beta) < 1000:
                        ensemble_estimates.append((alpha, beta))
                except np.linalg.LinAlgError:
                    continue
    else:
        logger.info(f"[{site}] Processing all {total_pairs:,} pairs...")
        
        for i_idx in tqdm(range(n_sub), desc=f"[{site}] Processing pairs"):
            for j_idx in range(i_idx + 1, n_sub):
                i, j = subsample_indices[i_idx], subsample_indices[j_idx]
                
                # Skip very small GIC pairs
                if abs(gic_tr[i]) < min_gic_thresh and abs(gic_tr[j]) < min_gic_thresh:
                    continue
                    
                # Solve 2x2 system: GIC[i,j] = α*Ex[i,j] + β*Ey[i,j]
                A = np.array([[Ex_tr[i], Ey_tr[i]], 
                             [Ex_tr[j], Ey_tr[j]]])
                b = np.array([gic_tr[i], gic_tr[j]])
                
                # Check for numerical stability
                if abs(np.linalg.det(A)) > 1e-10:
                    try:
                        alpha, beta = np.linalg.solve(A, b)
                        # Basic outlier rejection
                        if abs(alpha) < 1000 and abs(beta) < 1000:
                            ensemble_estimates.append((alpha, beta))
                    except np.linalg.LinAlgError:
                        continue
    
    ensemble_estimates = np.array(ensemble_estimates)
    n_estimates = len(ensemble_estimates)
    
    if n_estimates == 0:
        raise ValueError("No valid parameter estimates found")
    
    logger.info(f"[{site}] Generated {n_estimates:,} valid parameter estimates")
    
    alpha_ensemble = ensemble_estimates[:, 0]
    beta_ensemble = ensemble_estimates[:, 1]
    
    # Use median (as in original) but also compute mean for comparison
    alpha_median = np.median(alpha_ensemble)
    beta_median = np.median(beta_ensemble)
    alpha_mean = np.mean(alpha_ensemble)
    beta_mean = np.mean(beta_ensemble)
    
    alpha_iqr = np.percentile(alpha_ensemble, [25, 75])
    beta_iqr = np.percentile(beta_ensemble, [25, 75])

    logger.info(f"[{site}] Running test predictions...")
    
    gic_ts = gic_data.gic.sel(device=site, time=test_slice).values
    bx_ts = mag_data.sel(device=mag, time=test_slice).Bx.values
    by_ts = mag_data.sel(device=mag, time=test_slice).By.values
    m = min(len(gic_ts), len(bx_ts), len(by_ts))
    w_ts = tukey_window(m, taper_pct)

    Ex_ts, Ey_ts = mt.convolve_fft(
        taper(bx_ts[:m], w_ts), taper(by_ts[:m], w_ts), dt=1.0
    )
    
    pred = edge_blend(alpha_median * Ex_ts + beta_median * Ey_ts)
    pred_mean = edge_blend(alpha_mean * Ex_ts + beta_mean * Ey_ts)
    pred_lower = edge_blend(alpha_iqr[0] * Ex_ts + beta_iqr[0] * Ey_ts)
    pred_upper = edge_blend(alpha_iqr[1] * Ex_ts + beta_iqr[1] * Ey_ts)

    test_times = gic_data.gic.sel(device=site, time=test_slice).time.values[:m]

    res_median = gic_ts[:m] - pred
    pe_median = 1 - np.mean(res_median**2) / np.var(gic_ts[:m])
    
    res_mean = gic_ts[:m] - pred_mean  
    pe_mean = 1 - np.mean(res_mean**2) / np.var(gic_ts[:m])

    logger.info(f"[{site}] Heyns TD complete: PE_median = {pe_median:.4f}, PE_mean = {pe_mean:.4f}")

    return {
        "pred": pred,
        "pred_mean": pred_mean,
        "pred_lower": pred_lower,
        "pred_upper": pred_upper,
        "pe": pe_median,
        "pe_mean": pe_mean,
        "coeffs": (alpha_median, beta_median),
        "coeffs_mean": (alpha_mean, beta_mean),
        "ensemble_size": n_estimates,
        "alpha_ensemble": alpha_ensemble,
        "beta_ensemble": beta_ensemble,
        "alpha_iqr": alpha_iqr,
        "beta_iqr": beta_iqr,
        "measured": gic_ts[:m],
        "test_times": test_times,
        "sampling_info": {
            "strategy": "temporal_subsampling",
            "subsample_step": subsample_step,
            "subsampled_points": n_sub,
            "total_pairs": total_pairs,
            "valid_estimates": n_estimates,
        }
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
    """Heyns B-field to GIC ensemble transfer function method."""
    mag = site_relations[site]["magnetometer"]

    logger.info(f"[{site}] Starting Heyns B→GIC ensemble method...")
    
    gic_tr = gic_data.gic.sel(device=site, time=train_slice).values
    bx_tr = mag_data.sel(device=mag, time=train_slice).Bx.values
    by_tr = mag_data.sel(device=mag, time=train_slice).By.values
    n = min(len(gic_tr), len(bx_tr), len(by_tr))
    
    logger.info(f"[{site}] Training data length: {n} points")
    
    # Get actual sampling rate from dataset
    time_coords = gic_data.time.sel(time=train_slice)
    dt_seconds = (time_coords[1] - time_coords[0]).dt.total_seconds().values
    
    # Convert hours to samples using actual sampling rate
    window_pts = int(window_hours * 3600 / dt_seconds)
    shift_pts = int(shift_hours * 3600 / dt_seconds)
    
    if window_pts > n:
        window_pts = n
        shift_pts = n // 4
    
    logger.info(f"[{site}] Window: {window_pts} pts ({window_pts*dt_seconds/3600:.1f}h), Shift: {shift_pts} pts")
    
    # Create overlapping windows
    window_starts = list(range(0, n - window_pts + 1, shift_pts))
    logger.info(f"[{site}] Number of windows: {len(window_starts)}")
    
    tf_ensemble = {}  # freq_band -> list of (zx, zy) estimates
    
    for start in tqdm(window_starts, desc=f"[{site}] Processing windows"):
        end = start + window_pts
        
        # Extract window
        gic_win = gic_tr[start:end]
        bx_win = bx_tr[start:end]
        by_win = by_tr[start:end]
        
        # Taper and pad
        w = tukey_window(len(gic_win), taper_pct)
        gic_win = taper(gic_win, w)
        bx_win = taper(bx_win, w)
        by_win = taper(by_win, w)
        
        bx_pad, p = zero_pad(bx_win, pad_factor)
        by_pad, _ = zero_pad(by_win, pad_factor)
        gic_pad, _ = zero_pad(gic_win, pad_factor)
        
        # FFT
        dt = dt_seconds
        freqs = np.fft.rfftfreq(len(bx_pad), dt)
        bx_f = np.fft.rfft(bx_pad)
        by_f = np.fft.rfft(by_pad)
        gic_f = np.fft.rfft(gic_pad)
        
        # Frequency bands
        f_min, f_max = freqs[1], freqs[-1]
        eval_f = np.logspace(np.log10(f_min), np.log10(f_max), n_eval)
        bands = list(zip(eval_f[:-1], eval_f[1:]))
        
        # Estimate TF for each band
        for f_lo, f_hi in bands:
            mask = (freqs >= f_lo) & (freqs < f_hi)
            if mask.sum() < 5:
                continue
                
            bx_b, by_b, g_b = bx_f[mask], by_f[mask], gic_f[mask]
            
            # Complex LSQ
            A = np.vstack([
                np.column_stack([bx_b.real, -bx_b.imag, by_b.real, -by_b.imag]),
                np.column_stack([bx_b.imag, bx_b.real, by_b.imag, by_b.real]),
            ])
            b = np.hstack([g_b.real, g_b.imag])
            
            try:
                coef = robust_complex_lsq(A, b)
                zx = coef[0] + 1j * coef[1]
                zy = coef[2] + 1j * coef[3]
                
                # Store in ensemble
                band_key = (f_lo, f_hi)
                if band_key not in tf_ensemble:
                    tf_ensemble[band_key] = []
                tf_ensemble[band_key].append((zx, zy))
            except:
                continue
    
    # Ensemble statistics: median TF and bounds
    tf_median = []
    tf_bounds = []
    
    for (f_lo, f_hi), estimates in tf_ensemble.items():
        if len(estimates) < 2:
            continue
            
        zx_vals = np.array([est[0] for est in estimates])
        zy_vals = np.array([est[1] for est in estimates])
        
        # Use median of real/imag parts separately
        zx_med = np.median(zx_vals.real) + 1j * np.median(zx_vals.imag)
        zy_med = np.median(zy_vals.real) + 1j * np.median(zy_vals.imag)
        
        tf_median.append((f_lo, f_hi, zx_med, zy_med))
        
        # Store ensemble spread for uncertainty
        zx_spread = np.std(np.abs(zx_vals))
        zy_spread = np.std(np.abs(zy_vals))
        tf_bounds.append((f_lo, f_hi, zx_spread, zy_spread))

    if not tf_median:
        raise ValueError("No valid transfer function estimates found")

    logger.info(f"[{site}] Valid frequency bands: {len(tf_median)}")

    logger.info(f"[{site}] Running test predictions...")
    
    gic_ts = gic_data.gic.sel(device=site, time=test_slice).values
    bx_ts = mag_data.sel(device=mag, time=test_slice).Bx.values
    by_ts = mag_data.sel(device=mag, time=test_slice).By.values
    m = min(len(gic_ts), len(bx_ts), len(by_ts))
    w_ts = tukey_window(m, taper_pct)

    bx_pad, p_ts = zero_pad(taper(bx_ts[:m], w_ts), pad_factor)
    by_pad, _ = zero_pad(taper(by_ts[:m], w_ts), pad_factor)
    bx_f = np.fft.rfft(bx_pad)
    by_f = np.fft.rfft(by_pad)
    freqs_ts = np.fft.rfftfreq(len(bx_pad), dt)

    # Apply median TF
    g_pred_f = np.zeros_like(bx_f, dtype=complex)
    for f_lo, f_hi, zx, zy in tf_median:
        msk = (freqs_ts >= f_lo) & (freqs_ts < f_hi)
        g_pred_f[msk] = zx * bx_f[msk] + zy * by_f[msk]

    g_pred_pad = np.real(np.fft.irfft(g_pred_f))
    pred = edge_blend(g_pred_pad[p_ts : p_ts + m])

    # Create confidence bounds from ensemble spread
    all_ensemble_preds = []
    for (f_lo, f_hi), estimates in tf_ensemble.items():
        if len(estimates) < 10:  # Need reasonable ensemble size
            continue
        
        # Sample subset of estimates for efficiency
        sample_size = min(20, len(estimates))
        sampled_estimates = np.random.choice(len(estimates), sample_size, replace=False)
        
        for idx in sampled_estimates:
            zx, zy = estimates[idx]
            g_pred_f_sample = np.zeros_like(bx_f, dtype=complex)
            msk = (freqs_ts >= f_lo) & (freqs_ts < f_hi)
            g_pred_f_sample[msk] = zx * bx_f[msk] + zy * by_f[msk]
            
            g_pred_pad_sample = np.real(np.fft.irfft(g_pred_f_sample))
            pred_sample = edge_blend(g_pred_pad_sample[p_ts : p_ts + m])
            all_ensemble_preds.append(pred_sample)
    
    if all_ensemble_preds:
        all_ensemble_preds = np.array(all_ensemble_preds)
        pred_lower = np.percentile(all_ensemble_preds, 25, axis=0)
        pred_upper = np.percentile(all_ensemble_preds, 75, axis=0)
    else:
        # Fallback: use fixed percentage bounds
        pred_lower = pred * 0.8
        pred_upper = pred * 1.2

    test_times = gic_data.gic.sel(device=site, time=test_slice).time.values[:m]

    res = gic_ts[:m] - pred
    pe = 1 - np.mean(res**2) / np.var(gic_ts[:m])

    logger.info(f"[{site}] Heyns Ensemble complete: PE = {pe:.4f}")

    return {
        "pred": pred,
        "pred_lower": pred_lower,
        "pred_upper": pred_upper,
        "pe": pe,
        "transfer_median": tf_median,
        "transfer_bounds": tf_bounds,
        "ensemble_sizes": {k: len(v) for k, v in tf_ensemble.items()},
        "n_windows": len(window_starts),
        "n_frequency_bands": len(tf_median),
        "measured": gic_ts[:m],
        "test_times": test_times
    }