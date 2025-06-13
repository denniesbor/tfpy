"""
Signal processing utilities for GIC analysis.

This module contains general-purpose signal processing functions used
across the GIC analysis pipeline.
"""

import numpy as np
import scipy.signal
import pandas as pd
from bezpy.mt import Site


def convolve_fft_freq(self, mag_x, mag_y, dt=60):
    """
    Calculate frequency-domain convolution for MT site response.
    """
    N0 = len(mag_x)
    N = 2 ** (int(np.log2(N0)) + 2)
    freqs = np.fft.rfftfreq(N, d=dt)
    Z = self.calcZ(freqs)

    bx = np.fft.rfft(mag_x, n=N)
    by = np.fft.rfft(mag_y, n=N)

    Ex = Z[0] * bx + Z[1] * by
    Ey = Z[2] * bx + Z[3] * by
    return freqs, Ex, Ey


# Monkey patch the Site class
Site.convolve_fft_freq = convolve_fft_freq


def tukey_window(n, pct=0.05):
    """Reusable Tukey window (edge-only taper)."""
    return scipy.signal.windows.tukey(n, alpha=2 * pct)


def taper(x, w):  # element-wise multiply
    """Apply a window function to a signal."""
    return x[: len(w)] * w


def zero_pad(x, factor=1.0):
    """Symmetric zero-padding."""
    p = int(len(x) * factor)
    return np.pad(x, (p, p), "constant"), p


def edge_blend(y, frac=0.02):
    """Blend to zero at edges to avoid FFT wrap-around."""
    n = len(y)
    m = int(n * frac)
    if m == 0:
        return y
    w = np.ones(n)
    w[:m] = np.linspace(0, 1, m)
    w[-m:] = w[:m][::-1]
    return y * w


def robust_complex_lsq(A, b, δ=1.345, max_iter=20, tol=1e-6):
    """
    Huber-weighted IRLS without building an N×N diagonal matrix.
    Works in-place on A (m×4) and b (m,) with m up to ~1e5.
    """
    x = np.linalg.lstsq(A, b, rcond=None)[0]  # initial LSQ
    for _ in range(max_iter):
        r = b - A @ x
        σ = 1.4826 * np.median(np.abs(r)) or 1.0  # scale (MAD)
        w = np.where(np.abs(r) <= δ * σ, 1.0, (δ * σ) / np.abs(r))

        Aw = A * w[:, None]  # apply row weights
        bw = b * w
        x_new = np.linalg.lstsq(Aw, bw, rcond=None)[0]

        if np.linalg.norm(x_new - x) < tol * np.linalg.norm(x):
            break
        x = x_new
    return x


# ----- Added functions from PINN code -----

def bxby_to_ExEy(mt_site, Bx, By, *, dt=1.0, taper_pct=0.05, blend=False):
    """
    Compute E-field at the MT site from magnetometer data without any zero-padding.
    Uses Site.convolve_fft → Ex,Ey (time).
    """
    n = len(Bx)
    win = tukey_window(n, taper_pct)
    Bx_ = taper(Bx, win)
    By_ = taper(By, win)

    # Site's built-in convolution (already Δt-aware)
    Ex_t, Ey_t = mt_site.convolve_fft(Bx_, By_, dt=dt)

    Ex_t, Ey_t = Ex_t.astype("f4"), Ey_t.astype("f4")

    if blend:  # overlap-blend to soften wrap-around
        Ex_t, Ey_t = edge_blend(Ex_t), edge_blend(Ey_t)

    return Ex_t, Ey_t


def fix_nans(arr, max_gap_pts=600):
    """
    Fill NaN gaps in array up to max_gap_pts length with linear interpolation,
    followed by forward and backward filling.
    """
    if not np.isnan(arr).any():
        return
        
    # Linear interpolation
    idx = np.arange(arr.size)
    good = np.isfinite(arr)
    arr[~good] = np.interp(idx[~good], idx[good], arr[good])
    
    # Handle very long runs with pandas
    s = pd.Series(arr)
    arr[:] = (s.ffill(limit=max_gap_pts)
               .bfill(limit=max_gap_pts)
               .to_numpy(dtype=arr.dtype))


def clean_timeseries(ds, var, max_gap=60):
    """
    Fill short NaN stretches in `ds[var]` (xarray DataArray) in-place.
    """
    import pandas as pd
    
    da = ds[var]

    # Remember original NaNs
    mask_nan = np.isnan(da)

    # Convert max_gap from seconds to proper time delta
    max_gap_td = pd.Timedelta(seconds=max_gap)

    # Linear interpolation of gaps up to max_gap
    da_interp = da.interpolate_na("time", max_gap=max_gap_td, fill_value="extrapolate")

    # Forward/backward fill up to max_gap
    # For ffill and bfill, limit is in terms of data points, not time
    limit_points = max_gap  # Assuming 1 point per second, adjust if needed
    da_filled = (
        da_interp.ffill("time", limit=limit_points)
                 .bfill("time", limit=limit_points)
    )

    # Optional smoothing at seams (tiny taper)
    jump = mask_nan ^ np.isnan(da_filled)
    if jump.any():
        from scipy.signal.windows import tukey
        idx = np.where(jump)[0]
        for k in idx:
            sl = slice(max(k-2,0), min(k+3, len(da_filled)))
            w = tukey(sl.stop-sl.start, alpha=0.5)
            da_filled[sl] = (da_filled[sl]*w + da[sl]*(1-w)).astype(da.dtype)

    ds[var] = da_filled


def patch_bezpy_tl():
    """
    Patch the bezpy TransmissionLine class with improved voltage component calculation.
    
    This enhances the bezpy.tl.TransmissionLine class with a method that calculates
    voltage components separately (Vx, Vy) rather than just the total voltage.
    """
    from bezpy.tl import TransmissionLine

    def calc_voltage_components(self, E, how="delaunay"):
        """
        Return tuple (Vx, Vy)  in volts, keeping the Ex·dx and Ey·dy pieces
        separate.

        Parameters
        ----------
        E   : ndarray (nt, n_sites, 2)   electric field mV/km  ([..., 0]=Ex, [...,1]=Ey)
        how : 'nn' | '1d' | 'delaunay'   interpolation choice (same as calc_voltages)
        """
        # ---- interpolate E to every segment start-point ---------------------
        if how == "nn":
            if self.nearest_sites is None:
                raise RuntimeError("call .set_nearest_sites() first")
            E3d = np.atleast_3d(E)[:, self.nearest_sites[:-1], :]

        elif how == "1d":
            if self.regions1d is None:
                raise RuntimeError("call .set_1d_regions() first")
            E3d = np.atleast_3d(E)[:, self.regions1d[:-1], :]

        elif how == "delaunay":
            if self.delaunay_vtx is None or self.delaunay_wts is None:
                raise RuntimeError("call .set_delaunay_weights() first")
            E3d = np.sum(np.atleast_3d(E)[:, self.delaunay_vtx[:-1], :] *
                        self.delaunay_wts[np.newaxis, :-1, :, np.newaxis], axis=2)
            E3d[:, np.any(self.delaunay_wts[:-1] < 0, axis=1), :] = np.nan
        else:
            raise ValueError("how must be 'nn', '1d' or 'delaunay'")

        # ---- integrate separately -------------------------------------------
        # self.dl has shape (n_segments, 2) with columns (dLat=dx, dLon=dy) in km
        Vx = np.nansum(E3d[..., 0] * self.dl[:, 0], axis=1) / 1000.0   # mV/km * km → V
        Vy = np.nansum(E3d[..., 1] * self.dl[:, 1], axis=1) / 1000.0

        # return scalar if nt==1 for backward compatibility
        if Vx.size == 1:
            return float(Vx), float(Vy)
        return Vx, Vy

    # Monkey patch the tl class
    TransmissionLine.calc_voltage_components = calc_voltage_components