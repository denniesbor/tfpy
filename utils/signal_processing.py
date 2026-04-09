"""
Author: Dennies Bor
Role:   Signal processing utilities for the GIC analysis pipeline.
"""

import numpy as np
import pandas as pd
import scipy.signal
from bezpy.mt import Site


# Monkey-patch Site with frequency-domain convolution
def convolve_fft_freq(self, mag_x, mag_y, dt=60):
    """Return (freqs, Ex_f, Ey_f) via MT impedance convolution in the frequency domain."""
    N     = 2 ** (int(np.log2(len(mag_x))) + 2)
    freqs = np.fft.rfftfreq(N, d=dt)
    Z     = self.calcZ(freqs)
    bx    = np.fft.rfft(mag_x, n=N)
    by    = np.fft.rfft(mag_y, n=N)
    return freqs, Z[0] * bx + Z[1] * by, Z[2] * bx + Z[3] * by

Site.convolve_fft_freq = convolve_fft_freq


def tukey_window(n, pct=0.05):
    """Return a Tukey window of length n with edge taper fraction pct."""
    return scipy.signal.windows.tukey(n, alpha=2 * pct)


def taper(x, w):
    """Multiply signal x by window w."""
    return x[: len(w)] * w


def zero_pad(x, factor=1.0):
    """Symmetrically zero-pad x by factor * len(x) on each side; return (padded, pad_width)."""
    p = int(len(x) * factor)
    return np.pad(x, (p, p), "constant"), p


def edge_blend(y, frac=0.02):
    """Taper signal to zero at both edges to suppress FFT wrap-around artefacts."""
    n = len(y)
    m = int(n * frac)
    if m == 0:
        return y
    w       = np.ones(n)
    w[:m]   = np.linspace(0, 1, m)
    w[-m:]  = w[:m][::-1]
    return y * w


def robust_complex_lsq(A, b, δ=1.345, max_iter=20, tol=1e-6):
    """Huber-weighted IRLS for robust least-squares; operates on A (m×k) and b (m,)."""
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    for _ in range(max_iter):
        r  = b - A @ x
        σ  = 1.4826 * np.median(np.abs(r)) or 1.0
        w  = np.where(np.abs(r) <= δ * σ, 1.0, (δ * σ) / np.abs(r))
        x_new = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)[0]
        if np.linalg.norm(x_new - x) < tol * np.linalg.norm(x):
            break
        x = x_new
    return x


def bxby_to_ExEy(mt_site, Bx, By, *, dt=1.0, taper_pct=0.05, blend=False):
    """Compute time-domain E-field at an MT site from magnetometer Bx/By."""
    win    = tukey_window(len(Bx), taper_pct)
    Ex, Ey = mt_site.convolve_fft(taper(Bx, win), taper(By, win), dt=dt)
    Ex, Ey = Ex.astype("f4"), Ey.astype("f4")
    if blend:
        Ex, Ey = edge_blend(Ex), edge_blend(Ey)
    return Ex, Ey


def fix_nans(arr, max_gap_pts=600):
    """Fill NaN gaps up to max_gap_pts with linear interpolation then forward/back fill."""
    if not np.isnan(arr).any():
        return
    idx  = np.arange(arr.size)
    good = np.isfinite(arr)
    arr[~good] = np.interp(idx[~good], idx[good], arr[good])
    s      = pd.Series(arr)
    arr[:] = s.ffill(limit=max_gap_pts).bfill(limit=max_gap_pts).to_numpy(dtype=arr.dtype)


def clean_timeseries(ds, var, max_gap=60):
    """Fill short NaN stretches in an xarray DataArray in-place."""
    da      = ds[var]
    mask    = np.isnan(da)
    da_int  = da.interpolate_na("time", max_gap=pd.Timedelta(seconds=max_gap), fill_value="extrapolate")
    da_fill = da_int.ffill("time", limit=max_gap).bfill("time", limit=max_gap)

    # Smooth any sharp transitions introduced at fill boundaries
    jump = mask ^ np.isnan(da_fill)
    if jump.any():
        from scipy.signal.windows import tukey
        for k in np.where(jump)[0]:
            sl = slice(max(k - 2, 0), min(k + 3, len(da_fill)))
            w  = tukey(sl.stop - sl.start, alpha=0.5)
            da_fill[sl] = (da_fill[sl] * w + da[sl] * (1 - w)).astype(da.dtype)

    ds[var] = da_fill


def patch_bezpy_tl():
    """Patch TransmissionLine with calc_voltage_components returning separate (Vx, Vy)."""
    from bezpy.tl import TransmissionLine

    def calc_voltage_components(self, E, how="delaunay"):
        """Return (Vx, Vy) in volts keeping Ex·dx and Ey·dy contributions separate."""
        if how == "nn":
            if self.nearest_sites is None:
                raise RuntimeError("Call .set_nearest_sites() first.")
            E3d = np.atleast_3d(E)[:, self.nearest_sites[:-1], :]

        elif how == "1d":
            if self.regions1d is None:
                raise RuntimeError("Call .set_1d_regions() first.")
            E3d = np.atleast_3d(E)[:, self.regions1d[:-1], :]

        elif how == "delaunay":
            if self.delaunay_vtx is None or self.delaunay_wts is None:
                raise RuntimeError("Call .set_delaunay_weights() first.")
            E3d = np.sum(
                np.atleast_3d(E)[:, self.delaunay_vtx[:-1], :]
                * self.delaunay_wts[np.newaxis, :-1, :, np.newaxis],
                axis=2,
            )
            E3d[:, np.any(self.delaunay_wts[:-1] < 0, axis=1), :] = np.nan
        else:
            raise ValueError("how must be 'nn', '1d', or 'delaunay'.")

        Vx = np.nansum(E3d[..., 0] * self.dl[:, 0], axis=1) / 1000.0
        Vy = np.nansum(E3d[..., 1] * self.dl[:, 1], axis=1) / 1000.0
        return (float(Vx), float(Vy)) if Vx.size == 1 else (Vx, Vy)

    TransmissionLine.calc_voltage_components = calc_voltage_components