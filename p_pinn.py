# Prepare data for physics informed neural network (PINN) training

# %%
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# parent_dir = Path(__file__).resolve().parent.parent
# sys.path.append(str(parent_dir))
from config.settings import get_data_dir, setup_logger

logger = setup_logger(name="logs/pinn")

# Import from your data loading module
from scripts.data_loading import (
    load_tva_gic,
    load_tva_magnetometer,
    find_closest_magnetometers_to_gic,
    get_selected_sites,
    load_and_process_transmission_lines
)

# Load the data
tva_gic = load_tva_gic()
tva_mag = load_tva_magnetometer()
selected_sites = get_selected_sites()
tl_gdf = load_and_process_transmission_lines()
# %%
# Get relationship between GIC monitors and magnetometers/MT sites
closest_mag_to_gic = find_closest_magnetometers_to_gic(
    gic_data=tva_gic, mag_data=tva_mag
)

# get spatial intersection of the gic sites and transmission lines
# %%
gic_data = []
for device, info in closest_mag_to_gic.items():
    gic_data.append({
        'device': device,
        'lat': info['lat'],
        'lon': info['lon'],
        'closest_mag': info['magnetometer'],
        'distance_to_mag': info['distance_to_mag']
    })

# Convert to DataFrame
gic_df = pd.DataFrame(gic_data)

# Create geometry column using lat/lon
geometry = [Point(xy) for xy in zip(gic_df['lon'], gic_df['lat'])]

# Create GeoDataFrame
gic_gdf = gpd.GeoDataFrame(gic_df, geometry=geometry, crs="EPSG:4326")

# %%
def get_intersecting_transmission_lines(
    gic_gdf: gpd.GeoDataFrame,
    tl_gdf: gpd.GeoDataFrame,
    buffer_distance: float = 50,
    wgs84: str = "EPSG:4326",
    proj: str = "EPSG:5070",
):
    """
    Find transmission lines intersecting with buffered GIC devices.
    
    Parameters
    ----------
    gic_gdf : GeoDataFrame
        GeoDataFrame of GIC devices
    tl_gdf : GeoDataFrame
        GeoDataFrame of transmission lines
    buffer_distance : float
        Buffer distance in meters
    wgs84 : str
        WGS84 CRS string
    proj : str
        Projection CRS string
        
    Returns
    -------
    GeoDataFrame
        Contains device and its intersecting transmission lines
    """
    # ----- CRS -----
    gic_proj = gic_gdf.to_crs(wgs84).to_crs(proj)
    tl_proj = tl_gdf.to_crs(wgs84).to_crs(proj)
    
    # ----- buffer + spatial join -----
    gic_proj["buffered"] = gic_proj.geometry.buffer(buffer_distance)
    buffered_gic = gic_proj.set_geometry("buffered")
    
    intersection = gpd.sjoin(
        tl_proj, buffered_gic, how="inner", predicate="intersects"
    )
    
    # if geometry col rename to geom_left for downstream compatibility
    if "geometry" in intersection.columns:
        intersection.rename(columns={"geometry": "geometry_left"}, inplace=True)
    
    inter_gdf = gpd.GeoDataFrame(intersection, geometry="geometry_left", crs=proj)
    
    # Convert back to WGS84
    result_gdf = inter_gdf.to_crs(wgs84)
    
    return result_gdf

# Interesction gdf
intersections_gdf = get_intersecting_transmission_lines(
    gic_gdf, 
    tl_gdf, 
    buffer_distance=150
)
# %%
import folium
from folium import Popup

site = "Widows Creek 2"

# Get the Union GIC device coordinates
union_gic = gic_gdf[gic_gdf['device'] == site]
union_lat = union_gic.geometry.y.iloc[0]
union_lon = union_gic.geometry.x.iloc[0]

# Get the intersecting lines for Union
union_intersections = intersections_gdf[intersections_gdf['device'] == site]

# Create a map centered on the Union GIC device
m = folium.Map(location=[union_lat, union_lon], zoom_start=12, 
              tiles='https://mt1.google.com/vt/lyrs=s,h&x={x}&y={y}&z={z}',
              attr='Google Earth')

# Add the Union GIC device as a circle marker
folium.CircleMarker(
    location=[union_lat, union_lon],
    radius=8,
    color='red',
    fill=True,
    fill_opacity=0.7,
    tooltip=f"{site} GIC Device"
).add_to(m)

# Add a buffer to show the search radius
folium.Circle(
    location=[union_lat, union_lon],
    radius=100,  # 100m buffer
    color='red',
    fill=False,
    weight=1,
    opacity=0.5
).add_to(m)

# Add the intersecting transmission lines
for idx, row in union_intersections.iterrows():
    line_geom = row.geometry_left
    line_coords = [(y, x) for x, y in zip(line_geom.xy[0], line_geom.xy[1])]
    
    popup_text = f"line_id: {row['line_id']}<br>VOLTAGE: {row['VOLTAGE']} kV"
    
    folium.PolyLine(
        line_coords,
        color='blue',
        weight=2,
        opacity=0.8,
        popup=popup_text
    ).add_to(m)

# Add a legend
legend_html = '''
<div style="position: fixed; 
     bottom: 50px; left: 50px; width: 200px; height: 90px; 
     background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
     ">&nbsp; Legend <br>
     &nbsp; <i class="fa fa-circle" style="color:red"></i>&nbsp; Union GIC Device <br>
     &nbsp; <i class="fa fa-minus" style="color:blue"></i>&nbsp; Intersecting Lines <br>
     &nbsp; <i class="fa fa-circle-o" style="color:red"></i>&nbsp; 100m Buffer
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))


m
# %%


# %%
# ── patch once, right after `import bezpy` ─────────────────────────────────
import numpy as np
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

#%%
# Initialize lists to store MT sites and their coordinates
mt_sites = []
site_xys = []

# First, collect all unique MT sites from the clusters
for site, values in closest_mag_to_gic.items():
    mt_cluster = values['mt_cluster']
    
    for site_mag in mt_cluster:
        mt = site_mag['mt_site']
        mt_sites.append(mt)
        site_xys.append((mt.latitude, mt.longitude))

# Convert site_xys to numpy array for easier indexing
site_xys = np.array(site_xys)

# Get the first magnetometer to determine time series length
first_mag = list(closest_mag_to_gic.values())[0]['magnetometer']
time_steps = len(tva_mag.sel(device=first_mag).Bx.values)

# Initialize E_pred with actual number of time steps
E_pred = np.zeros((time_steps, len(mt_sites), 2))

# Now calculate E for each MT site using its closest magnetometer
for i, mt in enumerate(mt_sites):
    # Find the cluster this MT site belongs to
    for site, values in closest_mag_to_gic.items():
        mt_cluster = values['mt_cluster']
        
        for site_mag in mt_cluster:
            if site_mag['mt_site'] == mt:  # Found the right cluster entry
                closest_mag = site_mag['closest_mag']
                
                # Get magnetometer data
                site_Bx = tva_mag.sel(device=closest_mag).Bx.values
                site_By = tva_mag.sel(device=closest_mag).By.values
                
                # Calculate E-field using the MT site's transfer function
                Ex, Ey = mt.convolve_fft(site_Bx, site_By, dt=1.0)
                
                # Store in E_pred (ensuring dimensions match)
                for t in range(min(len(E_pred), len(Ex))):
                    E_pred[t, i, 0] = Ex[t]
                    E_pred[t, i, 1] = Ey[t]
                
                break  # Found the matching entry, no need to keep looking
# %%
import time
import bezpy

# Apply crs and prepare GeoDataFrame
trans_lines_gdf = intersections_gdf.to_crs(epsg=4326)
trans_lines_gdf.rename(columns={"geometry_left": "geometry"}, inplace=True)
trans_lines_gdf.set_geometry("geometry", inplace=True)
trans_lines_gdf["obj"] = trans_lines_gdf.apply(bezpy.tl.TransmissionLine, axis=1)
trans_lines_gdf["length"] = trans_lines_gdf.obj.apply(lambda x: x.length)

# Test E-field
E_test = np.ones((1, len(site_xys), 2))

# Test Delaunay
trans_lines_gdf.obj.apply(lambda x: x.set_delaunay_weights(site_xys))
arr_delaunay = np.zeros(shape=(1, len(trans_lines_gdf)))
for i, tLine in enumerate(trans_lines_gdf.obj):
    try:
        arr_delaunay[:, i] = tLine.calc_voltages(E_test, how="delaunay")
    except:
        arr_delaunay[:, i] = np.nan

# Test NN
trans_lines_gdf.obj.apply(lambda x: x.set_nearest_sites(site_xys))
arr_nn = np.zeros(shape=(1, len(trans_lines_gdf)))
for i, tLine in enumerate(trans_lines_gdf.obj):
    try:
        arr_nn[:, i] = tLine.calc_voltages(E_test, how="nn")
    except:
        arr_nn[:, i] = np.nan

# Create method flag arrays
valid_delaunay = ~np.isnan(arr_delaunay[0, :])
valid_nn = ~np.isnan(arr_nn[0, :])
valid_either = valid_delaunay | valid_nn

# Print summary
print(f"Delaunay valid: {valid_delaunay.sum()}/{len(trans_lines_gdf)}")
print(f"NN valid: {valid_nn.sum()}/{len(trans_lines_gdf)}")
print(f"Either method valid: {valid_either.sum()}/{len(trans_lines_gdf)}")

# Filter using best method
trans_lines_gdf['method'] = 'invalid'
trans_lines_gdf.loc[valid_delaunay, 'method'] = 'delaunay'
trans_lines_gdf.loc[(~valid_delaunay) & valid_nn, 'method'] = 'nn'

# Get valid lines
trans_lines_gdf_valid = trans_lines_gdf[trans_lines_gdf['method'] != 'invalid']

# List of valid line_id values
valid_line_ids = trans_lines_gdf_valid['line_id'].tolist()

# %%
# For calculating voltage components with your patched method
data_path = "data/array_vs.npy"
if os.path.exists(data_path):
    print(f"Loading voltage data from {data_path}")
    array_vs = np.load(data_path)
else:
    # Calculate voltage components
    print("Calculating voltage components...")
    array_vs = np.zeros(shape=(E_pred.shape[0], trans_lines_gdf.shape[0], 2))
    
    for i, tLine in enumerate(trans_lines_gdf.obj):
        method = trans_lines_gdf.iloc[i]['method']
        try:
            Vx, Vy = tLine.calc_voltage_components(E_pred, how=method)
            array_vs[:, i, 0] = Vx
            array_vs[:, i, 1] = Vy
        except Exception as e:
            print(f"Error calculating voltage for line {i}: {str(e)}")
            array_vs[:, i, 0] = np.nan
            array_vs[:, i, 1] = np.nan
    
    # Save results
    os.makedirs("data", exist_ok=True)
    np.save(data_path, array_vs)
    print(f"Saved voltage data to {data_path}")

# %%
# Create line2yard mapping
line2yard = []
yards = ["Bull Run", "Paradise", "Raccoon Mountain", "Shelby", "Union", "Widows Creek 2"]

# First, get the device (yard) for each line in trans_lines_gdf_valid
for idx, row in trans_lines_gdf_valid.iterrows():
    line2yard.append(row['device'])

# Convert to numpy array for consistency with the PINN code
line2yard = np.array(line2yard)

# Verify the mapping
print(f"Number of lines mapped to yards: {len(line2yard)}")
yard_counts = {yard: sum(line2yard == yard) for yard in yards}
print("Lines per yard:")
for yard, count in yard_counts.items():
    print(f"  {yard}: {count}")
    
# %% clean som data
# Check for NaN values in the voltage data
import xarray as xr
import numpy as np

def clean_timeseries(ds, var, max_gap=60):
    """
    Fill short NaN stretches in `ds[var]` (DataArray) *in-place*.

    Parameters
    ----------
    ds : xarray.Dataset
    var : str               variable name (e.g. "Bx")
    max_gap : int           max length (seconds) to patch
    """
    import pandas as pd
    
    da = ds[var]

    # remember original NaNs
    mask_nan = np.isnan(da)

    # Convert max_gap from seconds to proper time delta
    max_gap_td = pd.Timedelta(seconds=max_gap)

    # --- 1. linear interpolation of gaps up to max_gap ---
    da_interp = da.interpolate_na("time", max_gap=max_gap_td, fill_value="extrapolate")

    # --- 2. forward/backward fill up to max_gap ---
    # For ffill and bfill, limit is in terms of data points, not time
    limit_points = max_gap  # Assuming 1 point per second, adjust if needed
    da_filled = (
        da_interp.ffill("time", limit=limit_points)
                 .bfill("time", limit=limit_points)
    )

    # --- 3. optional smoothing at seams (tiny taper) ---
    jump = mask_nan ^ np.isnan(da_filled)
    if jump.any():
        from scipy.signal.windows import tukey
        idx = np.where(jump)[0]
        for k in idx:
            sl = slice(max(k-2,0), min(k+3, len(da_filled)))
            w  = tukey(sl.stop-sl.start, alpha=0.5)
            da_filled[sl] = (da_filled[sl]*w + da[sl]*(1-w)).astype(da.dtype)

    ds[var] = da_filled
    
# fill magnetometer channels
for v in ["Bx", "By"]:
    clean_timeseries(tva_mag, v, max_gap=60)      # ≤60 s gaps only

# fill GIC channels
for dev in tva_gic.device.values:
    clean_timeseries(tva_gic.sel(device=dev), "gic", max_gap=60)
    
# %%
# Find common time range
gic_start = tva_gic.time.values[0]
gic_end = tva_gic.time.values[-1]
mag_start = tva_mag.time.values[0]
mag_end = tva_mag.time.values[-1]

# Use the later start time and earlier end time
common_start = max(gic_start, mag_start)
common_end = min(gic_end, mag_end)

print(f"Common time range: {common_start} to {common_end}")

# Create masks for each dataset
gic_mask = (tva_gic.time >= common_start) & (tva_gic.time <= common_end)
mag_mask = (tva_mag.time >= common_start) & (tva_mag.time <= common_end)

# Check the lengths after masking
gic_length = gic_mask.sum().item()
mag_length = mag_mask.sum().item()
print(f"GIC data points in common range: {gic_length}")
print(f"Magnetometer data points in common range: {mag_length}")

# Make sure the lengths match
if gic_length != mag_length:
    print(f"WARNING: Data lengths don't match after time filtering! GIC: {gic_length}, Mag: {mag_length}")
    # If needed, truncate to shorter length later
    common_length = min(gic_length, mag_length)
    print(f"Will use the shorter length: {common_length}")
else:
    common_length = gic_length
    print(f"Data lengths match: {common_length}")
    
# %%
# Pinn

#!/usr/bin/env python
# ────────────────────────────────────────────────────────────────────────
#  Physics-informed GIC-PINN  (leave-one-yard-out)
# ────────────────────────────────────────────────────────────────────────
import os, sys, time, logging
from pathlib import Path
from math import ceil

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from tqdm.auto import tqdm

# optional nicer logging
try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
except ImportError:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)5s | %(message)s",
                        datefmt="%H:%M:%S")
    logger = logging.getLogger(__name__)

# ── project-specific imports ───────────────────────────────────────────
from utils.signal_processing import (
    tukey_window, taper, zero_pad, edge_blend
)
# tva_mag, tva_gic, common_length, closest_mag_to_gic, … must
# already be in your workspace (e.g. via `from scripts.data_loading import *`)
# ----------------------------------------------------------------------

# ========== 1. helper : B⃗ ➜ E⃗  reference =================================
def bxby_to_ExEy(mt_site, Bx, By, *, dt=1.0,
                 taper_pct=0.05, blend=False):
    """
    Compute E-field at the MT site from 1-s magnetometer data
    *without* any zero-padding.  Uses Site.convolve_fft → Ex,Ey (time).

    Parameters
    ----------
    mt_site    : bezpy.mt.site.Site3d
    Bx, By     : 1-D float32 arrays (same length)
    dt         : sample interval [s]
    taper_pct  : Tukey window percentage for gentle edge-taper
    blend      : optional edge-blend (overlap-add) for very long records

    Returns
    -------
    Ex_t, Ey_t : float32 arrays, same length as Bx / By
    """
    n   = len(Bx)
    win = tukey_window(n, taper_pct)
    Bx_ = taper(Bx, win)
    By_ = taper(By, win)

    # ---- Site's built-in convolution (already Δt-aware) -------------
    Ex_t, Ey_t = mt_site.convolve_fft(Bx_, By_, dt=dt)      # ← time-domain!

    Ex_t, Ey_t = Ex_t.astype("f4"), Ey_t.astype("f4")

    if blend:                      # overlap-blend to soften wrap-around
        Ex_t, Ey_t = edge_blend(Ex_t), edge_blend(Ey_t)

    return Ex_t, Ey_t


def dump_bad(t, name, yard, step):
    bad = ~torch.isfinite(t)
    if bad.any():
        idx = bad.nonzero(as_tuple=False)
        logger.warning(
            f"{yard} • first non-finite in {name} @ minibatch step {step} "
            f"tensor-idx={idx[0].tolist()}  value={t.flatten()[idx[0,0]].item():.3g}"
        )
        return True
    return False


# ========== 2. dataset builder =========================================
win = 256
β_E, λ_Z = 0.2, 1e-3

def build_feature_tensor(yard):
    info     = closest_mag_to_gic[yard]
    mag_name = info["magnetometer"]

    Bx = tva_mag.sel(device=mag_name).Bx.where(mag_mask,  drop=True).values.astype("f4")
    By = tva_mag.sel(device=mag_name).By.where(mag_mask,  drop=True).values.astype("f4")
    y  = tva_gic.gic.sel(device=yard   ).where(gic_mask, drop=True).values.astype("f4")

    L = min(common_length, len(Bx), len(y))
    Bx, By, y = Bx[:L], By[:L], y[:L]

    mt_site = info["mt_site"]
    Ex_ref, Ey_ref = bxby_to_ExEy(mt_site, Bx, By, dt=1.0)

    absB = np.sqrt(Bx**2 + By**2)
    dBx, dBy = np.gradient(Bx), np.gradient(By)
    kern = np.ones(30, "f4") / 30
    stdBx = np.sqrt(np.clip(np.convolve(Bx**2, kern, "same") -
                            np.convolve(Bx, kern, "same")**2, 0, None))
    stdBy = np.sqrt(np.clip(np.convolve(By**2, kern, "same") -
                            np.convolve(By, kern, "same")**2, 0, None))

    mask     = np.array([yy == yard for yy in line2yard])
    Vx_yard  = np.nansum(array_vs[:L, mask, 0], 1)
    Vy_yard  = np.nansum(array_vs[:L, mask, 1], 1)
    n_lines  = mask.sum()
    L_tot    = (trans_lines_gdf_valid.loc[mask, "length"].sum()
                if n_lines else 0.)

    X = np.column_stack([     # 11 engineered channels
        Bx, By, absB, dBx, dBy, stdBx, stdBy,
        Vx_yard, Vy_yard,
        np.full_like(Bx, n_lines, "f4"),
        np.full_like(Bx, L_tot,   "f4")
    ])
    E_ref = np.column_stack([Ex_ref, Ey_ref])
    return X, y, E_ref


# ========== 3. window dataset ==========================================
class WindowDS(Dataset):
    def __init__(self, X, y, E, win=256, stats=None, y_stats=None):
        self.win = win
        L = min(len(X), len(y), len(E))
        self.X = torch.tensor(X[:L], dtype=torch.float32)
        self.y = torch.tensor(y[:L], dtype=torch.float32)
        self.E = torch.tensor(E[:L], dtype=torch.float32)  # (L,2)

        # robust scaling
        if stats is None:
            med = torch.median(self.X, 0).values
            mad = torch.median(torch.abs(self.X - med), 0).values; mad[mad==0]=1.
            stats = (med, 1.4826 * mad)
        if y_stats is None:
            my = torch.median(self.y)
            sy = 1.4826 * torch.median(torch.abs(self.y - my)); sy = sy if sy > 0 else 1.
            y_stats = (my, sy)
        self.stats, self.y_stats = stats, y_stats

    def __len__(self):
        return len(self.X) - self.win

    def __getitem__(self, i):
        sl = slice(i, i + self.win)
        x = ((self.X[sl] - self.stats[0]) / self.stats[1]).T          # (C,win)
        e = self.E[sl].T                                              # (2,win)
        y = (self.y[i + self.win - 1] - self.y_stats[0]) / self.y_stats[1]
        return x, e, y


# ========== 4. model ===================================================
class ImpedanceLayer(nn.Module):
    def __init__(self, Z0):            # Z0 (4,F) complex
        super().__init__()
        self.register_buffer("Z0", torch.tensor(Z0, dtype=torch.cfloat))
        self.dZ = nn.Parameter(torch.zeros_like(self.Z0))

    def forward(self, Bx_f, By_f):
        Z = self.Z0 + self.dZ
        Ex_f = Z[0] * Bx_f + Z[1] * By_f
        Ey_f = Z[2] * Bx_f + Z[3] * By_f
        return Ex_f, Ey_f


class GICPINN(nn.Module):
    def __init__(self, Z0, n_other=11, hidden=64):
        super().__init__()
        self.imp = ImpedanceLayer(Z0)
        self.gru = nn.GRU(
            input_size=2 + 2 + n_other - 2,   # Bx,By, Ex,Ey, other(9)
            hidden_size=hidden, num_layers=2,
            batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):            # x (B,C,win)
        Bx, By = x[:, 0], x[:, 1]
        other  = x[:, 2:]            # (B,C-2,win)

        Bx_f = torch.fft.rfft(Bx, dim=-1)[:, 1:]
        By_f = torch.fft.rfft(By, dim=-1)[:, 1:]
        Ex_f, Ey_f = self.imp(Bx_f, By_f)

        pad = lambda f: torch.cat([torch.zeros_like(f[..., :1]), f], -1)
        Ex = torch.fft.irfft(pad(Ex_f), n=Bx.size(1), dim=-1).real
        Ey = torch.fft.irfft(pad(Ey_f), n=By.size(1), dim=-1).real

        seq = torch.cat([Bx.unsqueeze(1), By.unsqueeze(1),
                         Ex.unsqueeze(1), Ey.unsqueeze(1), other], 1
                        ).permute(0, 2, 1)           # (B,win,C_gru)

        _, h = self.gru(seq)
        gic = self.fc(torch.cat([h[-2], h[-1]], 1)).squeeze(-1)
        return gic, (Ex, Ey)


# ---------- utility ----------------------------------------------------
def assert_finite(*tensors, where=""):
    for t in tensors:
        if not torch.isfinite(t).all():
            raise RuntimeError(f"Non-finite detected {where}, shape={t.shape}")


def save_best(state_dict, yard):
    out = Path("models"); out.mkdir(exist_ok=True)
    f   = out / f"GICPINN_{yard.replace(' ','_')}_best.pt"
    torch.save(state_dict, f); logger.info(f"✓ saved → {f}")


# ========== 5. trainer =================================================
def train_pinn(model, dl_tr, dl_val, epochs=40, lr=3e-4, yard=""):
    opt = torch.optim.AdamW(model.parameters(), lr)
    best, patience, best_state = 1e9, 6, None

    for ep in range(epochs):
        # --- train ---
        model.train(); train_loss=0
        for step, (xb, eref, yb) in enumerate(tqdm(dl_tr, desc=f"train {yard} ep{ep}", leave=False)):
            xb, eref, yb = xb.to(device), eref.to(device), yb.to(device)
            if dump_bad(xb,   "X",   yard, step): break
            if dump_bad(eref, "Eref", yard, step): break
            if dump_bad(yb,   "y",   yard, step): break
            assert_finite(xb, yb, where="input")
            pred, (Ex, Ey) = model(xb)
            assert_finite(pred, Ex, Ey, where="output")

            L_gic = F.mse_loss(pred, yb)
            L_E   = F.mse_loss(torch.stack([Ex, Ey], 1), eref)
            L_Z   = torch.mean(model.imp.dZ.abs() ** 2)
            loss  = L_gic + β_E * L_E + λ_Z * L_Z

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
            train_loss += loss.item()

        # --- validate ---
        model.eval(); val_loss=0
        with torch.no_grad():
            for xb, eref, yb in dl_val:
                xb, eref, yb = xb.to(device), eref.to(device), yb.to(device)
                p,(Ex,Ey) = model(xb)
                val_loss += (F.mse_loss(p,yb) +
                             β_E*F.mse_loss(torch.stack([Ex,Ey],1),eref)
                             ).item()
        val_loss /= len(dl_val)
        logger.info(f"{yard} ep{ep:02d} | train {train_loss/len(dl_tr):.5f}"
                    f" | val {val_loss:.5f}")

        if val_loss < best - 1e-4:
            best, patience = val_loss, 6
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            save_best(best_state, yard)
        else:
            patience -= 1
        if patience == 0:
            break

    model.load_state_dict(best_state)


# ========== 6. leave-one-yard-out main ================================
def main():
    yards   = ["Bull Run", "Paradise", "Union", "Raccoon Mountain"]
    batch   = 1024
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    winF    = ceil(win / 2)
    dummy_Z = np.zeros((4, winF), dtype=np.complex64)
    results = {}

    for test_yard in yards:
        logger.info(f"▶ leave-out: {test_yard}")
        train_yards = [y for y in yards if y != test_yard]

        # build datasets ------------------------------------------------
        ds_train = []
        for y in train_yards:
            X, y_gic, E = build_feature_tensor(y)
            ds_train.append(WindowDS(X, y_gic, E, win))
            logger.debug(f"{y}  windows:{len(ds_train[-1]):,}")

        X_te, y_te, E_te = build_feature_tensor(test_yard)
        ds_te = WindowDS(X_te, y_te, E_te, win,
                         stats=ds_train[0].stats, y_stats=ds_train[0].y_stats)

        full = ConcatDataset(ds_train)
        n_val = max(1, int(0.1 * len(full)))
        ds_val, ds_tr = random_split(full, [n_val, len(full) - n_val])

        dl_tr  = DataLoader(ds_tr , batch_size=min(batch, len(ds_tr)),
                            shuffle=True,  drop_last=True)
        dl_val = DataLoader(ds_val, batch_size=min(batch, len(ds_val)),
                            shuffle=False)
        dl_te  = DataLoader(ds_te , batch_size=min(batch, len(ds_te)),
                            shuffle=False)

        # model ---------------------------------------------------------
        model = GICPINN(dummy_Z, n_other=11, hidden=64).to(device)
        train_pinn(model, dl_tr, dl_val, epochs=40, lr=3e-4, yard=test_yard)

        # evaluate ------------------------------------------------------
        model.eval(); preds=[]; obs=[]
        with torch.no_grad():
            for xb, _, yb in dl_te:
                p,_ = model(xb.to(device))
                preds.append(p.cpu()); obs.append(yb)
        preds, obs = torch.cat(preds), torch.cat(obs)
        pe = 1 - torch.mean((obs - preds) ** 2) / torch.var(obs)
        results[test_yard] = pe.item()
        logger.info(f"★ {test_yard:<15s} PE = {pe:.3f}")

    # summary ----------------------------------------------------------
    logger.info("\n==== summary ====")
    for y, pe in results.items():
        logger.info(f"{y:<15s} {pe:.3f}")

if __name__ == "__main__":
    main()


# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd

def evaluate_all_gic_sites(model, window_size=256, batch_size=1024):
    """
    Evaluate the trained model on all available GIC sites in the closest_mag_to_gic dictionary.
    
    Parameters:
    -----------
    model : GICPINN
        The trained model
    window_size : int
        Window size used during training
    batch_size : int
        Batch size for evaluation
        
    Returns:
    --------
    dict
        Dictionary with site names as keys and evaluation metrics as values
    """
    # Make sure model is in evaluation mode
    model.eval()
    
    # Get all available GIC sites
    all_sites = list(closest_mag_to_gic.keys())
    print(f"Evaluating model on {len(all_sites)} GIC sites")
    
    # Create results dictionary
    results = {}
    
    # Evaluate each site
    for site in all_sites:
        print(f"\nEvaluating site: {site}")
        
        try:
            # Build features for this site
            X, y = build_feature_tensor(site)
            
            # Check if we have enough data
            if len(X) <= window_size:
                print(f"WARNING: Data for {site} too short ({len(X)}) for window size {window_size}, skipping")
                continue
            
            # Create dataset
            # Use stats from the training sets if you have them, otherwise compute new ones
            if 'ds_train_list' in globals() and len(ds_train_list) > 0:
                stats = ds_train_list[0].stats
                y_stats = ds_train_list[0].y_stats
                print("Using stats from training data")
            else:
                stats = None
                y_stats = None
                print("Computing new stats")
                
            ds = WindowDS(X, y, win=window_size, stats=stats, y_stats=y_stats)
            print(f"Created dataset with {len(ds)} samples")
            
            # Create dataloader
            dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=False)
            
            # Evaluate
            preds, obs, times = [], [], []
            with torch.no_grad():
                for i, (xb, yb) in enumerate(dl):
                    try:
                        p, _ = model(xb.to(device))
                        preds.append(p.cpu())
                        obs.append(yb)
                        
                        # Store time indices if needed for plotting
                        if i == 0:  # Just for the first batch
                            # Get actual timestamps if available
                            if hasattr(tva_gic, 'time'):
                                # Adjust for window size
                                start_idx = window_size - 1
                                end_idx = start_idx + len(yb)
                                batch_times = tva_gic.time.values[start_idx:end_idx]
                                times.append(batch_times)
                    except Exception as e:
                        print(f"Error during evaluation batch {i}: {e}")
                        continue
            
            if not preds:
                print(f"No valid predictions for {site}, skipping")
                continue
                
            # Concatenate predictions and observations
            all_preds = torch.cat(preds)
            all_obs = torch.cat(obs)
            
            # Convert back to original scale
            if y_stats is not None:
                orig_preds = all_preds * y_stats[1] + y_stats[0]
                orig_obs = all_obs * y_stats[1] + y_stats[0]
            else:
                orig_preds = all_preds
                orig_obs = all_obs
            
            # Calculate metrics
            mse = torch.mean((all_obs - all_preds) ** 2).item()
            rmse = torch.sqrt(torch.mean((orig_obs - orig_preds) ** 2)).item()
            var_obs = torch.var(all_obs).item()
            pe = 1 - mse / var_obs if var_obs > 0 else float('nan')
            mae = torch.mean(torch.abs(orig_obs - orig_preds)).item()
            correlation = torch.corrcoef(torch.stack([orig_obs, orig_preds]))[0, 1].item()
            
            # Store results
            results[site] = {
                'PE': pe,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'Correlation': correlation,
                'predictions': orig_preds.numpy(),
                'observations': orig_obs.numpy(),
                'times': np.concatenate(times) if times else None
            }
            
            print(f"Site: {site:15s} | PE: {pe:.3f} | RMSE: {rmse:.3f} | Corr: {correlation:.3f}")
            
            # Plot
            plt.figure(figsize=(12, 6))
            plt.plot(orig_obs.numpy(), label='Actual')
            plt.plot(orig_preds.numpy(), label='Predicted')
            plt.legend()
            plt.title(f"GIC Predictions for {site} (PE = {pe:.3f}, Corr = {correlation:.3f})")
            plt.ylabel('GIC (A)')
            plt.xlabel('Time Steps')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(f"plots/gic_prediction_{site.replace(' ', '_').lower()}.png")
            plt.close()
            
        except Exception as e:
            print(f"Error processing site {site}: {e}")
            continue
    
    # Print summary table
    print("\nEvaluation Summary")
    print("-" * 80)
    print(f"{'Site':15s} | {'PE':>8s} | {'RMSE':>8s} | {'MAE':>8s} | {'Corr':>8s}")
    print("-" * 80)
    
    pe_values = []
    for site, metrics in results.items():
        pe = metrics['PE']
        rmse = metrics['RMSE']
        mae = metrics['MAE']
        corr = metrics['Correlation']
        pe_values.append(pe)
        print(f"{site:15s} | {pe:8.3f} | {rmse:8.3f} | {mae:8.3f} | {corr:8.3f}")
    
    mean_pe = np.nanmean(pe_values)
    print("-" * 80)
    print(f"{'Average':15s} | {mean_pe:8.3f} | {'-':8s} | {'-':8s} | {'-':8s}")
    
    # Create plots directory if it doesn't exist
    import os
    os.makedirs("plots", exist_ok=True)
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    sites = list(results.keys())
    pe_values = [results[s]['PE'] for s in sites]
    
    # Sort by PE value
    sorted_indices = np.argsort(pe_values)[::-1]  # Descending
    sorted_sites = [sites[i] for i in sorted_indices]
    sorted_pe = [pe_values[i] for i in sorted_indices]
    
    # Plot bar chart
    bars = plt.bar(sorted_sites, sorted_pe)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.ylim(min(-0.5, min(sorted_pe) - 0.1), max(1.0, max(sorted_pe) + 0.1))
    
    # Add values on top of bars
    for bar, pe in zip(bars, sorted_pe):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height + 0.05 if height > 0 else height - 0.1, 
                f'{pe:.2f}', 
                ha='center', va='bottom' if height > 0 else 'top')
    
    plt.title("Prediction Efficiency (PE) by GIC Site")
    plt.ylabel("Prediction Efficiency (1 - MSE/Variance)")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save summary plot
    plt.savefig("plots/gic_prediction_summary.png")
    plt.close()
    
    return results

# Create directories
import os
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Evaluate all sites
if 'model' in locals() and model is not None:
    print("Evaluating trained model on all sites...")
    all_results = evaluate_all_gic_sites(model, window_size=win, batch_size=batch)
    
    # Save results
    try:
        import pickle
        with open("results/gic_evaluation_results.pkl", "wb") as f:
            pickle.dump(all_results, f)
        print("Saved evaluation results to results/gic_evaluation_results.pkl")
    except Exception as e:
        print(f"Error saving results: {e}")
else:
    print("No trained model available for evaluation")