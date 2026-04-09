"""
Author: Dennies Bor
Role:   Geographic utility functions for GIC site and transmission line analysis.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium


def haversine_dist(lat1, lon1, lat2, lon2):
    """Return great-circle distance in km between two lat/lon points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def prepare_gic_gdf(closest_mag_to_gic):
    """Build a GeoDataFrame of GIC monitoring sites from the proximity mapping."""
    records = [
        {
            "device": device,
            "lat": info["lat"],
            "lon": info["lon"],
            "closest_mag": info["magnetometer"],
            "distance_to_mag": info["distance_to_mag"],
        }
        for device, info in closest_mag_to_gic.items()
    ]
    df = pd.DataFrame(records)
    geometry = [Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def get_intersecting_transmission_lines(
    gic_gdf: gpd.GeoDataFrame,
    tl_gdf: gpd.GeoDataFrame,
    buffer_distance: float = 50,
    wgs84: str = "EPSG:4326",
    proj: str = "EPSG:5070",
):
    """Return transmission lines that intersect a buffered radius around each GIC device."""
    gic_proj = gic_gdf.to_crs(proj)
    tl_proj = tl_gdf.to_crs(proj)

    gic_proj["buffered"] = gic_proj.geometry.buffer(buffer_distance)
    buffered = gic_proj.set_geometry("buffered")

    inter = gpd.sjoin(tl_proj, buffered, how="inner", predicate="intersects")
    if "geometry" in inter.columns:
        inter.rename(columns={"geometry": "geometry_left"}, inplace=True)

    return gpd.GeoDataFrame(inter, geometry="geometry_left", crs=proj).to_crs(wgs84)


def create_intersection_map(gic_gdf, intersections_gdf, site="Widows Creek 2"):
    """Build a folium map of a GIC device and its intersecting transmission lines."""
    site_row = gic_gdf[gic_gdf["device"] == site]
    lat = site_row.geometry.y.iloc[0]
    lon = site_row.geometry.x.iloc[0]

    m = folium.Map(
        location=[lat, lon],
        zoom_start=12,
        tiles="https://mt1.google.com/vt/lyrs=s,h&x={x}&y={y}&z={z}",
        attr="Google Earth",
    )

    folium.CircleMarker(
        location=[lat, lon],
        radius=8,
        color="red",
        fill=True,
        fill_opacity=0.7,
        tooltip=f"{site} GIC Device",
    ).add_to(m)

    folium.Circle(
        location=[lat, lon],
        radius=100,
        color="red",
        fill=False,
        weight=1,
        opacity=0.5,
    ).add_to(m)

    for _, row in intersections_gdf[intersections_gdf["device"] == site].iterrows():
        coords = [
            (y, x) for x, y in zip(row.geometry_left.xy[0], row.geometry_left.xy[1])
        ]
        folium.PolyLine(
            coords,
            color="blue",
            weight=2,
            opacity=0.8,
            popup=f"line_id: {row['line_id']}<br>VOLTAGE: {row['VOLTAGE']} kV",
        ).add_to(m)

    m.get_root().html.add_child(
        folium.Element(
            """
        <div style="position:fixed;bottom:50px;left:50px;width:200px;height:90px;
            background-color:white;border:2px solid grey;z-index:9999;font-size:14px;">
            &nbsp; Legend <br>
            &nbsp; <i class="fa fa-circle" style="color:red"></i>&nbsp; GIC Device <br>
            &nbsp; <i class="fa fa-minus" style="color:blue"></i>&nbsp; Intersecting Lines <br>
            &nbsp; <i class="fa fa-circle-o" style="color:red"></i>&nbsp; 100m Buffer
        </div>
    """
        )
    )

    return m
