"""
Geographic utility functions for the application.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium import Popup


def haversine_dist(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance between two points in km."""
    R = 6371  # Earth radius in km
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2_rad - lat1_rad, lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def prepare_gic_gdf(closest_mag_to_gic):
    """
    Create a GeoDataFrame for GIC monitoring sites
    
    """
    gic_data = []
    for device, info in closest_mag_to_gic.items():
        gic_data.append({
            'device': device,
            'lat': info['lat'],
            'lon': info['lon'],
            'closest_mag': info['magnetometer'],
            'distance_to_mag': info['distance_to_mag']
        })
    
    gic_df = pd.DataFrame(gic_data)
    geometry = [Point(xy) for xy in zip(gic_df['lon'], gic_df['lat'])]
    gic_gdf = gpd.GeoDataFrame(gic_df, geometry=geometry, crs="EPSG:4326")
    
    return gic_gdf


def get_intersecting_transmission_lines(
    gic_gdf: gpd.GeoDataFrame,
    tl_gdf: gpd.GeoDataFrame,
    buffer_distance: float = 50,
    wgs84: str = "EPSG:4326",
    proj: str = "EPSG:5070",
):
    """
    Find transmission lines intersecting with buffered GIC devices.

    """
    gic_proj = gic_gdf.to_crs(wgs84).to_crs(proj)
    tl_proj = tl_gdf.to_crs(wgs84).to_crs(proj)
    
    gic_proj["buffered"] = gic_proj.geometry.buffer(buffer_distance)
    buffered_gic = gic_proj.set_geometry("buffered")
    
    intersection = gpd.sjoin(
        tl_proj, buffered_gic, how="inner", predicate="intersects"
    )
    
    # if geometry col rename to geom_left for downstream compatibility
    if "geometry" in intersection.columns:
        intersection.rename(columns={"geometry": "geometry_left"}, inplace=True)
    
    inter_gdf = gpd.GeoDataFrame(intersection, geometry="geometry_left", crs=proj)
    
    result_gdf = inter_gdf.to_crs(wgs84)
    
    return result_gdf


def create_intersection_map(gic_gdf, intersections_gdf, site="Widows Creek 2"):
    """
    Create a folium map showing GIC device and intersecting transmission lines
    
    """
    site_gic = gic_gdf[gic_gdf['device'] == site]
    site_lat = site_gic.geometry.y.iloc[0]
    site_lon = site_gic.geometry.x.iloc[0]

    site_intersections = intersections_gdf[intersections_gdf['device'] == site]

    m = folium.Map(location=[site_lat, site_lon], zoom_start=12, 
                  tiles='https://mt1.google.com/vt/lyrs=s,h&x={x}&y={y}&z={z}',
                  attr='Google Earth')

    folium.CircleMarker(
        location=[site_lat, site_lon],
        radius=8,
        color='red',
        fill=True,
        fill_opacity=0.7,
        tooltip=f"{site} GIC Device"
    ).add_to(m)

    folium.Circle(
        location=[site_lat, site_lon],
        radius=100,  # 100m buffer
        color='red',
        fill=False,
        weight=1,
        opacity=0.5
    ).add_to(m)

    for idx, row in site_intersections.iterrows():
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

    legend_html = '''
    <div style="position: fixed; 
        bottom: 50px; left: 50px; width: 200px; height: 90px; 
        background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
        ">&nbsp; Legend <br>
        &nbsp; <i class="fa fa-circle" style="color:red"></i>&nbsp; GIC Device <br>
        &nbsp; <i class="fa fa-minus" style="color:blue"></i>&nbsp; Intersecting Lines <br>
        &nbsp; <i class="fa fa-circle-o" style="color:red"></i>&nbsp; 100m Buffer
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m