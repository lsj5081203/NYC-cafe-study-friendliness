"""
NYC Open Data spatial features for cafe study-friendliness.

Downloads and processes Wi-Fi hotspot locations and eatery directory
from NYC Open Data, then computes density features around cafe locations.
"""

import os
import numpy as np
import pandas as pd
import requests


# NYC Open Data SODA API endpoints
WIFI_ENDPOINT = "https://data.cityofnewyork.us/resource/yjub-udmw.json"
EATERIES_ENDPOINT = "https://data.cityofnewyork.us/resource/8792-ebcp.json"


def download_wifi_hotspots(cache_path=None, limit=5000):
    """Download NYC Wi-Fi hotspot locations from NYC Open Data.

    Args:
        cache_path: If provided, save/load from this CSV path.
        limit: Max number of records to fetch.

    Returns:
        DataFrame with columns including latitude, longitude, location info.
    """
    if cache_path and os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    print("Downloading NYC Wi-Fi hotspot data...")
    response = requests.get(WIFI_ENDPOINT, params={"$limit": limit})
    response.raise_for_status()
    df = pd.DataFrame(response.json())

    # Convert lat/lon to numeric
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude"])

    if cache_path:
        dir_name = os.path.dirname(cache_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"  Cached to {cache_path}")

    print(f"  {len(df)} hotspots loaded.")
    return df


def download_eateries(cache_path=None, limit=10000):
    """Download NYC eatery directory from NYC Open Data.

    Args:
        cache_path: If provided, save/load from this CSV path.
        limit: Max number of records to fetch.

    Returns:
        DataFrame with eatery locations.
    """
    if cache_path and os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    print("Downloading NYC eatery directory...")
    response = requests.get(EATERIES_ENDPOINT, params={"$limit": limit})
    response.raise_for_status()
    df = pd.DataFrame(response.json())

    # Try to extract lat/lon (dataset may use different column names)
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if cache_path:
        dir_name = os.path.dirname(cache_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"  Cached to {cache_path}")

    print(f"  {len(df)} eateries loaded.")
    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute haversine distance in meters between two points.

    Args:
        lat1, lon1: Coordinates of point 1 (degrees).
        lat2, lon2: Coordinates of point 2 (degrees).

    Returns:
        Distance in meters.
    """
    R = 6371000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def compute_density(cafe_lat, cafe_lon, locations_df, radius_m=200):
    """Count number of locations within a given radius of a cafe.

    Args:
        cafe_lat: Cafe latitude.
        cafe_lon: Cafe longitude.
        locations_df: DataFrame with 'latitude' and 'longitude' columns.
        radius_m: Radius in meters.

    Returns:
        Count of locations within the radius.
    """
    distances = haversine_distance(
        cafe_lat, cafe_lon,
        locations_df["latitude"].values,
        locations_df["longitude"].values,
    )
    return int(np.sum(distances <= radius_m))


def build_spatial_features(cafe_locations, wifi_df, eateries_df, radius_m=200):
    """Compute spatial context features for each cafe location.

    Args:
        cafe_locations: DataFrame with 'name', 'latitude', 'longitude' columns.
        wifi_df: Wi-Fi hotspots DataFrame.
        eateries_df: Eateries DataFrame.
        radius_m: Radius for density computation.

    Returns:
        DataFrame with columns: name, latitude, longitude, wifi_count, eatery_count.
    """
    eateries_valid = (
        eateries_df.dropna(subset=["latitude", "longitude"])
        if "latitude" in eateries_df.columns and "longitude" in eateries_df.columns
        else None
    )

    results = []
    for _, cafe in cafe_locations.iterrows():
        wifi_count = compute_density(
            cafe["latitude"], cafe["longitude"], wifi_df, radius_m
        )

        eatery_count = 0
        if eateries_valid is not None:
            eatery_count = compute_density(
                cafe["latitude"], cafe["longitude"], eateries_valid, radius_m
            )

        results.append({
            "name": cafe["name"],
            "latitude": cafe["latitude"],
            "longitude": cafe["longitude"],
            "wifi_count": wifi_count,
            "eatery_count": eatery_count,
        })

    return pd.DataFrame(results)
