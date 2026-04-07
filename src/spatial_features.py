"""\nNYC Open Data spatial features for cafe study-friendliness.\n\nDownloads and processes Wi-Fi hotspot locations and eatery directory\nfrom NYC Open Data, then computes density features around cafe locations.\n"""\n\nimport os\nimport numpy as np\nimport pandas as pd\nimport requests\n\n\n# NYC Open Data SODA API endpoints\nWIFI_ENDPOINT = "https://data.cityofnewyork.us/resource/yjub-udmw.json"\nEATERIES_ENDPOINT = "https://data.cityofnewyork.us/resource/8792-ebcp.json"\n\n\ndef download_wifi_hotspots(cache_path=None, limit=5000):\n    """Download NYC Wi-Fi hotspot locations from NYC Open Data.\n\n    Args:\n        cache_path: If provided, save/load from this CSV path.\n        limit: Max number of records to fetch.\n\n    Returns:\n        DataFrame with columns including latitude, longitude, location info.\n    """\n    if cache_path and os.path.exists(cache_path):\n        return pd.read_csv(cache_path)\n\n    print("Downloading NYC Wi-Fi hotspot data...")\n    response = requests.get(WIFI_ENDPOINT, params={"$limit": limit})\n    response.raise_for_status()\n    df = pd.DataFrame(response.json())\n\n    # Convert lat/lon to numeric\n    for col in ["latitude", "longitude"]:\n        if col in df.columns:\n            df[col] = pd.to_numeric(df[col], errors="coerce")\n\n    df = df.dropna(subset=["latitude", "longitude"])\n\n    if cache_path:\n        dir_name = os.path.dirname(cache_path)\n        if dir_name:\n            os.makedirs(dir_name, exist_ok=True)\n        df.to_csv(cache_path, index=False)\n        print(f"  Cached to {cache_path}")\n\n    if len(df) >= limit:\n        print(f"  Warning: returned {len(df)} rows (hit limit={limit}). Results may be truncated.")\n\n    print(f"  {len(df)} hotspots loaded.")\n    return df\n\n\ndef download_eateries(cache_path=None, limit=50000):\n    """Download NYC eatery directory from NYC Open Data.\n\n    Args:\n        cache_path: If provided, save/load from this CSV path.\n        limit: Max number of records to fetch.\n\n    Returns:\n        DataFrame with eatery locations.\n    """\n    if cache_path and os.path.exists(cache_path):\n        return pd.read_csv(cache_path)\n\n    print("Downloading NYC eatery directory...")\n    response = requests.get(EATERIES_ENDPOINT, params={"$limit": limit})\n    response.raise_for_status()\n    df = pd.DataFrame(response.json())\n\n    # Try to extract lat/lon (dataset may use different column names)\n    for col in ["latitude", "longitude"]:\n        if col in df.columns:\n            df[col] = pd.to_numeric(df[col], errors="coerce")\n\n    if cache_path:\n        dir_name = os.path.dirname(cache_path)\n        if dir_name:\n            os.makedirs(dir_name, exist_ok=True)\n        df.to_csv(cache_path, index=False)\n        print(f"  Cached to {cache_path}")\n\n    if len(df) >= limit:\n        print(f"  Warning: returned {len(df)} rows (hit limit={limit}). Results may be truncated.")\n\n    print(f"  {len(df)} eateries loaded.")\n    return df\n\n\ndef haversine_distance(lat1, lon1, lat2, lon2):\n    """Compute haversine distance in meters between two points.\n\n    Args:\n        lat1, lon1: Coordinates of point 1 (degrees).\n        lat2, lon2: Coordinates of point 2 (degrees).\n\n    Returns:\n        Distance in meters.\n    """\n    R = 6371000  # Earth radius in meters\n    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])\n    dlat = lat2 - lat1\n    dlon = lon2 - lon1\n    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2\n    c = 2 * np.arcsin(np.sqrt(a))\n    return R * c\n\n\ndef compute_density(cafe_lat, cafe_lon, locations_df, radius_m=200):\n    """Count number of locations within a given radius of a cafe.\n\n    Args:\n        cafe_lat: Cafe latitude.\n        cafe_lon: Cafe longitude.\n        locations_df: DataFrame with 'latitude' and 'longitude' columns.\n        radius_m: Radius in meters.\n\n    Returns:\n        Count of locations within the radius.\n    """\n    distances = haversine_distance(\n        cafe_lat, cafe_lon,\n        locations_df["latitude"].values,\n        locations_df["longitude"].values,\n    )\n    return int(np.sum(distances <= radius_m))\n\n\ndef build_spatial_features(cafe_locations, wifi_df, eateries_df, radius_m=200):\n    """Compute spatial context features for each cafe location.\n\n    Args:\n        cafe_locations: DataFrame with 'name', 'latitude', 'longitude' columns.\n        wifi_df: Wi-Fi hotspots DataFrame.\n        eateries_df: Eateries DataFrame.\n        radius_m: Radius for density computation.\n\n    Returns:\n        DataFrame with columns: name, latitude, longitude, wifi_count, eatery_count.\n    """\n    eateries_valid = (\n        eateries_df.dropna(subset=["latitude", "longitude"])\n        if "latitude" in eateries_df.columns and "longitude" in eateries_df.columns\n        else None\n    )\n\n    results = []\n    for _, cafe in cafe_locations.iterrows():\n        wifi_count = compute_density(\n            cafe["latitude"], cafe["longitude"], wifi_df, radius_m\n        )\n\n        eatery_count = 0\n        if eateries_valid is not None:\n            eatery_count = compute_density(\n                cafe["latitude"], cafe["longitude"], eateries_valid, radius_m\n            )\n\n        results.append({\n            "name": cafe["name"],\n            "latitude": cafe["latitude"],\n            "longitude": cafe["longitude"],\n            "wifi_count": wifi_count,\n            "eatery_count": eatery_count,\n        })\n\n    return pd.DataFrame(results)\n
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

    Rows missing lat/lon are dropped with dropna() before the result is
    cached, so the cached CSV only contains geometrically valid records.
    Re-downloading and re-caching will always produce a clean file.

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

    df = df.dropna(subset=[c for c in ["latitude", "longitude"] if c in df.columns])

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

    eateries_df is pre-filtered for valid lat/lon into eateries_valid once
    before the per-cafe loop. This avoids re-running dropna() on every
    iteration and is safe because download_eateries() already applies
    dropna() before caching; the in-loop guard is a defensive fallback.

    Args:
        cafe_locations: DataFrame with 'name', 'latitude', 'longitude' columns.
        wifi_df: Wi-Fi hotspots DataFrame.
        eateries_df: Eateries DataFrame.
        radius_m: Radius for density computation.

    Returns:
        DataFrame with columns: name, latitude, longitude, wifi_count, eatery_count.
    """
    # Pre-filter once outside the loop
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
        if eateries_valid is not None and len(eateries_valid) > 0:
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
