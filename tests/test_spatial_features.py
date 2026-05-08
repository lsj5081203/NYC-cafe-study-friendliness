"""Unit tests for src/spatial_features.py; no API calls required."""

import pandas as pd
import pytest

from src.spatial_features import build_spatial_features, compute_density, haversine_distance


class TestHaversineDistance:
    def test_same_point_is_zero(self):
        d = haversine_distance(40.7128, -74.0060, 40.7128, -74.0060)
        assert d == pytest.approx(0.0, abs=0.1)

    def test_known_nyc_distance(self):
        d = haversine_distance(40.7580, -73.9855, 40.7648, -73.9738)
        assert 900 < d < 1300

    def test_symmetry(self):
        d1 = haversine_distance(40.7128, -74.0060, 40.7580, -73.9855)
        d2 = haversine_distance(40.7580, -73.9855, 40.7128, -74.0060)
        assert d1 == pytest.approx(d2)

    def test_large_distance(self):
        d = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        assert 3_900_000 < d < 4_000_000


class TestComputeDensity:
    @pytest.fixture
    def locations_df(self):
        return pd.DataFrame({
            "latitude": [40.7128, 40.7130, 40.7200, 40.8000],
            "longitude": [-74.0060, -74.0062, -74.0100, -74.0500],
        })

    def test_count_within_radius(self, locations_df):
        count = compute_density(40.7128, -74.0060, locations_df, radius_m=200)
        assert count == 2

    def test_large_radius_captures_all(self, locations_df):
        count = compute_density(40.7128, -74.0060, locations_df, radius_m=50_000)
        assert count == 4

    def test_zero_radius(self, locations_df):
        count = compute_density(40.7128, -74.0060, locations_df, radius_m=0)
        assert count >= 0


class TestBuildSpatialFeatures:
    def test_output_columns(self):
        cafes = pd.DataFrame({
            "name": ["Cafe A"],
            "latitude": [40.7128],
            "longitude": [-74.0060],
        })
        wifi = pd.DataFrame({
            "latitude": [40.7130],
            "longitude": [-74.0062],
        })
        eateries = pd.DataFrame({
            "latitude": [40.7129],
            "longitude": [-74.0061],
        })
        result = build_spatial_features(cafes, wifi, eateries, radius_m=200)
        assert list(result.columns) == [
            "name", "latitude", "longitude", "wifi_count", "eatery_count"
        ]
        assert len(result) == 1

    def test_multiple_cafes(self):
        cafes = pd.DataFrame({
            "name": ["Cafe A", "Cafe B"],
            "latitude": [40.7128, 40.8000],
            "longitude": [-74.0060, -74.0500],
        })
        wifi = pd.DataFrame({
            "latitude": [40.7130],
            "longitude": [-74.0062],
        })
        eateries = pd.DataFrame({
            "latitude": [40.7129],
            "longitude": [-74.0061],
        })
        result = build_spatial_features(cafes, wifi, eateries, radius_m=200)
        assert len(result) == 2
        assert result.iloc[0]["wifi_count"] >= 1
        assert result.iloc[1]["wifi_count"] == 0

    def test_eateries_without_lat_lon(self):
        cafes = pd.DataFrame({
            "name": ["Cafe A"],
            "latitude": [40.7128],
            "longitude": [-74.0060],
        })
        wifi = pd.DataFrame({
            "latitude": [40.7130],
            "longitude": [-74.0062],
        })
        eateries = pd.DataFrame({"some_col": ["value"]})
        result = build_spatial_features(cafes, wifi, eateries, radius_m=200)
        assert result.iloc[0]["eatery_count"] == 0
