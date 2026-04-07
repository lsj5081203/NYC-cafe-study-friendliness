"""Unit tests for src/scoring.py."""

import numpy as np
import pytest
from src.scoring import (
    compute_acoustic_score,
    compute_study_friendliness,
    classify_score,
    WIFI_SATURATION_COUNT,
    EATERY_SATURATION_COUNT,
)


# --- compute_acoustic_score ---

class TestComputeAcousticScore:
    def test_all_air_conditioner(self):
        """All class-0 (weight 0.1) → score 90."""
        score = compute_acoustic_score(np.array([0, 0, 0]))
        assert score == pytest.approx(90.0)

    def test_all_gunshot(self):
        """All class-6 (weight 1.0) → score 0."""
        score = compute_acoustic_score(np.array([6, 6, 6]))
        assert score == pytest.approx(0.0)

    def test_mixed_classes(self):
        """50% class-0 (0.1) + 50% class-6 (1.0) → score 45."""
        score = compute_acoustic_score(np.array([0, 6]))
        assert score == pytest.approx(45.0)

    def test_empty_class_counts(self):
        """Empty class_counts → neutral score 50."""
        score = compute_acoustic_score(np.array([]), class_counts={})
        assert score == pytest.approx(50.0)

    def test_class_counts_dict(self):
        """Using class_counts dict directly."""
        # 30% car_horn (0.9) + 70% engine_idling (0.15)
        score = compute_acoustic_score(np.array([]), class_counts={1: 3, 5: 7})
        expected = (1.0 - (0.3 * 0.9 + 0.7 * 0.15)) * 100
        assert score == pytest.approx(expected)

    def test_unknown_class_falls_back(self):
        """Unknown class ID uses default weight 0.5."""
        score = compute_acoustic_score(np.array([]), class_counts={99: 1})
        assert score == pytest.approx(50.0)


# --- compute_study_friendliness ---

class TestComputeStudyFriendliness:
    def test_acoustic_only(self):
        """No spatial features → 0.9 * acoustic."""
        score = compute_study_friendliness(80, wifi_count=0, eatery_count=0)
        assert score == pytest.approx(72.0)

    def test_wifi_bonus(self):
        """WIFI_SATURATION_COUNT hotspots → full bonus (+10 points at default weight)."""
        score = compute_study_friendliness(80, wifi_count=WIFI_SATURATION_COUNT, eatery_count=0)
        assert score == pytest.approx(82.0)

    def test_eatery_penalty(self):
        """EATERY_SATURATION_COUNT eateries → full penalty (-5 points at default weight)."""
        score = compute_study_friendliness(80, wifi_count=0, eatery_count=EATERY_SATURATION_COUNT)
        assert score == pytest.approx(67.0)

    def test_clamped_to_zero(self):
        """Score cannot go below 0."""
        score = compute_study_friendliness(0, wifi_count=0, eatery_count=100)
        assert score == 0.0

    def test_clamped_to_hundred(self):
        """Score cannot exceed 100."""
        score = compute_study_friendliness(100, wifi_count=20, eatery_count=0)
        assert score == 100.0


# --- classify_score ---

class TestClassifyScore:
    @pytest.mark.parametrize("score,expected", [
        (90, "Excellent"),
        (75, "Excellent"),   # boundary
        (60, "Good"),
        (55, "Good"),        # boundary
        (40, "Fair"),
        (35, "Fair"),        # boundary
        (25, "Poor"),
        (20, "Poor"),        # boundary
        (10, "Avoid"),
        (0, "Avoid"),
    ])
    def test_labels(self, score, expected):
        assert classify_score(score) == expected
