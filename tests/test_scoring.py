"""Unit tests for src/scoring.py — no dataset required."""

import pytest
from src.scoring import (
    compute_acoustic_score,
    compute_study_friendliness,
    classify_score,
    WIFI_SATURATION_COUNT,
    EATERY_SATURATION_COUNT,
)


class TestComputeAcousticScore:
    def test_all_air_conditioner(self):
        # class 0 weight = 0.1  →  score = (1 - 0.1) * 100 = 90.0
        assert compute_acoustic_score(None, class_counts={0: 10}) == pytest.approx(90.0)

    def test_all_gunshot(self):
        # class 6 weight = 1.0  →  score = 0.0
        assert compute_acoustic_score(None, class_counts={6: 5}) == pytest.approx(0.0)

    def test_mixed_classes(self):
        # 50% class-0 (0.1) + 50% class-6 (1.0) → distraction = 0.55 → score = 45.0
        assert compute_acoustic_score(None, class_counts={0: 5, 6: 5}) == pytest.approx(45.0)

    def test_empty_class_counts(self):
        # No data → neutral score 50.0
        assert compute_acoustic_score(None, class_counts={}) == pytest.approx(50.0)

    def test_predictions_array(self):
        import numpy as np
        # All class-8 (siren, weight=0.85) → score = 15.0
        preds = np.array([8, 8, 8, 8])
        assert compute_acoustic_score(preds) == pytest.approx(15.0)

    def test_unknown_class_uses_default_weight(self):
        # Unknown class 99 → default weight 0.5 → score = 50.0
        assert compute_acoustic_score(None, class_counts={99: 10}) == pytest.approx(50.0)

    def test_score_clamped_to_zero(self):
        result = compute_acoustic_score(None, class_counts={6: 100})
        assert result >= 0.0

    def test_score_clamped_to_hundred(self):
        result = compute_acoustic_score(None, class_counts={0: 100})
        assert result <= 100.0


class TestComputeStudyFriendliness:
    def test_acoustic_only(self):
        # No wifi/eateries: 0.9 * 80 = 72.0
        assert compute_study_friendliness(80, 0, 0) == pytest.approx(72.0)

    def test_wifi_bonus(self):
        # 10 hotspots → wifi_bonus = 100, 0.9*80 + 0.1*100 = 82.0
        assert compute_study_friendliness(80, 10, 0) == pytest.approx(82.0)

    def test_eatery_penalty(self):
        # 50 eateries → eatery_penalty = 100, 0.9*80 - 0.05*100 = 67.0
        assert compute_study_friendliness(80, 0, 50) == pytest.approx(67.0)
        """WIFI_SATURATION_COUNT hotspots → full bonus (+10 points at default weight)."""
        score = compute_study_friendliness(80, wifi_count=WIFI_SATURATION_COUNT, eatery_count=0)
        assert score == pytest.approx(82.0)

    def test_eatery_penalty(self):
        """EATERY_SATURATION_COUNT eateries → full penalty (-5 points at default weight)."""
        score = compute_study_friendliness(80, wifi_count=0, eatery_count=EATERY_SATURATION_COUNT)
        assert score == pytest.approx(67.0)

    def test_clamped_to_zero(self):
        result = compute_study_friendliness(0, 0, 1000)
        assert result == pytest.approx(0.0)

    def test_clamped_to_hundred(self):
        result = compute_study_friendliness(100, 1000, 0)
        assert result == pytest.approx(100.0)


class TestClassifyScore:
    @pytest.mark.parametrize("score,expected", [
        (100, "Excellent"),
        (75,  "Excellent"),
        (74,  "Good"),
        (55,  "Good"),
        (54,  "Fair"),
        (35,  "Fair"),
        (34,  "Poor"),
        (20,  "Poor"),
        (19,  "Avoid"),
        (0,   "Avoid"),
    ])
    def test_labels(self, score, expected):
        assert classify_score(score) == expected
