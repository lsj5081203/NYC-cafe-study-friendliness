"""
Study-friendliness scoring for NYC cafe locations.

Combines acoustic predictions (urban sound classification) with spatial
context features (Wi-Fi density, business density) to produce a
study-friendliness score for each cafe.

Design notes:
- DISTRACTION_WEIGHTS are subjective heuristics assigned by the project
  authors. They are not derived from user studies or perceptual data.
  Treat them as reasonable defaults that should be recalibrated if
  ground-truth distraction ratings become available.
- WIFI_SATURATION_COUNT and EATERY_SATURATION_COUNT are empirical
  placeholders chosen to reflect typical NYC neighborhood densities.
  They should be recalibrated from actual data distributions before
  any production use.
"""

import numpy as np


# Distraction weights for each UrbanSound8K class (0 = not distracting, 1 = very distracting)
DISTRACTION_WEIGHTS = {
    0: 0.1,   # air_conditioner — steady hum, barely noticeable
    1: 0.9,   # car_horn — sudden, very distracting
    2: 0.5,   # children_playing — moderate
    3: 0.6,   # dog_bark — sudden, moderately distracting
    4: 0.95,  # drilling — loud, sustained, very distracting
    5: 0.15,  # engine_idling — steady low rumble
    6: 1.0,   # gun_shot — extremely distracting
    7: 0.95,  # jackhammer — loud, sustained, very distracting
    8: 0.85,  # siren — loud, attention-grabbing
    9: 0.4,   # street_music — can be pleasant or distracting
}


def compute_acoustic_score(predictions, class_counts=None):
    """Compute an acoustic study-friendliness score from sound classifications.

    Higher score = more study-friendly (less distracting).

    Class IDs not present in DISTRACTION_WEIGHTS (e.g., IDs from a model
    trained on a different taxonomy) default to a neutral weight of 0.5,
    representing moderate distraction. This prevents hard failures when
    the classifier produces out-of-vocabulary predictions.

    Args:
        predictions: Array of predicted class IDs for audio windows
            from a single cafe recording.
        class_counts: Optional dict of {classID: count}. If provided,
            used instead of predictions.

    Returns:
        Score between 0 (very distracting) and 100 (very quiet/study-friendly).
    """
    if class_counts is None:
        unique, counts = np.unique(predictions, return_counts=True)
        class_counts = dict(zip(unique, counts))

    total = sum(class_counts.values())
    if total == 0:
        return 50.0  # No data, neutral score

    # Weighted distraction: proportion of each class * its distraction weight
    weighted_distraction = sum(
        (count / total) * DISTRACTION_WEIGHTS.get(cls, 0.5)
        for cls, count in class_counts.items()
    )

    # Invert: low distraction = high study-friendliness
    score = (1.0 - weighted_distraction) * 100
    return max(0.0, min(100.0, score))


# Spatial normalization caps (empirically chosen for NYC scale; recalibrate from data).
# Wi-Fi: public hotspots within 200m — 10 is a reasonable upper bound for most neighborhoods.
# Eateries: businesses within 200m — 50 is achievable in Midtown, extreme elsewhere.
WIFI_SATURATION_COUNT = 10
EATERY_SATURATION_COUNT = 50


def compute_study_friendliness(
    acoustic_score,
    wifi_count,
    eatery_count,
    wifi_weight=0.1,
    eatery_weight=0.05,
    acoustic_weight=0.9,
):
    """Combine acoustic and spatial features into a final study-friendliness score.

    Higher Wi-Fi density is a positive signal (infrastructure for studying).
    Higher eatery density is a slight negative signal (more foot traffic, noise).

    Weight formula (defaults):
        score = 0.9 * acoustic_score
              + 0.1 * wifi_bonus       # max +10 points
              - 0.05 * eatery_penalty  # max -5 points

    where wifi_bonus and eatery_penalty are each normalized to [0, 100]
    using WIFI_SATURATION_COUNT and EATERY_SATURATION_COUNT respectively.
    The acoustic score therefore contributes 90% of the final value and
    completely dominates the spatial adjustments.

    Args:
        acoustic_score: Acoustic score (0-100).
        wifi_count: Number of Wi-Fi hotspots nearby.
        eatery_count: Number of eateries nearby.
        wifi_weight: Fractional weight for Wi-Fi bonus (adds to score).
            Default 0.1 caps the maximum Wi-Fi contribution at +10 points.
        eatery_weight: Fractional weight for eatery penalty (subtracts from score).
            Default 0.05 caps the maximum eatery penalty at -5 points.
        acoustic_weight: Weight for acoustic score. Default 0.9.

    Returns:
        Final study-friendliness score (0-100).
    """
    # Normalize spatial features to [0, 100]
    wifi_bonus = min(wifi_count / WIFI_SATURATION_COUNT, 1.0) * 100
    eatery_penalty = min(eatery_count / EATERY_SATURATION_COUNT, 1.0) * 100

    score = (
        acoustic_weight * acoustic_score
        + wifi_weight * wifi_bonus
        - eatery_weight * eatery_penalty
    )

    return max(0.0, min(100.0, score))


def classify_score(score):
    """Convert a numeric score to a human-readable label.

    Args:
        score: Study-friendliness score (0-100).

    Returns:
        String label.
    """
    if score >= 75:
        return "Excellent"
    elif score >= 55:
        return "Good"
    elif score >= 35:
        return "Fair"
    elif score >= 20:
        return "Poor"
    else:
        return "Avoid"
