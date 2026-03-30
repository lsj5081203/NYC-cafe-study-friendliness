"""
Study-friendliness scoring for NYC cafe locations.

Combines acoustic predictions (urban sound classification) with spatial
context features (Wi-Fi density, business density) to produce a
study-friendliness score for each cafe.
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


def compute_study_friendliness(
    acoustic_score,
    wifi_count,
    eatery_count,
    wifi_weight=0.1,
    eatery_weight=-0.05,
    acoustic_weight=0.9,
):
    """Combine acoustic and spatial features into a final study-friendliness score.

    Higher Wi-Fi density is a positive signal (infrastructure for studying).
    Higher eatery density is a slight negative signal (more foot traffic, noise).

    Args:
        acoustic_score: Acoustic score (0-100).
        wifi_count: Number of Wi-Fi hotspots nearby.
        eatery_count: Number of eateries nearby.
        wifi_weight: Weight for Wi-Fi bonus.
        eatery_weight: Weight for eatery penalty.
        acoustic_weight: Weight for acoustic score (should dominate).

    Returns:
        Final study-friendliness score (0-100).
    """
    # Normalize spatial features (rough normalization for NYC)
    wifi_bonus = min(wifi_count / 10.0, 1.0) * 100  # cap at 10 hotspots nearby
    eatery_penalty = min(eatery_count / 50.0, 1.0) * 100  # cap at 50 eateries

    score = (
        acoustic_weight * acoustic_score
        + wifi_weight * wifi_bonus
        + eatery_weight * eatery_penalty
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
