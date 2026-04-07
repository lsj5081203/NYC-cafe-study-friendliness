"""
End-to-end inference script: cafe recordings → study-friendliness scores.

Usage:
    python scripts/run_inference.py --model data/models/rf_model.pkl \
        --recordings data/cafe_recordings \
        --metadata data/cafe_recordings/cafe_metadata.csv \
        --output data/results/cafe_scores.csv
"""

import argparse
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

# Ensure project root is on path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio_features import extract_mfcc_batch
from src.baseline_model import load_model
from src.scoring import compute_acoustic_score, compute_study_friendliness, classify_score
from src.spatial_features import (
    download_wifi_hotspots,
    download_eateries,
    build_spatial_features,
)


# Feature config must match what was used at training time
SR = 22050
DURATION = 4.0          # window length in seconds
HOP_DURATION = 2.0      # sliding window hop (50% overlap)
N_MFCC = 40


def check_ffmpeg():
    """Raise if ffmpeg is not available (required for M4A/AAC decoding)."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "ffmpeg not found on PATH. It is required to decode .m4a files.\n"
            "Activate the conda environment: conda activate cafe-study"
        )


def load_and_window(file_path, sr=SR, window_sec=DURATION, hop_sec=HOP_DURATION):
    """Load a full audio file and split into fixed-length windows.

    Args:
        file_path: Path to audio file (any format supported by librosa/ffmpeg).
        sr: Target sample rate.
        window_sec: Window duration in seconds.
        hop_sec: Hop size in seconds.

    Returns:
        List of numpy arrays, each of length sr * window_sec.
        Returns empty list if the file is too short for one window.
    """
    y, _ = librosa.load(str(file_path), sr=sr)
    if len(y) == 0:
        print(f"  Warning: {file_path.name} returned empty audio, skipping.")
        return []

    window_samples = int(sr * window_sec)
    hop_samples = int(sr * hop_sec)

    windows = []
    for start in range(0, len(y) - window_samples + 1, hop_samples):
        window = y[start : start + window_samples]
        windows.append(window)

    # Include a final partial window if the recording has content beyond the last full hop
    if not windows:
        # Recording shorter than one window — pad it
        padded = np.pad(y, (0, window_samples - len(y)), mode="constant")
        windows.append(padded)

    return windows


def run_inference(model_path, recordings_dir, metadata_path, output_path,
                  wifi_cache="data/cache/wifi.csv", eatery_cache="data/cache/eateries.csv"):
    check_ffmpeg()

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading model from {model_path} ...")
    model = load_model(model_path)

    # ── Load cafe metadata ───────────────────────────────────────────────────
    meta = pd.read_csv(metadata_path)
    recordings_dir = Path(recordings_dir)

    # ── Spatial features (fetched once for all unique cafe locations) ────────
    print("Fetching spatial features ...")
    wifi_df = download_wifi_hotspots(cache_path=wifi_cache)
    eatery_df = download_eateries(cache_path=eatery_cache)

    cafe_locs = meta[["name", "latitude", "longitude"]].drop_duplicates("name")
    spatial_df = build_spatial_features(cafe_locs, wifi_df, eatery_df)
    spatial_lookup = spatial_df.set_index("name")

    # ── Per-recording inference ──────────────────────────────────────────────
    records = []
    for _, row in meta.iterrows():
        audio_path = recordings_dir / row["filename"]
        if not audio_path.exists():
            print(f"  Warning: {audio_path} not found, skipping.")
            continue

        print(f"  Processing {row['filename']} ({row.get('recording_type', '?')}) ...")
        windows = load_and_window(audio_path)
        if not windows:
            continue

        X = extract_mfcc_batch(windows, sr=SR, n_mfcc=N_MFCC)
        predictions = model.predict(X)
        acoustic_score = compute_acoustic_score(predictions)

        spatial = spatial_lookup.loc[row["name"]]
        final_score = compute_study_friendliness(
            acoustic_score,
            wifi_count=spatial["wifi_count"],
            eatery_count=spatial["eatery_count"],
        )
        label = classify_score(final_score)

        records.append({
            "name": row["name"],
            "filename": row["filename"],
            "recording_type": row.get("recording_type", ""),
            "n_windows": len(windows),
            "acoustic_score": round(acoustic_score, 2),
            "wifi_count": int(spatial["wifi_count"]),
            "eatery_count": int(spatial["eatery_count"]),
            "final_score": round(final_score, 2),
            "label": label,
        })

    if not records:
        print("No recordings processed. Check that filenames in metadata match files in recordings_dir.")
        return

    results = pd.DataFrame(records)

    # ── Aggregate: average inside + outside per cafe ─────────────────────────
    summary = (
        results.groupby("name")
        .agg(
            avg_acoustic_score=("acoustic_score", "mean"),
            avg_final_score=("final_score", "mean"),
            wifi_count=("wifi_count", "first"),
            eatery_count=("eatery_count", "first"),
            recordings=("filename", "count"),
        )
        .reset_index()
    )
    summary["label"] = summary["avg_final_score"].apply(classify_score)

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    summary_path = output_path.with_stem(output_path.stem + "_summary")
    summary.to_csv(summary_path, index=False)

    print(f"\nResults saved to {output_path}")
    print(f"Summary saved to {summary_path}")
    print("\nCafe scores:")
    print(summary[["name", "avg_final_score", "label"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score NYC cafes from audio recordings.")
    parser.add_argument("--model", required=True, help="Path to saved .pkl model file")
    parser.add_argument("--recordings", default="data/cafe_recordings",
                        help="Directory containing cafe audio files")
    parser.add_argument("--metadata", default="data/cafe_recordings/cafe_metadata.csv",
                        help="CSV with name, latitude, longitude, filename, recording_type")
    parser.add_argument("--output", default="data/results/cafe_scores.csv",
                        help="Output CSV path")
    parser.add_argument("--wifi-cache", default="data/cache/wifi.csv")
    parser.add_argument("--eatery-cache", default="data/cache/eateries.csv")
    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        recordings_dir=args.recordings,
        metadata_path=args.metadata,
        output_path=args.output,
        wifi_cache=args.wifi_cache,
        eatery_cache=args.eatery_cache,
    )
