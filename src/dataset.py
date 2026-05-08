"""
UrbanSound8K dataset loader.

Loads audio clips from the UrbanSound8K dataset while respecting the
predefined 10-fold splits.
"""

import os

import librosa
import numpy as np
import pandas as pd


CLASS_NAMES = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}

NUM_CLASSES = 10


def load_metadata(data_dir):
    """Load the UrbanSound8K metadata CSV."""
    csv_path = os.path.join(data_dir, "metadata", "UrbanSound8K.csv")
    return pd.read_csv(csv_path)


def load_audio(file_path, sr=22050, duration=4.0):
    """Load, resample, and pad/truncate one audio file to a fixed length."""
    n_samples = int(sr * duration)
    y, _ = librosa.load(file_path, sr=sr, duration=duration)

    if len(y) == 0:
        raise ValueError(f"Audio file returned empty waveform: {file_path}")

    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)), mode="constant")
    else:
        y = y[:n_samples]

    return y


def get_fold_data(data_dir, folds, sr=22050, duration=4.0):
    """Load all audio clips from the requested UrbanSound8K folds.

    Files that cannot be loaded are skipped with a warning, so returned data
    may contain fewer rows than the metadata subset.
    """
    df = load_metadata(data_dir)
    df_folds = df[df["fold"].isin(folds)].reset_index(drop=True)

    audios = []
    labels = []
    valid_indices = []

    for idx, row in df_folds.iterrows():
        file_path = os.path.join(
            data_dir, "audio", f"fold{row['fold']}", row["slice_file_name"]
        )
        try:
            y = load_audio(file_path, sr=sr, duration=duration)
            audios.append(y)
            labels.append(row["classID"])
            valid_indices.append(idx)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")

    metadata = df_folds.loc[valid_indices].reset_index(drop=True)
    return audios, np.array(labels), metadata


def get_default_split():
    """Return the default train/validation/test fold split."""
    return {
        "train": [1, 2, 3, 4, 5, 6, 7, 8],
        "val": [9],
        "test": [10],
    }
