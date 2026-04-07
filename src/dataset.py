"""
UrbanSound8K dataset loader.

Loads audio clips from the UrbanSound8K dataset, respecting the predefined
10-fold cross-validation splits to avoid data leakage.

Reference: J. Salamon, C. Jacoby, and J. P. Bello,
"A Dataset and Taxonomy for Urban Sound Research," ACM MM 2014.
"""

import os
import numpy as np
import pandas as pd
import librosa


# UrbanSound8K class labels (classID -> class name)
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
    """Load UrbanSound8K metadata CSV.

    Args:
        data_dir: Path to UrbanSound8K root directory
            (containing metadata/ and audio/ subdirs).

    Returns:
        DataFrame with columns: slice_file_name, fsID, start, end,
        salience, fold, classID, class.
    """
    csv_path = os.path.join(data_dir, "metadata", "UrbanSound8K.csv")
    return pd.read_csv(csv_path)


def load_audio(file_path, sr=22050, duration=4.0):
    """Load an audio file, resampled and padded/truncated to fixed length.

    Args:
        file_path: Path to audio file.
        sr: Target sample rate.
        duration: Target duration in seconds.

    Returns:
        numpy array of shape (n_samples,) where n_samples = sr * duration.

    Raises:
        ValueError: If the file loads successfully but librosa returns an
            empty waveform (len == 0). This can happen with corrupt files,
            unsupported codecs, or clips that consist entirely of silence
            that librosa trims to zero length.
    """
    n_samples = int(sr * duration)
    y, _ = librosa.load(file_path, sr=sr, duration=duration)

    if len(y) == 0:
        raise ValueError(f"Audio file returned empty waveform: {file_path}")

    # Pad if shorter than target duration
    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)), mode="constant")
    # Truncate if longer
    else:
        y = y[:n_samples]

    return y


def get_fold_data(data_dir, folds, sr=22050, duration=4.0):
    """Load all audio clips from specified folds.

    Files that cannot be loaded (e.g., missing files, corrupt audio, empty
    waveforms) are skipped with a printed warning. The returned arrays only
    contain successfully loaded clips, so len(audios) may be less than the
    number of rows in the metadata for the requested folds.

    Args:
        data_dir: Path to UrbanSound8K root directory.
        folds: List of fold numbers (1-10) to load.
        sr: Target sample rate.
        duration: Target duration in seconds.

    Returns:
        audios: List of numpy arrays (raw audio waveforms).
        labels: numpy array of class IDs.
        metadata: DataFrame with file info for loaded clips.
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

    return audios, np.array(labels), df_folds.loc[valid_indices].reset_index(drop=True)


def get_default_split():
    """Return default train/val/test fold split.

    For prototype: folds 1-8 for training, 9 for validation, 10 for test.
    For full evaluation, use 10-fold CV with all fold combinations.

    Returns:
        dict with keys 'train', 'val', 'test', each mapping to a list of fold numbers.
    """
    return {
        "train": [1, 2, 3, 4, 5, 6, 7, 8],
        "val": [9],
        "test": [10],
    }
