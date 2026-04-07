"""\nUrbanSound8K dataset loader.\n\nLoads audio clips from the UrbanSound8K dataset, respecting the predefined\n10-fold cross-validation splits to avoid data leakage.\n\nReference: J. Salamon, C. Jacoby, and J. P. Bello,\n\"A Dataset and Taxonomy for Urban Sound Research,\" ACM MM 2014.\n"""\n\nimport os\nimport warnings\nimport numpy as np\nimport pandas as pd\nimport librosa\n\n\n# UrbanSound8K class labels (classID -> class name)\nCLASS_NAMES = {\n    0: "air_conditioner",\n    1: "car_horn",\n    2: "children_playing",\n    3: "dog_bark",\n    4: "drilling",\n    5: "engine_idling",\n    6: "gun_shot",\n    7: "jackhammer",\n    8: "siren",\n    9: "street_music",\n}\n\nNUM_CLASSES = 10\n\n\ndef load_metadata(data_dir):\n    """Load UrbanSound8K metadata CSV.\n\n    Args:\n        data_dir: Path to UrbanSound8K root directory\n            (containing metadata/ and audio/ subdirs).\n\n    Returns:\n        DataFrame with columns: slice_file_name, fsID, start, end,\n        salience, fold, classID, class.\n    """\n    csv_path = os.path.join(data_dir, "metadata", "UrbanSound8K.csv")\n    return pd.read_csv(csv_path)\n\n\ndef load_audio(file_path, sr=22050, duration=4.0):\n    """Load an audio file, resampled and padded/truncated to fixed length.\n\n    Args:\n        file_path: Path to audio file.\n        sr: Target sample rate.\n        duration: Target duration in seconds.\n\n    Returns:\n        numpy array of shape (n_samples,) where n_samples = sr * duration.\n    """\n    n_samples = int(sr * duration)\n    y, _ = librosa.load(file_path, sr=sr, duration=duration)\n\n    # Pad if shorter than target duration\n    if len(y) < n_samples:\n        y = np.pad(y, (0, n_samples - len(y)), mode="constant")\n    # Truncate if longer\n    else:\n        y = y[:n_samples]\n\n    return y\n\n\ndef get_fold_data(data_dir, folds, sr=22050, duration=4.0):\n    """Load all audio clips from specified folds.\n\n    Args:\n        data_dir: Path to UrbanSound8K root directory.\n        folds: List of fold numbers (1-10) to load.\n        sr: Target sample rate.\n        duration: Target duration in seconds.\n\n    Returns:\n        audios: List of numpy arrays (raw audio waveforms).\n        labels: numpy array of class IDs.\n        metadata: DataFrame with file info for loaded clips.\n    """\n    df = load_metadata(data_dir)\n    df_folds = df[df["fold"].isin(folds)].reset_index(drop=True)\n\n    audios = []\n    labels = []\n    valid_indices = []\n\n    for idx, row in df_folds.iterrows():\n        file_path = os.path.join(\n            data_dir, "audio", f"fold{row['fold']}", row["slice_file_name"]\n        )\n        try:\n            y = load_audio(file_path, sr=sr, duration=duration)\n            audios.append(y)\n            labels.append(row["classID"])\n            valid_indices.append(idx)\n        except Exception as e:\n            print(f"Warning: Could not load {file_path}: {e}")\n\n    n_expected = len(df_folds)\n    n_loaded = len(audios)\n    if n_loaded < n_expected:\n        n_failed = n_expected - n_loaded\n        pct_failed = 100 * n_failed / n_expected\n        warnings.warn(\n            f"{n_failed}/{n_expected} clips failed to load ({pct_failed:.1f}%). "\n            f"Training will proceed with {n_loaded} clips."\n        )\n\n    return audios, np.array(labels), df_folds.loc[valid_indices].reset_index(drop=True)\n\n\ndef get_default_split():\n    """Return default train/val/test fold split.\n\n    For prototype: folds 1-8 for training, 9 for validation, 10 for test.\n    For full evaluation, use 10-fold CV with all fold combinations.\n\n    Returns:\n        dict with keys 'train', 'val', 'test', each mapping to a list of fold numbers.\n    """\n    return {\n        "train": [1, 2, 3, 4, 5, 6, 7, 8],\n        "val": [9],\n        "test": [10],\n    }\n
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
