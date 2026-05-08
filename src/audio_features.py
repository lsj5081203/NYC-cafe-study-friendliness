"""
Audio feature extraction for urban sound classification.

MFCC summaries feed the sklearn baselines. Log-mel spectrograms feed the CNN.
"""

import librosa
import numpy as np
from joblib import Parallel, delayed


def extract_mfcc(y, sr=22050, n_mfcc=40):
    """Extract a fixed-length MFCC feature vector from one waveform."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(mfcc_delta, axis=1),
        np.std(mfcc_delta, axis=1),
        np.mean(mfcc_delta2, axis=1),
        np.std(mfcc_delta2, axis=1),
    ])

    if not np.all(np.isfinite(features)):
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def extract_mfcc_batch(audios, sr=22050, n_mfcc=40, n_jobs=-1):
    """Extract MFCC features for a list of waveforms."""
    features = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(extract_mfcc)(y, sr=sr, n_mfcc=n_mfcc) for y in audios
    )
    return np.array(features)


def extract_mel_spectrogram(y, sr=22050, n_mels=128, hop_length=512):
    """Extract a log-mel spectrogram from one waveform."""
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=hop_length
    )
    return librosa.power_to_db(mel_spec, ref=np.max)
