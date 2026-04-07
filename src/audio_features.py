"""
Audio feature extraction for urban sound classification.

Provides MFCC feature extraction for the baseline model (SVM/RF)
and mel-spectrogram extraction for future CNN work.
"""

import numpy as np
import librosa
from joblib import Parallel, delayed


def extract_mfcc(y, sr=22050, n_mfcc=40):
    """Extract MFCC features from an audio waveform.

    Computes MFCCs, delta-MFCCs, and delta-delta-MFCCs, then summarizes
    each coefficient over time using mean and standard deviation.

    The mean+std summarization collapses the time axis into a fixed-length
    vector suitable for SVM and Random Forest classifiers, which require
    tabular input. This intentionally discards temporal order; if temporal
    dynamics matter (e.g., for a CNN/RNN), use extract_mel_spectrogram
    instead to retain the full time-frequency representation.

    Args:
        y: Audio waveform as numpy array.
        sr: Sample rate.
        n_mfcc: Number of MFCC coefficients.

    Returns:
        Feature vector of shape (n_mfcc * 6,).
        Layout: [mfcc_mean, mfcc_std, delta_mean, delta_std, delta2_mean, delta2_std]
    """
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Delta and delta-delta — require at least 9 frames (width=9 default).
    # Clips shorter than ~0.21s after padding may have degenerate deltas.
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Summarize over time: mean and std for each coefficient
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(mfcc_delta, axis=1),
        np.std(mfcc_delta, axis=1),
        np.mean(mfcc_delta2, axis=1),
        np.std(mfcc_delta2, axis=1),
    ])

    # Guard against NaN/Inf from silent or near-silent audio
    if not np.all(np.isfinite(features)):
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def extract_mfcc_batch(audios, sr=22050, n_mfcc=40, n_jobs=-1):
    """Extract MFCC features for a batch of audio waveforms.

    Args:
        audios: List of numpy audio arrays.
        sr: Sample rate.
        n_mfcc: Number of MFCC coefficients.
        n_jobs: Number of parallel jobs (-1 = all CPUs). Set to 1 to disable
            parallelism, e.g. when the RF pipeline already uses n_jobs=-1.

    Returns:
        Feature matrix of shape (n_samples, n_mfcc * 6).
    """
    features = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(extract_mfcc)(y, sr=sr, n_mfcc=n_mfcc) for y in audios
    )
    return np.array(features)


def extract_mel_spectrogram(y, sr=22050, n_mels=128, hop_length=512):
    """Extract log-mel spectrogram from an audio waveform.

    Intended for the planned CNN stage. The output retains full temporal
    structure and can be treated as a single-channel image by a 2-D
    convolutional network.

    Args:
        y: Audio waveform as numpy array.
        sr: Sample rate.
        n_mels: Number of mel filter-bank bands (frequency axis height).
            128 gives good spectral resolution for urban sounds; reduce to
            64 for lighter models.
        hop_length: Number of samples between successive STFT frames.
            512 samples at 22050 Hz ≈ 23 ms per frame. Smaller values
            increase time resolution at the cost of a wider output.

    Returns:
        Log-mel spectrogram of shape (n_mels, time_frames), where
        time_frames = ceil(len(y) / hop_length). For a 4-second clip at
        sr=22050 and hop_length=512, time_frames ≈ 173. Values are in
        dB relative to the peak energy (librosa.power_to_db with ref=max).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=hop_length
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec
