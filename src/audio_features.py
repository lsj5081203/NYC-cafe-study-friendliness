"""
Audio feature extraction for urban sound classification.

Provides MFCC feature extraction for the baseline model (SVM/RF)
and mel-spectrogram extraction for future CNN work.
"""

import numpy as np
import librosa


def extract_mfcc(y, sr=22050, n_mfcc=40):
    """Extract MFCC features from an audio waveform.

    Computes MFCCs, delta-MFCCs, and delta-delta-MFCCs, then summarizes
    each coefficient over time using mean and standard deviation.

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
    # Delta and delta-delta
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

    return features


def extract_mfcc_batch(audios, sr=22050, n_mfcc=40):
    """Extract MFCC features for a batch of audio waveforms.

    Args:
        audios: List of numpy audio arrays.
        sr: Sample rate.
        n_mfcc: Number of MFCC coefficients.

    Returns:
        Feature matrix of shape (n_samples, n_mfcc * 6).
    """
    features = [extract_mfcc(y, sr=sr, n_mfcc=n_mfcc) for y in audios]
    return np.array(features)


def extract_mel_spectrogram(y, sr=22050, n_mels=128, hop_length=512):
    """Extract log-mel spectrogram from an audio waveform.

    Args:
        y: Audio waveform as numpy array.
        sr: Sample rate.
        n_mels: Number of mel bands.
        hop_length: Hop length for STFT.

    Returns:
        Log-mel spectrogram of shape (n_mels, time_frames).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=hop_length
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec
