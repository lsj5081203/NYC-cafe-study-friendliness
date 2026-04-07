"""Unit tests for src/audio_features.py — uses synthetic sine waves, no dataset required."""

import numpy as np
import pytest
from src.audio_features import extract_mfcc, extract_mfcc_batch, extract_mel_spectrogram


SR = 22050
DURATION = 1.0


def make_sine(freq, sr=SR, duration=DURATION):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


class TestExtractMfcc:
    def test_output_shape_default(self):
        y = make_sine(440)
        features = extract_mfcc(y, sr=SR, n_mfcc=40)
        assert features.shape == (240,)  # 40 * 6

    def test_output_shape_13(self):
        y = make_sine(440)
        features = extract_mfcc(y, sr=SR, n_mfcc=13)
        assert features.shape == (78,)  # 13 * 6

    def test_output_is_finite(self):
        y = make_sine(440)
        features = extract_mfcc(y, sr=SR)
        assert np.all(np.isfinite(features))

    def test_deterministic(self):
        y = make_sine(440)
        f1 = extract_mfcc(y, sr=SR)
        f2 = extract_mfcc(y, sr=SR)
        np.testing.assert_array_equal(f1, f2)

    def test_different_signals_differ(self):
        y_low = make_sine(200)
        y_high = make_sine(4000)
        f_low = extract_mfcc(y_low, sr=SR)
        f_high = extract_mfcc(y_high, sr=SR)
        assert not np.allclose(f_low, f_high)


class TestExtractMfccBatch:
    def test_output_shape(self):
        audios = [make_sine(f) for f in [200, 440, 1000]]
        X = extract_mfcc_batch(audios, sr=SR, n_mfcc=40)
        assert X.shape == (3, 240)

    def test_consistent_with_single(self):
        audios = [make_sine(f) for f in [200, 440, 1000]]
        X = extract_mfcc_batch(audios, sr=SR, n_mfcc=40)
        single = extract_mfcc(audios[0], sr=SR, n_mfcc=40)
        np.testing.assert_array_equal(X[0], single)


class TestExtractMelSpectrogram:
    def test_output_shape_n_mels(self):
        y = make_sine(440)
        mel = extract_mel_spectrogram(y, sr=SR, n_mels=128)
        assert mel.shape[0] == 128

    def test_output_values_db(self):
        # log-mel with ref=max → all values ≤ 0 dB
        y = make_sine(440)
        mel = extract_mel_spectrogram(y, sr=SR)
        assert np.all(mel <= 0.0)

    def test_different_n_mels(self):
        y = make_sine(440)
        mel = extract_mel_spectrogram(y, sr=SR, n_mels=64)
        assert mel.shape[0] == 64
