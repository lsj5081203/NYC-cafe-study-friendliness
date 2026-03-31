"""Unit tests for src/audio_features.py using synthetic audio signals."""

import numpy as np
import pytest
from src.audio_features import extract_mfcc, extract_mfcc_batch, extract_mel_spectrogram


SR = 22050


def make_sine(freq=440, sr=SR, duration=1.0):
    """Generate a sine wave for testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


# --- extract_mfcc ---

class TestExtractMfcc:
    def test_output_shape_default(self):
        """n_mfcc=40 → 240-dim feature vector (40 * 6 stats)."""
        y = make_sine()
        feats = extract_mfcc(y, sr=SR, n_mfcc=40)
        assert feats.shape == (240,)

    def test_output_shape_13(self):
        """n_mfcc=13 → 78-dim feature vector."""
        y = make_sine()
        feats = extract_mfcc(y, sr=SR, n_mfcc=13)
        assert feats.shape == (78,)

    def test_output_is_finite(self):
        """All values should be finite (no NaN or Inf)."""
        y = make_sine()
        feats = extract_mfcc(y, sr=SR, n_mfcc=40)
        assert np.all(np.isfinite(feats))

    def test_deterministic(self):
        """Same input → same output."""
        y = make_sine()
        feats1 = extract_mfcc(y, sr=SR, n_mfcc=40)
        feats2 = extract_mfcc(y, sr=SR, n_mfcc=40)
        np.testing.assert_array_equal(feats1, feats2)

    def test_different_signals_differ(self):
        """Different frequencies should produce different features."""
        y_low = make_sine(freq=200)
        y_high = make_sine(freq=4000)
        feats_low = extract_mfcc(y_low, sr=SR, n_mfcc=40)
        feats_high = extract_mfcc(y_high, sr=SR, n_mfcc=40)
        assert not np.allclose(feats_low, feats_high)


# --- extract_mfcc_batch ---

class TestExtractMfccBatch:
    def test_output_shape(self):
        """3 signals → shape (3, 240)."""
        audios = [make_sine(f) for f in [200, 440, 1000]]
        feats = extract_mfcc_batch(audios, sr=SR, n_mfcc=40)
        assert feats.shape == (3, 240)

    def test_consistent_with_single(self):
        """Batch result[0] should match individual extraction."""
        audios = [make_sine(440), make_sine(1000)]
        batch = extract_mfcc_batch(audios, sr=SR, n_mfcc=40)
        single = extract_mfcc(audios[0], sr=SR, n_mfcc=40)
        np.testing.assert_array_almost_equal(batch[0], single)


# --- extract_mel_spectrogram ---

class TestExtractMelSpectrogram:
    def test_output_shape_n_mels(self):
        """First dimension should equal n_mels."""
        y = make_sine()
        spec = extract_mel_spectrogram(y, sr=SR, n_mels=128)
        assert spec.shape[0] == 128

    def test_output_values_db(self):
        """Log-mel spectrogram values should be <= 0 dB (ref=np.max)."""
        y = make_sine()
        spec = extract_mel_spectrogram(y, sr=SR)
        assert np.all(spec <= 0.0)

    def test_different_n_mels(self):
        """n_mels=64 → first dimension is 64."""
        y = make_sine()
        spec = extract_mel_spectrogram(y, sr=SR, n_mels=64)
        assert spec.shape[0] == 64
