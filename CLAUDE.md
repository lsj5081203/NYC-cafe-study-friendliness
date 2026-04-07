# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NYC Cafe Study-Friendliness from Sound — trains an urban sound classifier on UrbanSound8K, applies it to field recordings from 7 NYC cafes, and scores study-friendliness by combining acoustic predictions with spatial context (Wi-Fi density, eatery density) from NYC Open Data.

**Current status**: Pipeline code is complete. UrbanSound8K not yet downloaded, model not yet trained.

## Commands

```bash
# Setup
conda env create -f env.yaml
conda activate cafe-study

# Tests (no dataset required — uses synthetic data)
python -m pytest tests/ -v
python -m pytest tests/test_scoring.py -v          # scoring only
python -m pytest tests/test_spatial_features.py -v # spatial only

# Single test
python -m pytest tests/test_scoring.py::TestClassifyScore::test_labels -v

# Train baseline (requires UrbanSound8K)
python -c "from src.baseline_model import run_single_split; run_single_split('data/UrbanSound8K', model_type='rf')"

# Full 10-fold CV (requires UrbanSound8K)
python -c "from src.baseline_model import run_kfold_cv; run_kfold_cv('data/UrbanSound8K', model_type='svm')"

# Download NYC Open Data (Wi-Fi + eateries)
bash scripts/download_data.sh
```

## Architecture

### Data Flow
```
UrbanSound8K audio files
    → src/dataset.py (load_audio, get_fold_data)
    → src/audio_features.py (extract_mfcc → 240-dim vector)
    → src/baseline_model.py (SVM/RF via sklearn Pipeline)
    → predictions (class IDs 0-9)

predictions + cafe_metadata.csv
    → src/scoring.py (compute_acoustic_score)
    → src/spatial_features.py (wifi_count, eatery_count via haversine)
    → src/scoring.py (compute_study_friendliness → 0-100 score)
    → classify_score → Excellent/Good/Fair/Poor/Avoid
```

### Key Design Decisions

**MFCC features**: 40 coefficients × 3 (mfcc, delta, delta²) × 2 stats (mean, std) = 240-dim vector per clip. Better than the 78-dim baseline in the original Salamon et al. paper.

**10-fold CV protocol**: UrbanSound8K clips from the same source recording must never be split across train/test folds. `get_fold_data()` respects the predefined fold numbers in the metadata CSV. `run_kfold_cv()` in `baseline_model.py` implements the correct leave-one-fold-out loop.

**Scoring**: `DISTRACTION_WEIGHTS` in `scoring.py` maps class IDs 0–9 to weights (0.1 for air_conditioner → 1.0 for gun_shot). Acoustic score = `(1 - weighted_distraction) * 100`. Final score = `0.9 * acoustic + 0.1 * wifi_bonus - 0.05 * eatery_penalty`, clamped to [0, 100].

**mel-spectrogram normalization**: `ref=np.max` normalizes per-clip (max → 0dB). This discards absolute volume but improves robustness across recording devices. Use `ref=1.0` if absolute levels matter for future CNN work.

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `src/dataset.py` | UrbanSound8K loading, fold splits, `get_default_split()` |
| `src/audio_features.py` | MFCC + mel-spectrogram extraction |
| `src/baseline_model.py` | SVM/RF pipelines, `run_kfold_cv()` |
| `src/spatial_features.py` | NYC Open Data download, haversine density |
| `src/scoring.py` | Distraction weights, final score formula |

## Data

**UrbanSound8K** (6GB, not in repo): Download from https://urbansounddataset.weebly.com/urbansound8k.html → extract to `data/UrbanSound8K/`. Requires manual download due to license agreement. Expected structure: `audio/fold{1-10}/` and `metadata/UrbanSound8K.csv`.

**Cafe recordings** (not in repo): 14 `.m4a` files in `data/cafe_recordings/` (7 cafes × inside/outside). See `data/README.md` for filenames and `cafe_metadata.csv` for coordinates.

**NYC Open Data**: Auto-downloaded by `src/spatial_features.py` via SODA API on first use. Cached as CSV if `cache_path` is provided. Eatery limit is 50,000 (NYC has ~30K+ entries).

## Tests

All 43 tests run without any dataset — they use synthetic sine waves and synthetic DataFrames. Test files:
- `tests/test_audio_features.py` — MFCC shape, determinism, signal discrimination
- `tests/test_scoring.py` — Scoring functions with boundary/edge cases
- `tests/test_spatial_features.py` — Haversine distance (NYC coordinates), density counting, build_spatial_features
