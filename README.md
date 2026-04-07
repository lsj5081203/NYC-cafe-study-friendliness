# Silence in the City: NYC Cafe Study-Friendliness from Sound

**ML Class Project — Jun Lee**

Using machine learning to estimate how "study-friendly" a cafe environment is in New York City, based on the quality of sound — not just loudness.

## Overview

This project trains an urban sound event classifier on [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html), then applies it to field recordings from NYC cafes to detect distractive sound events (sirens, drilling, jackhammers, etc.). Audio predictions are combined with spatial context from [NYC Open Data](https://opendata.cityofnewyork.us/) (Wi-Fi hotspot density, nearby business counts) to produce a study-friendliness score for each location.

**Deliverables:**
- A map of NYC with predicted study-friendliness scores for recorded cafe locations
- Acoustic outlier analysis: locations that are unexpectedly quiet or noisy

## Current Status

> **Pipeline code is complete. Awaiting dataset download and first training run.**
>
> - [x] Source code implemented (`src/`)
> - [x] Notebooks ready (`notebooks/`)
> - [x] 7 NYC cafe field recordings collected (`data/cafe_recordings/`)
> - [ ] UrbanSound8K downloaded → `data/UrbanSound8K/`
> - [ ] Baseline model trained and evaluated
> - [ ] Cafe recordings scored and map generated

## Models

| Model | Features | Status |
|-------|----------|--------|
| **Baseline**: SVM / Random Forest | MFCC (40 coefficients + deltas) | Implemented (not yet trained) |
| **Proposed**: CNN | Mel-spectrograms + spatial context | Planned |
This project trains an urban sound event classifier on [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html), then applies it to field recordings from 7 NYC cafes to detect distractive sound events (sirens, drilling, jackhammers, etc.). Audio predictions are combined with spatial context from [NYC Open Data](https://opendata.cityofnewyork.us/) (Wi-Fi hotspot density within 200m, nearby eatery count within 200m) to produce a 0–100 study-friendliness score for each location.

## What's Implemented vs. Planned

| Component | Status | Details |
|-----------|--------|---------|
| MFCC feature extraction (40 coeff. + deltas + delta-deltas = 240-dim) | Implemented | Parallelized with joblib across all CPUs |
| SVM baseline (RBF kernel, C=10, StandardScaler) | Implemented | `build_svm_pipeline()` in `src/baseline_model.py` |
| RF baseline (200 trees, no scaler — scale-invariant) | Implemented | `build_rf_pipeline()` in `src/baseline_model.py` |
| 10-fold cross-validation | Implemented | Uses UrbanSound8K's predefined folds; fold 9 = val, fold 10 = test |
| Spatial features via NYC Open Data SODA API | Implemented | Wi-Fi hotspot count and eatery count within 200m haversine radius |
| Study-friendliness scoring | Implemented | Acoustic 90% weight; Wi-Fi +10% bonus; eatery −5% penalty |
| End-to-end inference script | Implemented | `scripts/run_inference.py` — cafe audio to scored CSV |
| CNN on mel-spectrograms | Planned | `extract_mel_spectrogram` exists in `audio_features.py`; model not yet built |
| CNN + spatial context fusion | Planned | Pending CNN implementation |
| Interactive NYC map visualization | Planned | folium is installed; visualization code not yet written |

## Quick Start

### 1. Set up environment

```bash
conda env create -f env.yaml
conda activate cafe-study
```

This installs Python 3.10, librosa, scikit-learn, torch, joblib, folium, and ffmpeg. ffmpeg is required to decode the .m4a cafe recordings during inference.

### 2. Download UrbanSound8K

The dataset is licensed CC BY-NC 4.0 and is available through two routes:

**Option A: Official site (requires form)**
1. Go to https://urbansounddataset.weebly.com/urbansound8k.html
2. Fill out the short access form and agree to the terms
3. Download the archive and extract to `data/UrbanSound8K/`

**Option B: Zenodo mirror (no form required)**
1. Go to https://zenodo.org/records/1203745
2. Download `UrbanSound8K.tar.gz`
3. Extract to `data/UrbanSound8K/`

Expected layout after extraction:

```
data/UrbanSound8K/
├── audio/
│   ├── fold1/
│   ├── fold2/
│   ...
│   └── fold10/
└── metadata/
    └── UrbanSound8K.csv
```

See `data/README.md` for details on all other datasets (SONYC-UST-V2, NYC Open Data, and self-collected cafe recordings).

### 3. Explore the data

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 4. Train baseline model

```bash
jupyter notebook notebooks/02_baseline_model.ipynb
```

Or directly from Python (~3 minutes for RF training on CPU):

```python
from src.baseline_model import run_single_split, save_model

# Folds 1-8 train, fold 9 val, fold 10 test
model, results = run_single_split("data/UrbanSound8K", model_type="rf")
save_model(model, "data/models/rf_model.pkl")
```

For full 10-fold cross-validation:

```python
from src.baseline_model import run_kfold_cv

fold_accs, mean_acc = run_kfold_cv("data/UrbanSound8K", model_type="rf")
```

To use SVM instead, pass `model_type="svm"` to either function.

### 5. Score cafe recordings (end-to-end inference)

After training and saving a model, run inference on the self-collected cafe recordings:

```bash
python scripts/run_inference.py \
    --model data/models/rf_model.pkl \
    --recordings data/cafe_recordings \
    --metadata data/cafe_recordings/cafe_metadata.csv \
    --output data/results/cafe_scores.csv
```

The script windows each recording into 4-second segments (50% overlap), classifies each window with the trained model, fetches Wi-Fi and eatery density from the NYC Open Data API (cached after first run), and combines the results into a score.

Outputs:
- `data/results/cafe_scores.csv` — per-recording detail (acoustic score, wifi_count, eatery_count, final score, label)
- `data/results/cafe_scores_summary.csv` — one row per cafe, averaged across inside/outside recordings

Full argument reference:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | Path to saved `.pkl` model file |
| `--recordings` | `data/cafe_recordings` | Directory containing cafe audio files |
| `--metadata` | `data/cafe_recordings/cafe_metadata.csv` | CSV with name, latitude, longitude, filename, recording_type |
| `--output` | `data/results/cafe_scores.csv` | Output CSV path |
| `--wifi-cache` | `data/cache/wifi.csv` | Cache path for Wi-Fi hotspot data |
| `--eatery-cache` | `data/cache/eateries.csv` | Cache path for eatery data |

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: URBAN SOUND CLASSIFICATION (Train on UrbanSound8K)  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  UrbanSound8K .wav          Fold 1-8              RF/SVM    │
│  (8732 clips)    ──►  MFCC extraction   ──►  Training  ───► │
│                       (240-dim features)        ~3min       │
│                                                              │
│  Fold 9 ────► Validation (hyperparameter tuning)            │
│  Fold 10 ───► Test evaluation (report once)                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ (trained model)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: CAFE STUDY-FRIENDLINESS SCORING                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Cafe .m4a          4s windows    MFCC      Predictions    │
│  recordings  ────► (50% overlap) ──► extract ───► RF/SVM    │
│                                                              │
│     ┌──────────────────────────────────────────┐            │
│     │ Combine Predictions into Acoustic Score  │            │
│     │ (lower distraction = higher score)       │            │
│     └──────────────────────────────────────────┘            │
│                           │                                 │
│     ┌─────────────────────▼────────────────────┐            │
│     │ Add Spatial Context (NYC Open Data)      │            │
│     │ · Wi-Fi count / 10, capped at 1.0       │            │
│     │   → +10% bonus                          │            │
│     │ · Eatery count / 50, capped at 1.0      │            │
│     │   → -5% penalty                         │            │
│     └──────────────────────┬────────────────────┘            │
│                            │                                │
│                            ▼                                │
│                  Final Score (0-100)                        │
│            Excellent / Good / Fair / Poor / Avoid           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
├── README.md                         # This file
├── env.yaml                          # Conda environment
├── .gitignore
├── docs/
│   ├── WORKFLOW.md                   # End-to-end pipeline walkthrough
│   ├── FAQ.md                        # Common questions & troubleshooting
│   ├── PROJECT_TEMPLATE.md           # Project scope and plan
│   └── blog.md                       # Technical blog post
├── data/
│   └── README.md                     # Dataset download & format
├── notebooks/
│   ├── 01_data_exploration.ipynb     # UrbanSound8K exploration
│   └── 02_baseline_model.ipynb       # MFCC + SVM/RF baseline
├── src/
│   ├── audio_features.py             # MFCC & mel-spectrogram extraction
│   ├── dataset.py                    # UrbanSound8K data loading
│   ├── baseline_model.py             # SVM/RF training & evaluation
│   ├── spatial_features.py           # NYC Open Data + geographic joins
│   └── scoring.py                    # Study-friendliness scoring
├── tests/
│   ├── test_scoring.py               # Scoring unit tests
│   └── test_audio_features.py        # Audio feature extraction tests
├── scripts/
│   ├── run_inference.py              # End-to-end inference pipeline
│   └── download_data.sh              # Data download helper
└── assets/                           # Figures for report
```

## Datasets

| Dataset | Source | Purpose | License |
|---------|--------|---------|---------|
| UrbanSound8K | [Official form](https://urbansounddataset.weebly.com/urbansound8k.html) or [Zenodo](https://zenodo.org/records/1203745) | Training sound classifier | CC BY-NC 4.0 |
| SONYC-UST-V2 | [Zenodo](https://zenodo.org/records/3966543) | NYC urban sound recordings with geolocation | CC BY 4.0 |
| NYC Wi-Fi Hotspots | [NYC Open Data](https://data.cityofnewyork.us/City-Government/NYC-Wi-Fi-Hotspot-Locations/yjub-udmw) | Spatial context — fetched automatically via SODA API | Public domain |
| Directory of Eateries | [NYC Open Data](https://data.cityofnewyork.us/Recreation/Directory-of-Eateries/8792-ebcp) | Spatial context — fetched automatically via SODA API | Public domain |
| Cafe field recordings | Self-collected (7 NYC cafes, 14 recordings) | Inference targets | N/A |

## Data Leakage Note

UrbanSound8K clips from the same source recording are deliberately distributed across folds by the dataset creators to prevent test-set contamination. `src/dataset.py` always uses these predefined fold assignments — never random splits.

- `run_single_split()`: folds 1–8 train, **fold 9 = validation** (model selection only), **fold 10 = held-out test** (reported once at the end)
- `run_kfold_cv()`: proper 10-fold CV, rotating each fold as test across all 10 iterations

Skipping this protocol and using random splits would cause clips from the same source recording to appear in both train and test, inflating accuracy.

## Running Tests

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_scoring.py

# Run a specific test class or function
pytest tests/test_scoring.py::TestComputeAcousticScore::test_mixed_classes
```

## References

1. Salamon, J., Jacoby, C., & Bello, J. P. (2014). A Dataset and Taxonomy for Urban Sound Research. *ACM MM 2014*.
2. Piczak, K. J. (2015). Environmental Sound Classification with Convolutional Neural Networks. *IEEE MLSP*.
3. Salamon, J., & Bello, J. P. (2017). Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification. *IEEE SPL*.
4. Cartwright, M., et al. (2020). SONYC Urban Sound Tagging (SONYC-UST): A Multilabel Dataset from an Urban Acoustic Sensor Network. *DCASE 2020*.
