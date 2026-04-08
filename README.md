# Silence in the City: NYC Cafe Study-Friendliness from Sound

**ML Class Project — Jun Lee**

Using machine learning to estimate how "study-friendly" a cafe environment is in New York City, based on the quality of sound — not just loudness.

## Overview

This project trains an urban sound event classifier on [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html), then applies it to field recordings from NYC cafes to detect distractive sound events (sirens, drilling, jackhammers, etc.). Audio predictions are combined with spatial context from [NYC Open Data](https://opendata.cityofnewyork.us/) (Wi-Fi hotspot density, nearby business counts) to produce a study-friendliness score for each location.

**Deliverables:**
- A map of NYC with predicted study-friendliness scores for recorded cafe locations
- Acoustic outlier analysis: locations that are unexpectedly quiet or noisy

## Current Status

> **All models trained and evaluated. Cafe recordings scored with results published.**
>
> - [x] Source code implemented (`src/`)
> - [x] Notebooks ready (`notebooks/`)
> - [x] 7 NYC cafe field recordings collected (`data/cafe_recordings/`)
> - [x] UrbanSound8K downloaded and baseline models trained (RF 72.64%, SVM 73.00%)
> - [x] CNN trained on Colab A100 GPU (83.39% test accuracy)
> - [x] Cafe recordings scored and results published

## Models

| Model | Test Accuracy | Features | Status |
|-------|---------------|----------|--------|
| **Baseline**: Random Forest | 72.64% | MFCC (40 coefficients + deltas) | Trained |
| **Baseline**: SVM | 73.00% | MFCC (40 coefficients + deltas) | Trained |
| **CNN**: UrbanSoundCNN | 83.39% | Mel-spectrograms | Trained |

## Results

All 7 NYC cafes were scored as "Fair" (40.56–47.74 points) based on acoustic and spatial context features:

| Cafe | Avg Final Score | Label |
|------|----------------|-------|
| Jacx & Co Food Hall | 47.74 | Fair |
| Blank Street Cafe | 47.14 | Fair |
| Rosecrans Cafe | 44.95 | Fair |
| Joe Coffee | 42.60 | Fair |
| Paris Baguette | 42.42 | Fair |
| Starbucks | 42.42 | Fair |
| Utopia Bagel | 40.56 | Fair |

The CNN model (83.39% accuracy) significantly outperforms the baseline classifiers and is recommended for production use. See `docs/blog.md` for detailed analysis and per-class performance.

## Quick Start

### 1. Set up environment

```bash
conda env create -f env.yaml
conda activate cafe-study
```

### 2. Download data

Download UrbanSound8K from https://urbansounddataset.weebly.com/urbansound8k.html and extract to `data/UrbanSound8K/`. See `data/README.md` for details.

### 3. Explore the data

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 4. View training results

```bash
# Baseline models (RF 72.64%, SVM 73.00%)
jupyter notebook notebooks/02_baseline_model.ipynb

# CNN model (83.39%)
jupyter notebook notebooks/03_cnn_model.ipynb

# Cafe scoring results
jupyter notebook notebooks/04_cafe_scoring.ipynb
```

## Project Structure

```
├── README.md                         # This file
├── env.yaml                          # Conda environment
├── .gitignore
├── docs/
│   ├── PROJECT_TEMPLATE.md           # Project scope and plan
│   └── blog.md                       # Technical blog post
├── data/
│   └── README.md                     # Dataset download instructions
├── notebooks/
│   ├── 01_data_exploration.ipynb     # UrbanSound8K exploration
│   ├── 02_baseline_model.ipynb       # MFCC + SVM/RF baseline (72.64% / 73.00%)
│   ├── 03_cnn_model.ipynb            # CNN training and evaluation (83.39%)
│   └── 04_cafe_scoring.ipynb         # Apply CNN to 7 NYC cafes, generate scores
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
│   └── download_data.sh              # Data download helper
└── assets/                           # Figures for report
```

## Datasets

| Dataset | Source | Purpose |
|---------|--------|----------|
| UrbanSound8K | [Link](https://urbansounddataset.weebly.com/urbansound8k.html) | Training sound classifier |
| SONYC-UST-V2 | [Zenodo](https://zenodo.org/records/3966543) | NYC urban sound recordings with geolocation |
| NYC Wi-Fi Hotspots | [NYC Open Data](https://data.cityofnewyork.us/City-Government/NYC-Wi-Fi-Hotspot-Locations/yjub-udmw) | Spatial context feature |
| Directory of Eateries | [NYC Open Data](https://data.cityofnewyork.us/Recreation/Directory-of-Eateries/8792-ebcp) | Spatial context feature |
| Cafe recordings | Self-collected | Inference targets |

## References

1. Salamon, J., Jacoby, C., & Bello, J. P. (2014). A Dataset and Taxonomy for Urban Sound Research. *ACM MM 2014*.
2. Piczak, K. J. (2015). Environmental Sound Classification with Convolutional Neural Networks. *IEEE MLSP*.
3. Salamon, J., & Bello, J. P. (2017). Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification. *IEEE SPL*.
