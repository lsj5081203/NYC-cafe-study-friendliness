# Silence in the City: NYC Cafe Study-Friendliness from Sound

**ML Class Project — Jun Lee**

Using machine learning to estimate how "study-friendly" a cafe environment is in New York City, based on the quality of sound — not just loudness.

## Key Results

### Model Performance on UrbanSound8K

| Model | Features | Single Split (Fold 10) | 10-Fold CV |
|-------|----------|----------------------|------------|
| Random Forest | MFCC (240-dim) | 72.64% | 67.36% ± 4.49% |
| SVM (RBF, C=10) | MFCC (240-dim) | 73.00% | 68.24% ± 5.54% |
| **CNN (4-block)** | **Mel-spectrogram** | **83.39%** | — |

TL;DR: CNN beats the baselines by ~10 points on a single fold. The baselines look worse under full cross-validation because some folds are genuinely harder.

### NYC Cafe Study-Friendliness Scores

| Cafe | Avg Score | Grade | Best Recording |
|------|-----------|-------|----------------|
| Jacx & Co Food Hall | 47.74 | Fair | 50.85 (outside) |
| Blank Street Cafe | 47.14 | Fair | **55.51 (inside)** |
| Rosecrans Cafe | 44.95 | Fair | 46.23 (outside) |
| Joe Coffee | 42.60 | Fair | 41.33 (inside) |
| Paris Baguette | 42.42 | Fair | 47.97 (inside) |
| Starbucks | 42.42 | Fair | 46.36 (outside) |
| Utopia Bagel | 40.56 | Fair | 46.10 (inside) |

All cafes scored "Fair" (35-54 range). Honestly, that's not surprising — see [Limitations](#limitations) for why scores cluster.

## Overview

I trained an urban sound classifier on UrbanSound8K, then pointed it at recordings from 7 NYC cafes to figure out what kinds of sounds are actually around each one. The idea is simple: a siren is more distracting than an air conditioner, so classify what you hear, weight it by how annoying it is, and you get a study-friendliness score.

I also pulled in spatial data from NYC Open Data — nearby Wi-Fi hotspots (good for studying) and restaurant density (proxy for foot traffic) — to round out the picture.

**Pipeline**: Audio → MFCC / Mel-spectrogram → SVM / RF / CNN → Distraction score → Spatial enrichment → Final grade (0-100)

## Methodology

### Feature Extraction

Each audio clip gets turned into a 240-dimensional MFCC vector: 40 coefficients × 3 (base + delta + delta-delta) × 2 (mean + std over time). For the CNN, I used 128-bin mel-spectrograms instead — basically images of sound that the network learns to read.

### Models

| Model | Config | Parameters |
|-------|--------|------------|
| Random Forest | 200 trees, no max depth | — |
| SVM | RBF kernel, C=10, gamma='scale' | — |
| CNN | 4 conv blocks (32→64→128→256), batch norm, dropout, global avg pool | ~390K |

The CNN was trained for 30 epochs on a Colab A100 with Adam + cosine annealing. Best validation accuracy was 80.51% at epoch 24.

### Distraction Scoring

Each UrbanSound8K class gets a distraction weight — low for steady hums (air conditioner = 0.10), high for sudden disruptions (gun shot = 1.00). The acoustic score is `(1 - weighted_distraction) × 100`, then combined with spatial data:

```
final_score = 0.9 × acoustic_score
            + 0.1 × min(wifi_count / 10, 1.0) × 100
            - 0.05 × min(eatery_count / 50, 1.0) × 100
```

Scores map to labels: Excellent (75+), Good (55+), Fair (35+), Poor (20+), Avoid (<20). Full details in [docs/blog.md](docs/blog.md).

## Getting Started

### Prerequisites

- Python 3.9+
- Conda (for environment management)
- GPU recommended for CNN training (I used a Colab A100)

### Installation

```bash
conda env create -f env.yaml
conda activate cafe-study
```

### Data

Download UrbanSound8K from https://urbansounddataset.weebly.com/urbansound8k.html and extract to `data/UrbanSound8K/`. See `data/README.md` for details.

### Reproduce via Notebooks

Walk through the project in order:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb   # explore the dataset
jupyter notebook notebooks/02_baseline_model.ipynb      # train RF + SVM
jupyter notebook notebooks/03_cnn_model.ipynb           # train CNN
jupyter notebook notebooks/04_cafe_scoring.ipynb        # score the cafes
```

### Reproduce via CLI

```bash
python scripts/run_inference.py
```

### Run Tests

```bash
pytest tests/ -v
```

## Notebooks

| # | Notebook | What it does | Key result |
|---|----------|-------------|------------|
| 01 | `01_data_exploration.ipynb` | Class distribution, waveforms, spectrograms | Dataset overview |
| 02 | `02_baseline_model.ipynb` | MFCC + SVM/RF training, 10-fold CV | RF 67.36%, SVM 68.24% |
| 03 | `03_cnn_model.ipynb` | Mel-spectrogram + 4-block CNN | **83.39% test accuracy** |
| 04 | `04_cafe_scoring.ipynb` | CNN inference on 14 cafe recordings + spatial | 7 cafe scores |

## Project Structure

```
├── README.md                              # This file
├── LICENSE
├── env.yaml                               # Conda environment
├── data/
│   ├── README.md                          # Dataset download instructions
│   └── cafe_recordings/
│       └── cafe_metadata.csv              # Recording metadata (lat/lon, filenames)
├── docs/
│   ├── blog.md                            # Technical blog post with full analysis
│   ├── FAQ.md                             # Frequently asked questions
│   ├── PROJECT_TEMPLATE.md                # Project scope and plan
│   └── WORKFLOW.md                        # Development workflow notes
├── notebooks/
│   ├── 01_data_exploration.ipynb          # UrbanSound8K exploration
│   ├── 02_baseline_model.ipynb            # MFCC + SVM/RF baselines
│   ├── 03_cnn_model.ipynb                 # CNN training and evaluation
│   └── 04_cafe_scoring.ipynb              # Cafe scoring with CNN + spatial data
├── scripts/
│   ├── download_data.sh                   # Data download helper
│   ├── insert_analysis_cells.py           # Inject analysis cells into notebooks
│   └── run_inference.py                   # CLI inference pipeline
├── src/
│   ├── __init__.py
│   ├── audio_features.py                  # MFCC & mel-spectrogram extraction
│   ├── baseline_model.py                  # SVM/RF training & evaluation
│   ├── cnn_model.py                       # CNN architecture & training
│   ├── dataset.py                         # UrbanSound8K data loading
│   ├── scoring.py                         # Study-friendliness scoring
│   └── spatial_features.py                # NYC Open Data API + geographic joins
└── tests/
    ├── __init__.py
    ├── test_audio_features.py             # Audio feature extraction tests
    ├── test_scoring.py                    # Scoring unit tests
    └── test_spatial_features.py           # Spatial feature tests
```

## Limitations

1. **Domain gap** — UrbanSound8K has no cafe-specific sounds (espresso machines, cutlery, conversation). The CNN force-maps cafe audio onto intermediate classes like `street_music` and `children_playing`, which compresses all acoustic scores into a narrow 40-55 range.

2. **All cafes scored "Fair"** — The 7-point spread (40.56-47.74) is a direct symptom of the domain gap above. It doesn't mean every NYC cafe is equally study-friendly — it means the model can't distinguish them well with these training categories.

3. **Eatery API returned 403** — The NYC Open Data eatery endpoint failed during inference, so `eatery_count=0` for all cafes. This only affects the 5% eatery penalty term (~2.5 points max), but it means the spatial signal is incomplete.

4. **No CNN cross-validation** — The CNN was evaluated on a single held-out fold only. The baselines got proper 10-fold CV; the CNN number (83.39%) is a single-split result.

5. **Small sample** — 7 cafes with 2 recordings each is enough to demonstrate the pipeline, but not enough to draw statistical conclusions about NYC cafes in general.

## Datasets

| Dataset | Size | Source | Purpose |
|---------|------|--------|----------|
| UrbanSound8K | 8,732 clips | [Link](https://urbansounddataset.weebly.com/urbansound8k.html) | Training sound classifier |
| SONYC-UST-V2 | 18,510 recordings | [Zenodo](https://zenodo.org/records/3966543) | NYC urban sound recordings with geolocation |
| NYC Wi-Fi Hotspots | ~3,000 hotspots | [NYC Open Data](https://data.cityofnewyork.us/City-Government/NYC-Wi-Fi-Hotspot-Locations/yjub-udmw) | Spatial context feature |
| Directory of Eateries | — | [NYC Open Data](https://data.cityofnewyork.us/Recreation/Directory-of-Eateries/8792-ebcp) | Spatial context feature (API 403 — unused) |
| Cafe recordings | 14 files (7 cafes) | Self-collected | Inference targets |

## References

1. Salamon, J., Jacoby, C., & Bello, J. P. (2014). A Dataset and Taxonomy for Urban Sound Research. *ACM MM 2014*.
2. Piczak, K. J. (2015). Environmental Sound Classification with Convolutional Neural Networks. *IEEE MLSP*.
3. Salamon, J., & Bello, J. P. (2017). Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification. *IEEE SPL*.

## License

See [LICENSE](LICENSE) for details.
