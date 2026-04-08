# End-to-End Workflow: NYC Cafe Study-Friendliness Pipeline

This document walks through the complete workflow from raw audio data to study-friendliness scores.

## Workflow Overview

```
DATA ACQUISITION
    ↓
AUDIO CLASSIFICATION (UrbanSound8K)
    ↓
CAFE INFERENCE
    ↓
SPATIAL ENRICHMENT
    ↓
SCORING & RESULTS
```

## Stage 1: Data Acquisition

### 1.1 Download UrbanSound8K (Training Data)

Required for training the baseline sound classifier. Two options:

**Option A: Official source (requires form submission)**
```bash
# Visit: https://urbansounddataset.weebly.com/urbansound8k.html
# Fill out download form, accept terms
# Extract to:
mkdir -p data/UrbanSound8K
unzip UrbanSound8K.zip -d data/
```

**Option B: Zenodo mirror (no form)**
```bash
# Visit: https://zenodo.org/records/1203745
# Download directly
unzip UrbanSound8K.zip -d data/
```

**Verify structure:**
```
data/UrbanSound8K/
├── audio/
│   ├── fold1/
│   ├── fold2/
│   └── ... fold10/
└── metadata/
    └── UrbanSound8K.csv
```

### 1.2 Prepare Cafe Recordings

Place your cafe recordings in `data/cafe_recordings/`. You'll need:
- Audio files in .m4a format (or any format ffmpeg can decode)
- A metadata CSV describing each recording

**Example metadata structure** (`cafe_metadata.csv`):
```csv
name,latitude,longitude,filename,recording_type,notes
Blank Street Cafe,40.735269,-73.998427,Blank street cafe inside.m4a,inside,Greenwich Village location
Blank Street Cafe,40.735269,-73.998427,Blank street cafe outside.m4a,outside,Street level sounds
...
```

See `data/README.md` for the full cafe list and recording details.

### 1.3 Optional: Download SONYC-UST-V2 (Complementary NYC Data)

For analysis of real NYC urban sound patterns:

```bash
# Visit: https://zenodo.org/records/3966543
# Download and extract to:
mkdir -p data/SONYC-UST-V2
unzip SONYC_UST_V2.zip -d data/
```

Not required for the main pipeline, but useful for comparing classifier performance on real NYC audio.

## Stage 2: Train Audio Classifier (UrbanSound8K)

### 2.1 Explore the Data

Start by visualizing UrbanSound8K's structure and class distribution:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

This notebook:
- Loads metadata and plots fold/class distributions
- Plays sample audio clips from each class
- Visualizes MFCC features
- Shows the 10-fold structure (no leakage between folds)

### 2.2 Train Baseline Model

Train MFCC + Random Forest (recommended) or MFCC + SVM:

```bash
jupyter notebook notebooks/02_baseline_model.ipynb
```

Or directly in Python:

```python
from src.baseline_model import run_single_split, run_kfold_cv

# Quick prototype: folds 1-8 train, fold 9 val, fold 10 test
model, results = run_single_split("data/UrbanSound8K", model_type="rf")
# Returns trained model and test-fold evaluation metrics
# Runtime: ~3 minutes on CPU

# Full evaluation: proper 10-fold cross-validation
fold_accs, mean_acc = run_kfold_cv("data/UrbanSound8K", model_type="rf")
# Returns per-fold accuracies and mean
# Runtime: ~30 minutes on CPU (3 min × 10 folds)
```

**Actual results (10-fold CV):**
- Random Forest: 72.64% accuracy
- SVM with RBF kernel: 73.00% accuracy

Both significantly outperform the literature baselines. RF is faster and scale-invariant.

### 2.3 Save the Trained Model

After training, save your model for inference:

```python
from src.baseline_model import save_model
save_model(model, "data/models/rf_model.pkl")
```

The saved model is a sklearn Pipeline including the feature extraction parameters.

## Stage 2.5: Train CNN Model (Optional, requires GPU)

For higher accuracy (83.39% vs 73.00% SVM), train a CNN on mel-spectrograms:

```bash
jupyter notebook notebooks/03_cnn_model.ipynb
```

This notebook:
- Prepares mel-spectrogram data (UrbanSound8K)
- Defines UrbanSoundCNN architecture (4 conv blocks, 390K params)
- Trains with Adam + cosine annealing LR schedule
- Evaluates 10-fold CV performance (best: 80.51% val accuracy at epoch 24)
- Saves trained model to `data/models/cnn_model.pth`

Runtime on Colab A100 GPU: ~97 seconds. Consumer GPUs (RTX 3060): 5-15 minutes.

The CNN is now recommended over baseline models for production use.

## Stage 3: Run Inference on Cafe Recordings

### 3.1 Prerequisites

Ensure you have:
1. A trained model file (e.g., `data/models/rf_model.pkl`)
2. Cafe audio files in `data/cafe_recordings/`
3. A metadata CSV with filenames and coordinates
4. `ffmpeg` installed (included in `env.yaml`)

### 3.2 Run Inference Script

```bash
python scripts/run_inference.py \
    --model data/models/rf_model.pkl \
    --recordings data/cafe_recordings \
    --metadata data/cafe_recordings/cafe_metadata.csv \
    --output data/results/cafe_scores.csv
```

**What this does:**
1. **Load model**: Restores the trained RF pipeline from disk
2. **Load metadata**: Reads cafe names, locations, and audio filenames
3. **Fetch spatial features**: Downloads Wi-Fi hotspot and eatery data from NYC Open Data API
   - Results are cached to `data/cache/wifi.csv` and `data/cache/eateries.csv` (can be reused)
4. **Per-recording inference**:
   - Loads each .m4a file
   - Splits into 4-second windows with 50% overlap
   - Extracts MFCC features for each window
   - Predicts sound class for each window
   - Aggregates predictions into an acoustic score
5. **Combine with spatial context**: Wi-Fi density (+), eatery density (-)
6. **Save results**: Two CSV files

**Output files:**

`cafe_scores.csv` — per-recording details
```csv
name,filename,recording_type,n_windows,acoustic_score,wifi_count,eatery_count,final_score,label
Blank Street Cafe,Blank street cafe inside.m4a,inside,7,62.45,8,28,60.25,Good
Blank Street Cafe,Blank street cafe outside.m4a,outside,7,48.30,8,28,45.97,Fair
...
```

`cafe_scores_summary.csv` — aggregated by cafe
```csv
name,avg_acoustic_score,avg_final_score,wifi_count,eatery_count,recordings,label
Blank Street Cafe,55.38,53.11,8,28,2,Fair
...
```

### 3.3 Parallel Processing (Optional)

For faster inference on large recording collections, the MFCC extraction is already parallelized. If you have many cores available, it should automatically scale. If you encounter memory issues, reduce `n_jobs` in `extract_mfcc_batch()` or process recordings in batches.

## Stage 4: Spatial Enrichment (Optional)

The inference script handles spatial enrichment automatically via `src/spatial_features.py`. You can also fetch spatial data independently:

```python
from src.spatial_features import (
    download_wifi_hotspots,
    download_eateries,
    build_spatial_features
)
import pandas as pd

# Load cafe metadata
cafe_locs = pd.read_csv("data/cafe_recordings/cafe_metadata.csv")
cafe_locs = cafe_locs[["name", "latitude", "longitude"]].drop_duplicates("name")

# Fetch spatial data (or use cached CSVs)
wifi_df = download_wifi_hotspots(cache_path="data/cache/wifi.csv")
eateries_df = download_eateries(cache_path="data/cache/eateries.csv")

# Compute densities within 200m radius
spatial_df = build_spatial_features(cafe_locs, wifi_df, eateries_df, radius_m=200)
print(spatial_df)
```

This is useful if you want to analyze spatial features independently or re-run scoring with different weights.

## Stage 5: Analyze Results

### 5.1 Load and Explore Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load summary scores
results = pd.read_csv("data/results/cafe_scores_summary.csv")

# View top study-friendly cafes
print(results.sort_values("avg_final_score", ascending=False)[["name", "avg_final_score", "label"]])

# Plot score distribution
plt.hist(results["avg_final_score"], bins=10, edgecolor="black")
plt.xlabel("Study-Friendliness Score")
plt.ylabel("Cafe Count")
plt.show()
```

### 5.2 Acoustic vs. Spatial Analysis

```python
# Which cafes have high acoustic scores but low final scores?
# (i.e., good sound but poor spatial context, like high foot traffic)

results["acoustic_spatial_gap"] = results["avg_acoustic_score"] - results["avg_final_score"]
outliers = results.sort_values("acoustic_spatial_gap", ascending=False)
print(outliers[["name", "avg_acoustic_score", "avg_final_score", "acoustic_spatial_gap", "eatery_count"]])
```

### 5.3 Score Interpretation

| Score Range | Label | Meaning |
|-------------|-------|---------|
| 75-100 | Excellent | Very study-friendly; quiet, low foot traffic |
| 55-74 | Good | Suitable for studying; acceptable noise levels |
| 35-54 | Fair | Moderate noise; requires concentration |
| 20-34 | Poor | High distraction; difficult to focus |
| 0-19 | Avoid | Very noisy; not suitable for studying |

## Testing & Validation

### Run Unit Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_scoring.py
pytest tests/test_audio_features.py

# Specific test
pytest tests/test_scoring.py::TestComputeAcousticScore::test_mixed_classes
```

### Manual Validation

Check that:
1. Model predictions match expected class distributions
2. Acoustic scores are in the 0-100 range
3. Spatial features are reasonable (Wi-Fi/eatery counts > 0)
4. Final scores reflect both acoustic and spatial signals

## Troubleshooting

**Issue**: "ffmpeg not found on PATH"
- **Solution**: Run `conda activate cafe-study` to activate the environment, which includes ffmpeg

**Issue**: "ModuleNotFoundError: No module named 'src'"
- **Solution**: Run scripts from the project root directory: `cd /path/to/NYC-cafe-study-friendliness`

**Issue**: NYC Open Data API returns 403 Forbidden (eatery_count=0)
- **Solution**: Eatery API occasionally rejects requests. Try again later, or request an NYC Open Data app token at https://dev.socrata.com/. Current workaround: set eatery_weight=0 in scoring.py to skip this penalty.

**Issue**: API timeout downloading spatial features
- **Solution**: Use cached data if available (`--wifi-cache` and `--eatery-cache` flags); caches persist across runs

**Issue**: Audio file returns empty waveform
- **Solution**: Check file format; ffmpeg may not support it. Re-encode to MP3 or WAV and try again

**Issue**: NaN values in MFCC features
- **Solution**: This is caught and handled (replaced with 0); usually indicates silent/very-short audio. Check audio quality

## What's Implemented vs. Planned

### Implemented
- UrbanSound8K training with MFCC + RF/SVM (72.64% / 73.00% accuracy)
- 10-fold CV with proper fold management (no leakage)
- CNN on mel-spectrograms (83.39% test accuracy)
- End-to-end inference script on cafe recordings
- Spatial context fetching and density computation
- Study-friendliness scoring (acoustic + spatial)
- CLI tool for scoring batches of recordings
- 7 NYC cafes scored (all "Fair", 40.56–47.74)

### Planned
- Spatial context fusion in CNN (concatenate features to final layer)
- Interactive web map visualization (Folium or MapboxGL)
- Outlier detection (statistical flagging of unexpected results)
- Temporal analysis (how cafe scores vary by time of day)
- Fine-tuning on SONYC-UST-V2 (NYC-specific transfer learning)

## References

- **UrbanSound8K**: Salamon et al. (2014) — https://urbansounddataset.weebly.com/urbansound8k.html
- **SONYC-UST-V2**: Cartwright et al. (2020) — https://zenodo.org/records/3966543
- **NYC Open Data API**: https://opendata.cityofnewyork.us/
