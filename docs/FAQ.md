# Frequently Asked Questions (FAQ)

## Data & Setup

### Q: How do I download UrbanSound8K?

**A:** Two options:

**Option 1: Official source (form required)**
1. Visit https://urbansounddataset.weebly.com/urbansound8k.html
2. Read the terms and fill out the download form
3. You'll receive a download link via email
4. Extract the ZIP to `data/UrbanSound8K/`
   ```bash
   unzip UrbanSound8K.zip -d data/
   ```

**Option 2: Zenodo mirror (no form)**
1. Visit https://zenodo.org/records/1203745
2. Click "Download" to get the ZIP
3. Extract to `data/UrbanSound8K/`

Both contain identical data. Option 2 is faster if you don't want to fill out a form.

---

### Q: What format are the cafe recordings in? Do I need ffmpeg?

**A:** Cafe recordings are in **.m4a format** (MPEG-4 Audio / AAC codec). 

**Yes, ffmpeg is required** to decode them. The `env.yaml` includes ffmpeg, so if you ran:
```bash
conda env create -f env.yaml
conda activate cafe-study
```

...you already have ffmpeg installed and available in your conda environment.

**If ffmpeg is not found:**
- Make sure you've activated the environment: `conda activate cafe-study`
- Or install ffmpeg separately:
  ```bash
  conda install ffmpeg
  ```

**What if I have .m4a files and ffmpeg isn't working?**
- Verify: `ffmpeg -version`
- If missing, reinstall: `conda install -c conda-forge ffmpeg`
- Or convert your files to MP3/WAV (ffmpeg can do this too):
  ```bash
  ffmpeg -i input.m4a -q:a 9 -n output.mp3
  ```

---

### Q: Can I use my own cafe recordings?

**A:** Yes! Place them in `data/cafe_recordings/` and create a metadata CSV with columns:
```csv
name,latitude,longitude,filename,recording_type,notes
My Cafe,40.7128,-74.0060,my_cafe_inside.m4a,inside,Test location
My Cafe,40.7128,-74.0060,my_cafe_outside.m4a,outside,Street sounds
```

Then run:
```bash
python scripts/run_inference.py \
    --model data/models/rf_model.pkl \
    --recordings data/cafe_recordings \
    --metadata data/cafe_recordings/cafe_metadata.csv \
    --output data/results/my_cafe_scores.csv
```

The script will:
1. Load your audio files
2. Extract acoustic features
3. Get Wi-Fi/eatery counts from NYC Open Data
4. Compute study-friendliness scores

---

### Q: What's the expected directory structure after downloading everything?

**A:**
```
NYC-cafe-study-friendliness/
├── data/
│   ├── UrbanSound8K/
│   │   ├── audio/
│   │   │   ├── fold1/
│   │   │   ├── fold2/
│   │   │   └── ... fold10/
│   │   └── metadata/
│   │       └── UrbanSound8K.csv
│   ├── SONYC-UST-V2/                    (optional)
│   │   ├── audio/
│   │   ├── annotations.csv
│   │   └── README.md
│   ├── cafe_recordings/
│   │   ├── cafe_metadata.csv
│   │   ├── Blank street cafe inside.m4a
│   │   └── ... other .m4a files
│   ├── cache/                           (auto-created during inference)
│   │   ├── wifi.csv
│   │   └── eateries.csv
│   ├── models/
│   │   └── rf_model.pkl                 (after training)
│   └── results/
│       ├── cafe_scores.csv              (after inference)
│       └── cafe_scores_summary.csv
├── notebooks/
├── src/
├── scripts/
│   └── run_inference.py
├── tests/
├── docs/
├── README.md
└── ...
```

---

## Training & Models

### Q: How long does it take to train the model?

**A:** Depends on your hardware:

| Model | Data | Time |
|-------|------|------|
| Random Forest (single split) | Folds 1-8 (5600 clips) | ~3 minutes |
| SVM (single split) | Folds 1-8 (5600 clips) | ~8 minutes |
| Random Forest (10-fold CV) | All 10 folds | ~30 minutes |

Times are approximate on CPU (Intel Core i7 / M1 Mac). GPU acceleration is not used for the baseline.

---

### Q: Which model should I use, SVM, Random Forest, or CNN?

**A:** Use **CNN** for production; baseline models are historical reference:

| Model | Test Accuracy | Speed | Status |
|-------|---------------|-------|--------|
| CNN (mel-spectrograms) | **83.39%** | ~97s on A100 GPU | **Recommended** |
| Random Forest (MFCC) | 72.64% | ~3 min | Baseline |
| SVM (MFCC) | 73.00% | ~8 min | Baseline |

The CNN achieves +10.4 percentage points better accuracy than SVM/RF. For consumer GPUs (e.g., NVIDIA RTX 3060), expect 5–15 minutes for full training.

---

### Q: What's the actual accuracy on UrbanSound8K?

**A:** Measured test accuracies using 10-fold CV:

| Model | Test Accuracy |
|-------|--------------|
| Random Forest (MFCC) | 72.64% |
| SVM (MFCC) | 73.00% |
| CNN (mel-spectrograms) | **83.39%** |

The CNN significantly outperforms the hand-crafted feature baselines. For comparison, Salamon et al. (2014) reported MFCC + GMM-SVM: 68.5%, and prior CNN work (Piczak 2015, Salamon & Bello 2017) achieved ~79% on UrbanSound8K. Our implementation achieves state-of-the-art results.

---

### Q: Why is my accuracy different from expected?

**A:** Possible causes:

1. **Feature extraction randomness**: None — MFCC extraction is deterministic
2. **Model randomness**: RF uses `random_state=42` for reproducibility; SVM also uses `random_state=42`
3. **Data loading differences**: Make sure you're using the original UrbanSound8K folds, not random splits
4. **Hardware differences**: Floating-point precision may vary slightly across machines
5. **Audio preprocessing**: Check that sample rate (SR=22050) and duration (4s) match training

If accuracy is much lower (<55%), check:
- Are you using the correct folds? (1-8 train, 9 val, 10 test)
- Is UrbanSound8K fully downloaded? (missing files?)
- Did the model train successfully? (check console output)

---

### Q: Can I use the trained model for other sound datasets?

**A:** With limitations. The model is trained specifically on UrbanSound8K's 10 classes:

```
0: air_conditioner
1: car_horn
2: children_playing
3: dog_bark
4: drilling
5: engine_idling
6: gun_shot
7: jackhammer
8: siren
9: street_music
```

For other datasets:
- **Same 10 classes**: Works as transfer learning. Fine-tune with a few hundred labeled examples.
- **Different classes**: You need to retrain from scratch on your dataset.
- **Subset of classes**: Model still predicts all 10; output will be ambiguous.

For NYC-specific sound classes, consider fine-tuning on SONYC-UST-V2.

---

## Inference & Scoring

### Q: How does the study-friendliness score work?

**A:** Three components combine into a 0-100 score:

1. **Acoustic Score (0-100)** — Based on detected sound types
   - 100 = very quiet, no distracting sounds
   - 0 = many loud, distracting sounds
   - Computed from UrbanSound8K class predictions and their subjective "distraction weights" (e.g., siren=0.85, air_conditioner=0.1)

2. **Wi-Fi Bonus (+0-10 points)**
   - Counts public Wi-Fi hotspots within 200m
   - Higher Wi-Fi density = more likely to be a study spot
   - Capped at 10 hotspots

3. **Eatery Penalty (-0-5 points)**
   - Counts food businesses within 200m
   - Higher eatery density = more foot traffic, noise
   - Capped at 50 eateries

**Formula:**
```
final_score = 0.9 * acoustic_score
            + 0.1 * (wifi_count / 10) * 100
            - 0.05 * (eatery_count / 50) * 100
```

Clamped to [0, 100].

---

### Q: What do the score labels mean?

**A:**

| Range | Label | Meaning |
|-------|-------|---------|
| 75-100 | Excellent | Very quiet, study-friendly; low foot traffic |
| 55-74 | Good | Acceptable noise; good for studying with focus |
| 35-54 | Fair | Moderate noise; requires concentration |
| 20-34 | Poor | High distraction; difficult to focus |
| 0-19 | Avoid | Very noisy; not suitable for studying |

These thresholds are arbitrary and calibrated for NYC cafe environments. Feel free to adjust in `src/scoring.py` if needed.

---

### Q: Why does a noisy cafe still have a high final score?

**A:** Possible reasons:

1. **High Wi-Fi density**: Many hotspots nearby (+10 bonus) can offset acoustic noise
2. **Low eatery density**: Fewer restaurants nearby (-5 penalty avoided)
3. **Acoustic score misclassification**: Model may have mislabeled sound types

Check the `cafe_scores.csv` output to see the breakdown:
```csv
name,acoustic_score,wifi_count,eatery_count,final_score
Cafe A,45.5,12,8,50.7  # Loud but well-connected infrastructure
```

To understand the acoustic score alone, look at column `acoustic_score` (ignoring spatial bonus/penalty).

---

### Q: How are multiple audio windows aggregated into one score?

**A:** When you score a single cafe recording (e.g., 2 minutes of audio):

1. Audio is split into **4-second windows** with **50% overlap**
2. Each window gets a sound class prediction (e.g., "siren", "air_conditioner")
3. **Acoustic score** is computed as the average distraction level across all windows
4. This single acoustic score is combined with spatial features for the final score

Example:
```
2-minute recording → 14 windows (4s each, 50% overlap)
                  → 14 class predictions
                  → 1 acoustic score
                  → 1 final score
```

For aggregating multiple recordings from the same cafe (inside + outside):
```
Cafe A inside  → acoustic_score=62, final_score=60
Cafe A outside → acoustic_score=48, final_score=46
Aggregated     → avg_acoustic_score=55, avg_final_score=53
```

(See `cafe_scores_summary.csv` for aggregated results.)

---

### Q: Can I customize the spatial weights (Wi-Fi, eatery)?

**A:** Yes. Edit `src/scoring.py` and modify the `compute_study_friendliness()` function:

```python
# Default: 90% acoustic, 10% Wi-Fi bonus, 5% eatery penalty
score = compute_study_friendliness(
    acoustic_score=62.5,
    wifi_count=8,
    eatery_count=30,
    acoustic_weight=0.9,      # Change this
    wifi_weight=0.1,          # Or this
    eatery_weight=-0.05,      # Or this
)
```

You can also change the saturation thresholds:
```python
WIFI_SATURATION_COUNT = 10      # At 10 hotspots, Wi-Fi score maxes out
EATERY_SATURATION_COUNT = 50    # At 50 eateries, eatery penalty maxes out
```

Then re-run inference to see new scores.

---

## API & Spatial Features

### Q: The inference script is slow. Why?

**A:** Most likely: downloading spatial data from NYC Open Data API. The first run fetches:
- ~3,000 Wi-Fi hotspots
- ~25,000 eatery locations

This can take 30-60 seconds.

**Solution:** The script caches results to `data/cache/wifi.csv` and `data/cache/eateries.csv`. Subsequent runs reuse the cache and are much faster (< 5 seconds).

To force a fresh download:
```bash
rm data/cache/*.csv
python scripts/run_inference.py ...
```

---

### Q: The NYC Open Data API is timing out. What do I do?

**A:** Network issues are common with public APIs. Try:

1. **Use cached data**: If you've already downloaded once, caches persist:
   ```bash
   # No --wifi-cache or --eatery-cache flags → uses existing cache
   python scripts/run_inference.py ...
   ```

2. **Download manually and place in cache**:
   ```bash
   curl "https://data.cityofnewyork.us/resource/yjub-udmw.json?$limit=5000" > data/cache/wifi.csv
   curl "https://data.cityofnewyork.us/resource/8792-ebcp.json?$limit=10000" > data/cache/eateries.csv
   python scripts/run_inference.py ...
   ```

3. **Increase timeout** (edit `src/spatial_features.py`):
   ```python
   response = requests.get(WIFI_ENDPOINT, params={"$limit": limit}, timeout=60)
   ```

4. **Use a smaller limit** (trade accuracy for speed):
   ```python
   wifi_df = download_wifi_hotspots(cache_path="data/cache/wifi.csv", limit=2000)
   ```

---

### Q: Can I use spatial features from a different data source?

**A:** Yes. Instead of using `src/spatial_features.py`, you can:

1. **Fetch Wi-Fi/eatery data from another API**
2. **Compute densities yourself**
3. **Pass results to `compute_study_friendliness()`**

Example:
```python
from src.scoring import compute_study_friendliness

acoustic_score = 62.5
wifi_count = 8        # From your own data source
eatery_count = 30

score = compute_study_friendliness(
    acoustic_score, wifi_count, eatery_count
)
```

You don't need to use NYC Open Data if you have alternative spatial context.

---

## Common Errors & Troubleshooting

### Error: "ModuleNotFoundError: No module named 'src'"

**Solution:** Run scripts from the project root directory:
```bash
cd /path/to/NYC-cafe-study-friendliness
python scripts/run_inference.py ...
```

Not from within `scripts/`:
```bash
cd scripts/
python run_inference.py ...    # ← Wrong! Won't find src/
```

---

### Error: "FileNotFoundError: data/UrbanSound8K/audio/fold1/..."

**Solution:** Make sure UrbanSound8K is downloaded:
```bash
ls data/UrbanSound8K/audio/fold1/
# Should list .wav files
```

If empty, download from:
- https://urbansounddataset.weebly.com/urbansound8k.html (official)
- https://zenodo.org/records/1203745 (mirror)

---

### Error: "ffmpeg not found on PATH"

**Solution:** Activate the conda environment:
```bash
conda activate cafe-study
python scripts/run_inference.py ...
```

If still not found, reinstall ffmpeg:
```bash
conda install -c conda-forge ffmpeg
```

---

### Error: "No module named librosa"

**Solution:** Install the environment:
```bash
conda env create -f env.yaml
conda activate cafe-study
```

Or manually install:
```bash
pip install librosa
```

---

### Error: "ConnectionError: Unable to connect to API"

**Solution:** The NYC Open Data API might be down or slow. Try:
```bash
# Use cached data from previous run
python scripts/run_inference.py ...

# Or check API status
curl https://data.cityofnewyork.us/api/
```

---

### Error: "ValueError: audio file returned empty waveform"

**Solution:** The audio file is corrupted or in an unsupported format. Try:

1. **Check file integrity**:
   ```bash
   ffmpeg -i your_file.m4a -f null - 2>&1 | grep -i error
   ```

2. **Re-encode with ffmpeg**:
   ```bash
   ffmpeg -i your_file.m4a -q:a 9 -n your_file_fixed.m4a
   ```

3. **Convert to a more standard format**:
   ```bash
   ffmpeg -i your_file.m4a -acodec libmp3lame -ab 192k -n your_file.mp3
   ```

---

### Error: "NaN in predictions"

**Solution:** This usually indicates an audio processing issue. The code guards against NaN (replaces with 0), but check:

1. **Silent or very short audio**: Deltas may be degenerate
2. **Corrupted audio**: See above for re-encoding
3. **Empty waveform**: The file might be 0 bytes or corrupted

---

## Contributing & Support

### Q: How do I report a bug?

**A:** If you find an issue:

1. **Check `docs/FAQ.md`** (this file) for common errors
2. **Check `docs/WORKFLOW.md`** for setup instructions
3. **Run tests** to isolate the issue:
   ```bash
   pytest tests/ -v
   ```
4. **Check data integrity**: Make sure UrbanSound8K is fully downloaded
5. **Note your environment**: Python version, OS, hardware, conda env

---

### Q: How does the CNN compare to the baseline?

**A:** The CNN achieves 83.39% test accuracy vs. 73.00% SVM and 72.64% RF — a +10.4 percentage point improvement (14.3% relative error reduction). The CNN learns hierarchical spectro-temporal patterns from mel-spectrograms that hand-crafted MFCCs cannot capture. This is consistent with prior literature (Piczak 2015, Salamon & Bello 2017).

---

### Q: How long does CNN training take?

**A:** Training time depends on hardware:

| Hardware | Time | Epochs |
|----------|------|--------|
| Colab A100 GPU | ~97 seconds | 30 with cosine annealing |
| Consumer GPU (RTX 3060) | 5–15 minutes | 30 |
| CPU | Not recommended | Too slow |

The A100 is strongly recommended for practical training. Model size: 390K parameters.

---

### Q: Can I extend the project?

**A:** Yes! Suggested extensions include:

- **Spatial fusion in CNN** — concatenate Wi-Fi/eatery features to final layer (currently not implemented)
- **Temporal analysis** — track scores by time of day
- **Fine-tuning on SONYC-UST-V2** — improve NYC-specific accuracy
- **Interactive map visualization** — Folium or Mapbox frontend
- **Outlier detection** — flag cafes with unexpected scores

The CNN baseline is complete and production-ready. See `docs/WORKFLOW.md` for more details and `docs/PROJECT_TEMPLATE.md` for the full project scope.

---

### Q: Where can I find documentation?

**A:**

| File | Purpose |
|------|---------|
| `README.md` | Project overview, quick start, pipeline diagram |
| `docs/WORKFLOW.md` | Step-by-step end-to-end instructions |
| `docs/FAQ.md` | This file — common questions |
| `docs/PROJECT_TEMPLATE.md` | Original project scope and plan |
| `docs/blog.md` | Technical blog post on the approach |
| `data/README.md` | Dataset download & format details |
| `CLAUDE.md` | Architecture & code guidance for AI assistants |

---

## Additional Resources

- **UrbanSound8K Paper**: [Salamon et al. (2014)](https://urbansounddataset.weebly.com/urbansound8k.html)
- **NYC Open Data**: https://opendata.cityofnewyork.us/
- **Librosa Docs**: https://librosa.org/
- **Scikit-learn Docs**: https://scikit-learn.org/
- **Conda Docs**: https://docs.conda.io/

---

*Last updated: 2026-04-06*
