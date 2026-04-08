# Class Project Template

## 1) Project Overview

- **Title**: Silence in the City: NYC Cafe Study-Friendliness from Sound
- **Team**: Jun Lee
- **Problem statement**: Can we predict how suitable an NYC cafe is for studying by classifying environmental sounds and combining acoustic features with urban spatial data? Many students and remote workers seek quiet cafe environments but have no way to assess noise quality before visiting.
- **Hypothesis**: MFCC-based classifiers (SVM, Random Forest) can distinguish urban sound events with reasonable accuracy on UrbanSound8K. Combining sound event predictions with neighborhood context (Wi-Fi density, business counts) will produce meaningful study-friendliness scores that correlate with subjective experience.

## 2) Related Work (Short)

- **Salamon et al. (2014)** — "A Dataset and Taxonomy for Urban Sound Research." Introduced UrbanSound8K, the benchmark we use for training. Established the 10-fold CV protocol.
- **Piczak (2015)** — "Environmental Sound Classification with Convolutional Neural Networks." Showed CNNs on mel-spectrograms outperform hand-crafted features for environmental sound classification.
- **Zheng et al. (2015)** — "Urban Computing with Taxicabs." Demonstrated combining spatial data with sensor data for urban analytics.
- **Salamon & Bello (2017)** — "Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification." State-of-the-art on UrbanSound8K with data augmentation.
- **NYC Open Data** — Public datasets on Wi-Fi hotspots and food establishments used as spatial context features.

## 3) Data

- **Dataset(s)**:
  - **UrbanSound8K**: 8,732 labeled urban sound clips (<=4s), 10 classes. Used for training/benchmarking.
  - **SONYC-UST-V2**: 18,510 real-world NYC urban sound recordings with block-level geolocation and 23 fine-grained sound classes. Used as complementary NYC-specific audio data. License: CC BY 4.0.
  - **NYC Wi-Fi Hotspot Locations**: ~3,000 public Wi-Fi hotspots from NYC Open Data.
  - **Directory of Eateries**: Restaurant/bar/food locations from NYC Open Data.
  - **Field recordings**: Self-collected ~2-minute ambient audio clips from 7 NYC cafes (14 total recordings: inside + outside per cafe). Selection criteria: sufficient seating for studying and 4+ star rating on Google Maps. Format: .m4a recorded via smartphone.
- **How to access**:
  - UrbanSound8K: https://urbansounddataset.weebly.com/urbansound8k.html (requires agreement to terms)
  - Wi-Fi hotspots: https://data.cityofnewyork.us/City-Government/NYC-Wi-Fi-Hotspot-Locations/yjub-udmw
  - Eateries: https://data.cityofnewyork.us/Recreation/Directory-of-Eateries/8792-ebcp
  - SONYC-UST-V2: https://zenodo.org/records/3966543
  - See `data/README.md` for download instructions.
- **License/ethics**: UrbanSound8K is CC BY-NC 4.0. SONYC-UST-V2 is CC BY 4.0. NYC Open Data is public domain. Field recordings capture ambient sound only (no identifiable speech).
- **Train/val/test split**: UrbanSound8K uses predefined 10-fold CV. For prototype: folds 1-8 train, fold 9 val, fold 10 test. Full evaluation uses 10-fold CV.

## 4) Baseline

- **Baseline model**: MFCC (40 coefficients + delta + delta-delta, 240-dim feature vector) + Random Forest (200 trees) and SVM (RBF kernel, C=10).
- **Baseline metrics**: Classification accuracy and per-class F1 score on UrbanSound8K 10-fold CV.
  - **Actual results**: Random Forest 72.64%, SVM 73.00%
  - **Expected from literature**: ~60-70% (Salamon et al. 2014)
- **Why this is a fair baseline**: MFCC + traditional ML is the standard pre-deep-learning approach for environmental sound classification (Salamon et al. 2014). It establishes a floor for the CNN model to beat.

## 5) Proposed Method

- **What we implemented**: CNN on mel-spectrograms (UrbanSoundCNN: 4 conv blocks, 390K params) with 30 epochs of training using Adam optimizer and cosine annealing learning rate schedule.
- **Results**: CNN achieves 83.39% test accuracy, a +10.4pp improvement over baseline SVM (73.00%). This exceeds state-of-the-art benchmarks from Piczak (2015) and Salamon & Bello (2017), both ~79%.
- **Why it helps**: CNNs learn hierarchical spectro-temporal patterns that hand-crafted MFCCs cannot capture. Per-class analysis shows easy classes (gun_shot F1=0.98) vs. hard classes (air_conditioner F1=0.77).
- **Future work**: Spatial context fusion (Wi-Fi/eatery features to final layer). Audio augmentation during training.
- **Ablations**: (1) CNN without spatial features vs. with spatial features. (2) Different numbers of mel bands. (3) Effect of audio augmentation (time shift, pitch shift, noise injection).

## 6) Experiments

- **Metrics**: Classification accuracy, per-class F1 score, confusion matrix, study-friendliness score correlation with subjective ratings.
- **Compute budget**: Local machine (CPU for baseline, GPU for CNN). No cluster needed.
- **Experiment plan (Status)**:
  1. ✓ Baseline: MFCC + RF and MFCC + SVM on default split (completed: RF 72.64%, SVM 73.00%)
  2. ✓ Baseline: Full 10-fold CV for proper evaluation (completed)
  3. ✓ CNN: Train on mel-spectrograms (completed: 83.39% test accuracy)
  4. CNN + spatial context: Add Wi-Fi/eatery features to final layer (future)
  5. ✓ Cafe scoring: Apply to field recordings (completed: 7 cafes, all "Fair")
  6. Outlier detection: Flag unexpected results (future)

## 7) Reproducibility

- **How to run training**:
  ```bash
  conda env create -f env.yaml
  conda activate cafe-study
  # Run baseline notebook
  jupyter notebook notebooks/02_baseline_model.ipynb
  ```
- **How to run evaluation**:
  ```python
  from src.baseline_model import run_single_split, run_kfold_cv
  model, results = run_single_split("data/UrbanSound8K", model_type="rf")
  fold_accs, mean_acc = run_kfold_cv("data/UrbanSound8K", model_type="rf")
  ```
- **Where you log results**: Notebook outputs in `notebooks/`, saved models in `models/`, figures in `assets/`.
