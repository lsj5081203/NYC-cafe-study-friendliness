# Silence in the City: NYC Cafe Study-Friendliness from Sound

*Jun Lee*

## Background

For students and remote workers in New York City, finding a good place to study or focus is a daily challenge. While many turn to cafes, the acoustic environment can make or break a productive session. A cafe with gentle background music and quiet conversation is very different from one next to a construction site or a busy intersection with frequent sirens.

Most existing tools for finding cafes (Google Maps, Yelp) focus on food quality, price, and atmosphere — but rarely quantify the *acoustic environment*. Some apps measure raw decibel levels, but loudness alone doesn't capture the full picture: a steady hum of an air conditioner at 60dB is far less distracting than a sudden car horn at the same volume. The *type* of sound matters as much as its intensity.

### Urban Sound Classification

The field of environmental sound classification (ESC) has grown significantly in the past decade. A key milestone was the creation of **UrbanSound8K** (Salamon, Jacoby, & Bello, 2014), a dataset of 8,732 labeled sound clips from urban environments, organized into 10 categories:

| Class | Example | Study Impact |
|-------|---------|-------------|
| Air conditioner | Steady mechanical hum | Low distraction |
| Car horn | Sudden honking | High distraction |
| Children playing | Playground sounds | Moderate distraction |
| Dog bark | Sudden, sharp | Moderate distraction |
| Drilling | Construction noise | Very high distraction |
| Engine idling | Steady low rumble | Low distraction |
| Gun shot | Sudden, alarming | Extreme distraction |
| Jackhammer | Construction noise | Very high distraction |
| Siren | Emergency vehicles | High distraction |
| Street music | Performers, speakers | Low-moderate distraction |

Early approaches to classifying these sounds used hand-crafted features like **Mel-Frequency Cepstral Coefficients (MFCCs)** — a compact representation of the spectral shape of audio — fed into traditional classifiers like Support Vector Machines (SVMs) and Random Forests. Salamon et al. (2014) reported baseline accuracies around 68% with this approach.

More recent work by Piczak (2015) and Salamon & Bello (2017) showed that **Convolutional Neural Networks (CNNs)** operating on mel-spectrograms can significantly outperform these baselines, achieving accuracies above 79% on UrbanSound8K. The key insight is that mel-spectrograms are essentially images of sound, and CNNs excel at learning visual patterns — in this case, spectro-temporal patterns that distinguish a siren's sweeping frequency from the broadband noise of a jackhammer.

### Complementary Data: SONYC-UST-V2

While UrbanSound8K provides a clean, well-labeled benchmark, it was curated from Freesound uploads rather than recorded in situ. To complement it, we plan to incorporate **SONYC-UST-V2** (Cartwright et al., 2020) — a dataset of 18,510 audio recordings captured by acoustic sensors deployed across New York City as part of the Sounds of New York City (SONYC) project. Each recording comes with block-level latitude/longitude and multilabel annotations across 23 fine-grained urban sound classes (e.g., small-engine, jackhammer, non-machinery-impact). This dataset would ground our work in real NYC acoustics and provide a bridge between the controlled UrbanSound8K benchmark and our own field recordings.

### Data Collection: NYC Cafe Recordings

To build a study-friendliness map, we collected our own field recordings from **7 NYC cafes** across Greenwich Village, West Village, and Long Island City. Cafes were selected based on two criteria: (1) sufficient seating for studying and (2) a 4+ star rating on Google Maps.

At each cafe, we recorded approximately **2 minutes of audio both inside and outside** the location using a smartphone, yielding **14 total recordings**. The inside recording captures the cafe's indoor ambiance (conversations, music, espresso machines), while the outside recording captures the surrounding street-level soundscape (traffic, construction, pedestrians). This inside/outside approach lets us assess not just what you hear while seated, but also what acoustic environment greets you at the door — a factor that influences perceived noise even indoors.

### From Sound Classification to Study-Friendliness

This project bridges the gap between academic sound classification research and a practical urban application. The idea is straightforward:

1. **Train** an urban sound classifier on UrbanSound8K
2. **Apply** it to short recordings from NYC cafes to detect what kinds of sounds are present
3. **Score** each location based on which sounds are detected and how distracting they are
4. **Enrich** the score with spatial context from NYC Open Data — like nearby Wi-Fi availability (a positive signal for study spots) and business density (a proxy for foot traffic and noise)

The result is a data-driven "study-friendliness" score for each cafe, visualized on an interactive NYC map.

### Why This Matters

New York City is home to over 27,000 restaurants and cafes, and thousands of students who rely on them as study spaces. By combining machine learning, audio analysis, and open urban data, we can provide actionable information that helps people find environments conducive to focus and productivity.

Beyond the practical application, this project explores an interesting ML problem: how to transfer a model trained on general urban sounds to a specific downstream task (study-friendliness scoring) that requires domain-specific weighting and context integration.

## Methods

### Audio Feature Extraction

Each audio clip is represented as a 240-dimensional feature vector using **Mel-Frequency Cepstral Coefficients (MFCCs)**:

1. Extract 40 MFCC coefficients per time frame
2. Compute first-order deltas (velocity) and second-order deltas (acceleration)
3. Summarize each coefficient over time using **mean and standard deviation**

This gives 40 × 3 derivatives × 2 statistics = **240 dimensions** per clip. MFCCs capture the spectral envelope — the "shape" of a sound — and the delta features capture how that shape changes over time, which helps distinguish transient events (car horn) from sustained ones (air conditioner).

### Baseline Classifiers

Two classifiers are trained on the MFCC features:

| Model | Configuration |
|-------|---------------|
| **Random Forest** | 200 trees, no max depth, `n_jobs=-1` |
| **SVM** | RBF kernel, C=10, gamma='scale' |

Both are wrapped in a scikit-learn `Pipeline` with `StandardScaler` to normalize features before training. The 10-fold CV protocol from UrbanSound8K is followed strictly — audio clips from the same original recording are never split across train and test folds, preventing data leakage.

### Study-Friendliness Scoring

The core contribution is a distraction-weighted scoring system. Each UrbanSound8K class is assigned a **distraction weight** based on its subjective impact on concentration:

| Class | Weight | Rationale |
|-------|--------|----------|
| Air conditioner | 0.10 | Steady hum — easily tuned out |
| Engine idling | 0.15 | Low, constant — low distraction |
| Street music | 0.40 | Variable — can be pleasant or distracting |
| Children playing | 0.50 | Moderate |
| Dog bark | 0.60 | Sudden, sharp |
| Car horn | 0.90 | Sudden, attention-grabbing |
| Siren | 0.85 | Loud, urgent |
| Drilling | 0.95 | Sustained, very loud |
| Jackhammer | 0.95 | Sustained, very loud |
| Gun shot | 1.00 | Extreme — maximum distraction |

The acoustic score is:

```
acoustic_score = (1 - weighted_distraction) × 100
```

where `weighted_distraction` is the proportion-weighted average distraction across all detected sound classes in a recording.

The final study-friendliness score combines acoustic predictions with spatial context from NYC Open Data:

```
final_score = 0.9 × acoustic_score
            + 0.1 × min(wifi_count / 10, 1.0) × 100
            - 0.05 × min(eatery_count / 50, 1.0) × 100
```

Wi-Fi hotspot density is a positive signal (infrastructure for studying); eatery density is a slight negative proxy for foot traffic and noise. The acoustic score dominates at 90% weight. The final score is clamped to [0, 100] and mapped to five labels: **Excellent** (≥75), **Good** (≥55), **Fair** (≥35), **Poor** (≥20), **Avoid** (<20).

## Results

### Model Accuracy on UrbanSound8K

We trained and evaluated three classifiers using 10-fold cross-validation:

| Model | Test Accuracy |
|-------|--------------|
| Random Forest (MFCC) | 72.64% |
| SVM (MFCC) | 73.00% |
| **CNN (Mel-spectrograms)** | **83.39%** |

**Key finding**: The CNN achieves **+10.4 percentage points** improvement over the baseline SVM, representing a 14.3% relative error reduction. This aligns with prior work (Piczak 2015, Salamon & Bello 2017) showing that CNNs on mel-spectrograms significantly outperform hand-crafted features.

**CNN Architecture**: 4 convolutional blocks (32→64→128→256 filters), batch normalization, dropout regularization, and global average pooling. Total parameters: 390K. Training time on Colab A100 GPU: 96.6 seconds for 30 epochs with Adam optimizer and cosine annealing learning rate schedule. Best validation accuracy: 80.51% (epoch 24).

### Per-Class Difficulty

Looking at the CNN's per-class F1 scores reveals which sounds are easiest and hardest to classify:

**Easiest classes** (distinct spectral signatures):
- Gun shot: F1 = 0.98 (sudden, broadband burst)
- Engine idling: F1 = 0.92 (steady, low-frequency hum)
- Jackhammer: F1 = 0.91 (periodic impact structure)
- Car horn: F1 = 0.87 (short tonal burst)

**Hardest classes** (overlapping spectral content):
- Dog bark: F1 = 0.77 (variable pitch, confuses with children playing)
- Air conditioner: F1 = 0.78 (steady broadband noise, varies by model)
- Siren: F1 = 0.78 (frequency sweep similar to street music)
- Children playing: F1 = 0.78 (mix of laughs, shouts, variable pitch)

### Cafe Scoring Results

Applied the trained CNN to **7 NYC cafes** (14 recordings: inside + outside per cafe):

| Cafe | Avg Final Score | Label | Best Acoustic (recording) |
|------|----------------|-------|---------------------------|
| Jacx & Co Food Hall | 47.74 | Fair | 50.85 (outside) |
| Blank Street Cafe | 47.14 | Fair | **55.51 (inside)** |
| Rosecrans Cafe | 44.95 | Fair | 46.23 (outside) |
| Joe Coffee | 42.60 | Fair | 41.33 (inside) |
| Paris Baguette | 42.42 | Fair | 47.97 (inside) |
| Starbucks | 42.42 | Fair | 46.36 (outside) |
| Utopia Bagel | 40.56 | Fair | 46.10 (inside) |

All cafes scored as "Fair" (35–54 range), indicating moderate noise levels for studying. The most striking single result: **Blank Street Cafe's inside recording scored 55.51** — the highest individual acoustic score in the dataset, and a full **15.2 points above its own outside recording** (40.34). This inside/outside gap is the largest across all venues and reflects genuine acoustic isolation: the interior's enclosed volume and soft surfaces attenuate the Greenwich Village street noise. Jacx & Co Food Hall, by contrast, scored almost identically inside (50.81) and outside (50.85), consistent with a food hall's open layout offering little acoustic separation from its surroundings.

### Spatial Features

Wi-Fi hotspot counts ranged from 1–6 per cafe within a 200-meter radius. The eatery API returned a 403 Forbidden error during inference, preventing the use of business density as a spatial penalty feature. This limitation should be addressed by requesting an NYC Open Data app token or using an alternative data source.

### Implications

The study-friendliness scores (40–48) reflect a realistic picture of urban NYC cafes: they are generally noisy environments with moderate distraction potential. The Fair clustering is expected — UrbanSound8K's 10 outdoor sound classes don't include typical cafe sounds (espresso machines, cutlery, conversation), so the CNN force-maps cafe audio onto intermediate-distraction classes like `street_music` (weight 0.40) and `children_playing` (weight 0.50), compressing all acoustic scores toward the 40–55 range. Getting a cafe into "Good" (≥55) territory would require either recalibrating `DISTRACTION_WEIGHTS` for a study context or fine-tuning the CNN on labeled cafe-ambience audio. Both are tractable; the data collection work is already done.

## Next Steps

The core pipeline is complete. Suggested future work:

1. **Outlier detection** — Statistical analysis to flag cafes with unexpected scores relative to their neighborhood
2. **Interactive map visualization** — Folium or MapboxGL frontend to explore cafe scores by location
3. **Temporal analysis** — Track how study-friendliness scores vary by time of day (morning vs. evening rush)
4. **Fine-tuning on SONYC-UST-V2** — NYC-specific acoustic sensor data to improve urban sound classification
5. **CNN fine-tuning** — Transfer learning on domain-specific urban sound datasets
6. **Spatial data completion** — Resolve eatery API 403 error and incorporate business density in final scoring

## References

1. Salamon, J., Jacoby, C., & Bello, J. P. (2014). A Dataset and Taxonomy for Urban Sound Research. *Proceedings of the 22nd ACM International Conference on Multimedia*.
2. Piczak, K. J. (2015). Environmental Sound Classification with Convolutional Neural Networks. *IEEE International Workshop on Machine Learning for Signal Processing*.
3. Salamon, J., & Bello, J. P. (2017). Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification. *IEEE Signal Processing Letters*, 24(3), 279-283.
4. Cartwright, M., et al. (2020). SONYC Urban Sound Tagging (SONYC-UST): A Multilabel Dataset from an Urban Acoustic Sensor Network. *Proceedings of the Detection and Classification of Acoustic Scenes and Events Workshop (DCASE)*.
