# Silence in the City: NYC Cafe Study-Friendliness from Sound

*Jun Lee — ML Class Project*

## Introduction

I spend a lot of time studying in cafes. Or trying to, anyway. If you've ever sat down at a coffee shop in Manhattan with your laptop and headphones only to realize the construction next door makes it impossible to think — you know the problem. Some places have this nice, steady background hum that you can tune out. Others hit you with random car horns every 30 seconds.

The thing is, existing tools don't help much here. Google Maps tells you the coffee is good and there's seating, but nothing about whether you'll be able to focus. Yelp might mention "noisy" or "quiet" in reviews, but that's subjective and inconsistent. A few apps measure decibel levels, but loudness alone misses the point — a 60dB air conditioner is fine, a 60dB car horn is not.

So I built a system that classifies what *types* of sounds are around a cafe, weights them by how distracting each type is, and produces a study-friendliness score. I picked 7 cafes across Greenwich Village and Long Island City, recorded 2 minutes of audio at each (inside and outside), and ran the recordings through a CNN trained on urban sounds.

The short version: all 7 cafes landed in the "Fair" range (40.56–47.74 out of 100). That's a narrower spread than I expected, and it taught me something interesting about domain gaps in ML that I'll get into later.

## Background & Related Work

### Urban Sound Classification

Environmental sound classification has grown a lot in the last decade. The landmark dataset for urban sounds is **UrbanSound8K** (Salamon, Jacoby & Bello, 2014) — 8,732 labeled audio clips organized into 10 categories like air conditioner, car horn, drilling, and siren. Each clip is ≤4 seconds, drawn from field recordings uploaded to Freesound.

The original paper established baselines around 68% accuracy using hand-crafted features (MFCCs) with SVMs and random forests. Piczak (2015) then showed that CNNs operating on mel-spectrograms — essentially treating audio as an image — could push past 73%. Salamon & Bello (2017) added data augmentation (time-stretching, pitch-shifting, background mixing) and reached 79% on the same benchmark.

The key insight across all this work: spectrograms turn audio classification into a vision problem, and CNNs are very good at vision problems.

### Spatial Urban Data

NYC Open Data provides geospatial datasets that are useful for characterizing neighborhoods. Two relevant ones:
- **NYC Wi-Fi Hotspot Locations** (~3,000 entries): free public Wi-Fi points with coordinates. A cluster of hotspots near a cafe suggests infrastructure that remote workers would find useful.
- **Directory of Eateries** (NYC DOHMH): licensed food establishments. High density = high foot traffic = likely noisier streets.

### SONYC-UST-V2

Cartwright et al. (2020) deployed acoustic sensors across NYC and produced SONYC-UST-V2 — 18,510 recordings with block-level coordinates and multilabel annotations across 23 fine-grained sound classes. This dataset represents real in-situ NYC acoustics, though its annotation scheme is different from UrbanSound8K (multilabel vs. single-label, different class taxonomies).

### Gap in Prior Work

Nobody has combined sound classification with spatial data to produce location-specific study-friendliness scores. The pieces exist — classifiers, urban datasets, open geodata — but they haven't been composed into an end-to-end scoring pipeline. That's what this project does.

## Why Machine Learning?

A fair question: do you actually need ML here? Couldn't you just measure decibels?

I thought about this. A simple approach would be: record audio → compute average dB → threshold it (below 50dB = quiet, above 70dB = loud). It's simple and interpretable.

But it misses the key dimension: sound *type*. Consider two cafes, both at 55dB average:
- Cafe A: steady hum of AC and distant traffic
- Cafe B: intermittent car horns and a dog barking nearby

Cafe A is much more study-friendly, even at the same volume. Intermittent, unpredictable sounds are cognitively disrupting in a way that steady background noise isn't. You can habituate to constant noise; you can't habituate to random surprises.

ML lets me go from "how loud is it?" to "what kind of sounds are these, and how distracting is each type?" That's the value-add over a decibel meter.

Could I have done manual labeling instead? Sure — listen to each recording, note what I hear. But that doesn't scale, it's subjective, and it doesn't give me class probabilities that I can weight and combine into a formula. The CNN gives me a probability distribution over 10 sound classes for each audio segment, which I can then feed into the scoring equation.

## Data

### UrbanSound8K

8,732 clips across 10 classes, pre-split into 10 folds. The fold structure matters — clips from the same original recording are kept together in one fold to prevent data leakage. I followed this protocol strictly for all evaluation.

Class distribution (from notebook 01):

| Class | Count | % of Dataset |
|-------|-------|-------------|
| Air conditioner | 1000 | 11.5% |
| Car horn | 429 | 4.9% |
| Children playing | 1000 | 11.5% |
| Dog bark | 1000 | 11.5% |
| Drilling | 1000 | 11.5% |
| Engine idling | 1000 | 11.5% |
| Gun shot | 374 | 4.3% |
| Jackhammer | 1000 | 11.5% |
| Siren | 929 | 10.6% |
| Street music | 1000 | 11.5% |

The dataset is roughly balanced for most classes, with car horn and gun shot being the smallest. This slight imbalance didn't significantly hurt model performance — the small classes actually turned out to be among the easiest to classify because they have very distinctive spectral signatures.

### NYC Cafe Recordings

I recorded 7 cafes across Greenwich Village, West Village, and Long Island City. Selection criteria: (1) enough seating that people actually study there, and (2) 4+ stars on Google Maps. At each cafe I recorded ~2 minutes of audio both inside and outside using my phone, giving 14 total recordings.

| Cafe | Neighborhood | Inside | Outside |
|------|-------------|--------|--------|
| Blank Street Cafe | Greenwich Village | ✓ | ✓ |
| Jacx & Co Food Hall | Long Island City | ✓ | ✓ |
| Rosecrans Cafe | West Village | ✓ | ✓ |
| Joe Coffee | Greenwich Village | ✓ | ✓ |
| Paris Baguette | Greenwich Village | ✓ | ✓ |
| Starbucks | Greenwich Village | ✓ | ✓ |
| Utopia Bagel | Long Island City | ✓ | ✓ |

The inside/outside split lets me assess two things: what acoustic environment greets you at the door, and how well the cafe's walls actually isolate you from street noise.

### Spatial Data

From NYC Open Data, I pulled Wi-Fi hotspot counts within a 200m radius of each cafe:

| Cafe | Wi-Fi Hotspots (200m) |
|------|----------------------|
| Joe Coffee | 6 |
| Rosecrans Cafe | 5 |
| Blank Street Cafe | 4 |
| Paris Baguette | 3 |
| Jacx & Co Food Hall | 2 |
| Starbucks | 2 |
| Utopia Bagel | 1 |

The eatery directory endpoint returned a 403 Forbidden error during inference — I tried a few approaches to get around it but ultimately moved on with `eatery_count=0` for all cafes. Since the eatery term only contributes a 5% penalty weight, the maximum impact is about 2.5 points on the final score.

## Methods

### Feature Extraction: MFCC

For the baseline classifiers, each audio clip becomes a 240-dimensional feature vector:

1. Extract 40 MFCC coefficients per time frame (captures spectral shape — what frequencies are dominant)
2. Compute delta (first derivative) and delta-delta (second derivative) of the MFCCs (captures how the spectral shape changes over time)
3. Take the mean and standard deviation of each coefficient across all time frames

That gives 40 coefficients × 3 (base + delta + delta-delta) × 2 (mean + std) = 240 dimensions per clip.

Why MFCCs? They're modeled on human hearing — the mel scale compresses higher frequencies where humans are less sensitive, focusing representation on the perceptually important part of the spectrum. They've been the standard feature for audio classification since the speech recognition days.

### Feature Extraction: Mel-Spectrogram

For the CNN, I used 128-bin log-mel spectrograms. These are basically heatmap images — time on the x-axis, frequency (mel-scaled) on the y-axis, color representing energy. The CNN learns to read these "images of sound" the same way an image classifier learns to read photos.

Each spectrogram has shape (128, 173) — 128 frequency bins × 173 time frames (for 4-second clips at the default hop length). The CNN sees these as single-channel grayscale images.

### Baseline Classifiers

| Model | Configuration | Rationale |
|-------|--------------|----------|
| Random Forest | 200 trees, no max depth | Good default for tabular data, handles feature interactions |
| SVM | RBF kernel, C=10, gamma='scale' | Strong on medium-dimensional features with StandardScaler |

Both wrapped in a scikit-learn Pipeline with StandardScaler. Feature normalization matters for SVM (RBF kernel is distance-based); less so for RF, but I kept the pipeline consistent.

### CNN Architecture

4 convolutional blocks with increasing filter depth:

```
Input (1, 128, 173)
→ Conv Block 1: 32 filters, 3×3, BatchNorm, ReLU, MaxPool 2×2
→ Conv Block 2: 64 filters, 3×3, BatchNorm, ReLU, MaxPool 2×2
→ Conv Block 3: 128 filters, 3×3, BatchNorm, ReLU, MaxPool 2×2
→ Conv Block 4: 256 filters, 3×3, BatchNorm, ReLU, MaxPool 2×2
→ Global Average Pooling
→ Dropout (0.3)
→ Dense 128, ReLU
→ Dropout (0.3)
→ Dense 10 (softmax)
```

Total parameters: ~390K. Trained for 30 epochs with Adam optimizer and cosine annealing LR schedule (starting at 1e-3). Training took 96.6 seconds on a Colab A100 GPU. Best validation accuracy: 80.51% at epoch 24.

I chose this architecture because it's in the sweet spot for UrbanSound8K — deep enough to learn spectro-temporal patterns, small enough to train quickly on a single GPU. Deeper models (ResNet-scale) would be overkill for 8,732 training clips and 10 classes.

### Study-Friendliness Scoring

The core idea: different sounds disrupt concentration differently. I assigned each UrbanSound8K class a distraction weight from 0 (no disruption) to 1 (maximum disruption):

| Class | Weight | Why |
|-------|--------|-----|
| Air conditioner | 0.10 | Steady hum, easily tuned out |
| Engine idling | 0.15 | Low-frequency constant drone |
| Street music | 0.40 | Variable — can be pleasant or distracting |
| Children playing | 0.50 | Moderate, variable pitch |
| Dog bark | 0.60 | Sudden, sharp, unpredictable |
| Siren | 0.85 | Loud, urgent, demands attention |
| Car horn | 0.90 | Short but attention-grabbing |
| Drilling | 0.95 | Sustained and very loud |
| Jackhammer | 0.95 | Sustained and very loud |
| Gun shot | 1.00 | Extreme — maximum alarm response |

The weights are subjective — I based them on my own experience of what breaks concentration. A more rigorous approach would survey a population of students, but for a class project this is reasonable.

The scoring formula:

```
acoustic_score = (1 - weighted_distraction) × 100
```

where `weighted_distraction` is the probability-weighted average distraction across the CNN's predicted class distribution for a recording.

The final score integrates spatial context:

```
final_score = 0.9 × acoustic_score
            + 0.1 × min(wifi_count / 10, 1.0) × 100
            - 0.05 × min(eatery_count / 50, 1.0) × 100
```

The acoustic component dominates (90% weight). Wi-Fi availability adds a small bonus (up to 10 points). Eatery density subtracts a small penalty (up to 5 points) as a proxy for foot traffic. Final scores are clamped to [0, 100] and mapped to labels:

| Score | Label |
|-------|-------|
| 75+ | Excellent |
| 55–74 | Good |
| 35–54 | Fair |
| 20–34 | Poor |
| <20 | Avoid |

## Experiments & Results

### Baseline Accuracy on UrbanSound8K

I first trained RF and SVM on a single train/test split (folds 1–8 for training, fold 9 for validation, fold 10 for testing):

| Model | Single Split (Fold 10) |
|-------|----------------------|
| Random Forest | 72.64% |
| SVM (RBF, C=10) | 73.00% |

Then I ran proper 10-fold cross-validation following the UrbanSound8K protocol:

| Model | 10-Fold CV |
|-------|----------|
| Random Forest | 67.36% ± 4.49% |
| SVM (RBF, C=10) | 68.24% ± 5.54% |

The CV scores are lower than the single-split scores because some folds are genuinely harder — fold 3 drops to 60% for RF. This is expected and shows why single-split evaluation can be misleading. The fold-to-fold variance (±4–5%) reflects real difficulty variation in the data.

### CNN Results

The CNN was trained on the same fold 1–8 / fold 9 / fold 10 split:

| Metric | Value |
|--------|-------|
| Test Accuracy (Fold 10) | **83.39%** |
| Best Validation Accuracy | 80.51% (epoch 24) |
| Training Time | 96.6 seconds (30 epochs, A100) |

That's a +10.4 percentage point improvement over SVM on the same test fold — a 14.3% relative error reduction. This aligns with Piczak (2015) and Salamon & Bello (2017), who showed CNNs on mel-spectrograms consistently outperform MFCC+SVM pipelines.

I didn't run full 10-fold CV for the CNN (it would take ~15 minutes on A100 and require retraining 10 models). The single-fold result is what I'm reporting — it's comparable to the single-fold baseline numbers (72.64% RF, 73.00% SVM), not the CV averages.

### Summary Comparison

| Model | Features | Single Split (Fold 10) | 10-Fold CV |
|-------|----------|----------------------|------------|
| Random Forest | MFCC (240-dim) | 72.64% | 67.36% ± 4.49% |
| SVM (RBF, C=10) | MFCC (240-dim) | 73.00% | 68.24% ± 5.54% |
| **CNN (4-block)** | **Mel-spectrogram** | **83.39%** | — |

### Per-Class Performance (CNN)

The CNN confusion matrix (notebook 03, cell 16) reveals which sounds are easy vs. hard:

**Easiest** (distinctive spectral signatures):
- Gun shot: F1 = 0.98 — sudden broadband burst, nothing else sounds like it
- Engine idling: F1 = 0.92 — steady low-frequency hum
- Jackhammer: F1 = 0.91 — periodic impact pattern
- Car horn: F1 = 0.87 — short tonal burst at specific frequencies

**Hardest** (overlapping spectral content):
- Dog bark: F1 = 0.77 — variable pitch, confuses with children playing
- Air conditioner: F1 = 0.78 — broad spectrum noise, varies a lot by unit
- Siren: F1 = 0.78 — frequency sweeps can resemble street music
- Children playing: F1 = 0.78 — mix of vocal sounds at varying pitches

The confusion pattern makes intuitive sense: sounds with stable, distinctive spectral structure are easy; sounds that are variable or share spectral characteristics with other classes are hard.

### Cafe Scoring Results

Applied the trained CNN to 14 cafe recordings (7 cafes × inside + outside). The "Avg Score" column is the final score after averaging inside and outside acoustic predictions, then applying the spatial adjustment (Wi-Fi bonus, eatery penalty):

| Cafe | Avg Score | Grade | Best Recording |
|------|-----------|-------|----------------|
| Jacx & Co Food Hall | 47.74 | Fair | 50.85 (outside) |
| Blank Street Cafe | 47.14 | Fair | 55.51 (inside) |
| Rosecrans Cafe | 44.95 | Fair | 46.23 (outside) |
| Joe Coffee | 42.60 | Fair | 41.33 (inside) |
| Paris Baguette | 42.42 | Fair | 47.97 (inside) |
| Starbucks | 42.42 | Fair | 46.36 (outside) |
| Utopia Bagel | 40.56 | Fair | 46.10 (inside) |

All 7 cafes scored "Fair" (35–54 range). The full range is only 7.2 points (40.56 to 47.74).

### Inside vs. Outside Analysis

The most interesting single finding: **Blank Street Cafe's inside recording scored 55.51** — the highest individual acoustic score in the dataset, and 15.2 points above its own outside recording (40.34). That inside/outside gap is the largest across all venues. It makes sense: Blank Street is a small enclosed space with soft materials that attenuate Greenwich Village street noise.

Jacx & Co Food Hall, by contrast, scored almost identically inside (50.81) and outside (50.85). A food hall's open layout provides basically zero acoustic isolation.

## Analysis & Discussion

### What Worked

1. **The CNN clearly outperforms baselines.** +10 points over SVM is a meaningful gap, achieved with a relatively simple 4-block architecture. The mel-spectrogram representation captures spectro-temporal patterns that flat MFCC vectors miss.

2. **The scoring pipeline produces interpretable output.** You get a number (0–100), a label (Fair/Good/etc.), and you can trace back to which sound classes drove the score. It's not a black box.

3. **Inside/outside recording reveals real differences.** Blank Street's 15-point gap validates that the system can detect acoustic isolation — which is exactly the kind of thing a student cares about.

### What Didn't Work

1. **All cafes scored "Fair."** A 7-point spread across 7 cafes is too narrow to be useful. If you can't distinguish cafes, the system doesn't help you pick one. The root cause is the domain gap (next section).

2. **The eatery API failed.** I got a 403 from the NYC Open Data endpoint for the Directory of Eateries. It might need an app token or the endpoint might have been deprecated. Since it only contributes 5% weight, I moved on — but it means the spatial signal is incomplete.

3. **The distraction weights are arbitrary.** I set them based on intuition. A dog bark at 0.60 vs. children playing at 0.50 — is that right? I don't have empirical evidence for these numbers. A proper calibration study (survey students about which sounds break their focus) would strengthen this.

### The Domain Gap Problem

This is the biggest limitation and the most instructive thing I learned.

UrbanSound8K has no cafe-specific sounds. There's no class for espresso machines, clinking cups, conversations, background music from speakers, or cash registers. When the CNN encounters these sounds in a cafe recording, it has to map them onto the closest UrbanSound8K class — and that's usually something like `street_music` (weight 0.40) or `children_playing` (weight 0.50).

This means all cafe recordings get force-mapped to mid-distraction classes, which compresses all acoustic scores into the 40–55 range. It doesn't mean every NYC cafe is equally study-friendly — it means my classifier can't distinguish them with these training categories.

If I were doing this again (or with more time), I'd either:
- Fine-tune the CNN on labeled cafe-ambience audio
- Add cafe-specific classes to the training set
- Use SONYC-UST-V2's finer-grained taxonomy (23 classes vs. 10)

## Ethical Considerations

A few things I thought about while working on this:

**Privacy of recordings.** I recorded audio in public and semi-public spaces (cafe interiors, sidewalks). The recordings capture ambient sound — including fragments of other people's conversations. I didn't transcribe or analyze speech content, and the recordings are processed only for sound class probabilities, but the raw audio files could theoretically be used to identify individuals. For a production system, I'd want to process audio on-device and only transmit class probabilities, never raw audio.

**Bias in the scoring system.** The distraction weights reflect my own preferences as a student. Someone with ADHD might find steady background noise more disruptive than I do; someone else might find street music relaxing rather than distracting. A single "study-friendliness" score flattens diverse cognitive needs into one number. A better version would let users set their own sensitivity profiles.

**Impact on businesses.** Publishing study-friendliness scores could steer customers toward or away from specific cafes based on factors outside the cafe's control (nearby construction, street traffic). A low score might hurt a cafe's reputation for reasons they can't fix. I'd want to be transparent about what the score measures — surrounding acoustic environment, not the cafe's own qualities.

## Limitations

1. **Domain gap** — UrbanSound8K lacks cafe-specific sounds (espresso machines, cutlery, conversation). The CNN force-maps cafe audio onto intermediate classes, compressing all acoustic scores into the 40–55 range. This is the single biggest limitation.

2. **All "Fair" clustering** — The 7.2-point spread (40.56–47.74) is a direct symptom of the domain gap. It doesn't mean every cafe is equally study-friendly; it means the model can't tell them apart well enough.

3. **Eatery API 403** — NYC Open Data eatery endpoint returned Forbidden during inference. `eatery_count=0` for all cafes. Max impact: ~2.5 points (5% weight × 50 max penalty).

4. **No CNN cross-validation** — CNN evaluated on a single held-out fold. The 83.39% number is not directly comparable to the baselines' 10-fold CV averages (67–68%). Compare it to the single-fold baseline numbers (72–73%) instead.

5. **Small sample** — 7 cafes × 2 recordings is enough to demonstrate the pipeline, not enough to draw statistical conclusions about NYC cafes.

6. **Subjective weights** — Distraction weights (0.10 for AC, 0.90 for car horn, etc.) are based on my judgment, not empirical measurement.

7. **Temporal snapshot** — Each recording is ~2 minutes at one time of day. A cafe's acoustic profile probably varies significantly between 7am and 5pm.

## Future Work

If I had more time and resources:

1. **Fix the domain gap.** Fine-tune the CNN on a dataset that includes cafe-specific sounds — or collect and label one. Even 100 labeled clips of espresso machines, conversations, and background music would help.

2. **SONYC-UST-V2 integration.** 18,510 NYC recordings with coordinates. Could use this to pre-characterize neighborhoods before even visiting a cafe.

3. **Temporal variation.** Record each cafe at multiple times of day to capture how the soundscape shifts (morning quiet vs. lunch rush vs. evening traffic).

4. **User calibration.** Let users rate how distracting they find different sounds, then personalize the scoring weights.

5. **Interactive map.** Visualize scores on a Folium/Mapbox map of NYC. The coordinate data is already there — it's just a frontend away.

6. **Resolve the eatery API.** Request an app token from NYC Open Data, or find an alternative source for business density.

## Conclusion

I set out to quantify how study-friendly NYC cafes are using sound classification and spatial data. The pipeline works end-to-end: record audio → extract features → classify sounds → weight by distraction → integrate spatial context → produce a score. The CNN achieves 83.39% accuracy on urban sound classification, which is competitive with published results on this benchmark.

The practical results were humbling. All 7 cafes scored "Fair," with barely any spread between them. That outcome taught me more than a clean gradient of scores would have — it exposed the domain gap between training data (general urban sounds) and deployment data (cafe-specific ambiences), which is one of the most common failure modes in applied ML.

If I were advising someone studying in NYC cafes based only on my data: Blank Street's inside space is genuinely quieter than the others (55.51 score, well above the pack). But honestly, for the rest, you'd be better off just checking if there's construction on the block that day.

## References

1. Salamon, J., Jacoby, C., & Bello, J. P. (2014). A Dataset and Taxonomy for Urban Sound Research. *Proceedings of the 22nd ACM International Conference on Multimedia*.
2. Piczak, K. J. (2015). Environmental Sound Classification with Convolutional Neural Networks. *IEEE International Workshop on Machine Learning for Signal Processing*.
3. Salamon, J., & Bello, J. P. (2017). Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification. *IEEE Signal Processing Letters*, 24(3), 279–283.
4. Cartwright, M., et al. (2020). SONYC Urban Sound Tagging (SONYC-UST): A Multilabel Dataset from an Urban Acoustic Sensor Network. *Proceedings of the Detection and Classification of Acoustic Scenes and Events Workshop (DCASE)*.
