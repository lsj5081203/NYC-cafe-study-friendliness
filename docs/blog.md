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

8,732 clips across 10 classes (air conditioner, car horn, children playing, dog bark, drilling, engine idling, gun shot, jackhammer, siren, street music), pre-split into 10 folds. Most classes have ~1,000 clips; car horn (429) and gun shot (374) are smaller. The fold structure keeps clips from the same original recording together to prevent data leakage — I followed this protocol strictly. The mild imbalance didn't hurt performance: car horn and gun shot ended up among the easiest classes due to their distinctive spectral signatures.

### NYC Cafe Recordings

I recorded 7 cafes across Greenwich Village, West Village, and Long Island City: Blank Street, Jacx & Co Food Hall, Rosecrans, Joe Coffee, Paris Baguette, Starbucks, Utopia Bagel. Selection criteria: (1) enough seating to actually study, and (2) 4+ stars on Google Maps. At each cafe I recorded ~2 minutes of audio both inside and outside using my phone (14 recordings total). The inside/outside split lets me assess both the seated environment and how well the walls isolate you from street noise.

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

### Feature Extraction

For the baseline classifiers, each clip becomes a 240-dim MFCC vector: 40 coefficients × 3 (base + delta + delta-delta) × 2 (mean + std over time). MFCCs capture spectral shape and the deltas capture how it changes — that's what distinguishes a transient sound (car horn) from a sustained one (AC). MFCCs are modeled on human hearing, which is why they've been the standard audio feature for decades.

For the CNN, I used 128-bin log-mel spectrograms instead — heatmap images of sound, with time on x, frequency on y, energy as color. Each one has shape (128, 173) for a 4-second clip. The CNN treats them as grayscale images.

### Baseline Classifiers

| Model | Configuration |
|-------|---------------|
| Random Forest | 200 trees, no max depth |
| SVM | RBF kernel, C=10, gamma='scale' |

Both wrapped in a scikit-learn Pipeline with StandardScaler.

### CNN Architecture

4 convolutional blocks with increasing filter depth:

```
Input (1, 128, 173)
→ Conv Block 1: 32 filters, 3×3, BatchNorm, ReLU, MaxPool 2×2
→ Conv Block 2: 64 filters, 3×3, BatchNorm, ReLU, MaxPool 2×2
→ Conv Block 3: 128 filters, 3×3, BatchNorm, ReLU, MaxPool 2×2
→ Conv Block 4: 256 filters, 3×3, BatchNorm, ReLU, MaxPool 2×2
→ Global Average Pooling → Dropout (0.3) → Dense 128 → Dropout (0.3) → Dense 10
```

Total parameters: ~390K. Trained for 30 epochs with Adam + cosine annealing (starting at 1e-3). Training took 96.6 seconds on a Colab A100. Best validation accuracy: 80.51% at epoch 24.

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

CV scores are lower than the single-split scores because some folds are genuinely harder — fold 3 drops to 60% for RF. The ±4–5% variance reflects real difficulty variation in the data.

### CNN Results

The CNN was trained on the same fold 1–8 / fold 9 / fold 10 split:

| Metric | Value |
|--------|-------|
| Test Accuracy (Fold 10) | **83.39%** |
| Best Validation Accuracy | 80.51% (epoch 24) |
| Training Time | 96.6 seconds (30 epochs, A100) |

That's +10.4 percentage points over SVM on the same test fold — a 14.3% relative error reduction, consistent with Piczak (2015) and Salamon & Bello (2017). I didn't run full 10-fold CV for the CNN (would take ~15 minutes and 10 retrainings), so the 83.39% should be compared to the single-fold baselines (72.64%, 73.00%), not the CV averages.

### Summary Comparison

| Model | Features | Single Split (Fold 10) | 10-Fold CV |
|-------|----------|----------------------|------------|
| Random Forest | MFCC (240-dim) | 72.64% | 67.36% ± 4.49% |
| SVM (RBF, C=10) | MFCC (240-dim) | 73.00% | 68.24% ± 5.54% |
| **CNN (4-block)** | **Mel-spectrogram** | **83.39%** | — |

### Per-Class Performance (CNN)

Per-class F1 scores reveal which sounds are easy vs. hard:

**Easiest**: Gun shot (0.98), Engine idling (0.92), Jackhammer (0.91), Car horn (0.87) — all have stable, distinctive spectral signatures.

**Hardest**: Dog bark (0.77), Air conditioner (0.78), Siren (0.78), Children playing (0.78) — variable pitch or broad spectrum, often confused with each other.

### Cafe Scoring Results

Applied the trained CNN to 14 cafe recordings (7 cafes × inside + outside). "Avg Score" is the final score after averaging inside/outside acoustic predictions and applying the spatial adjustment:

| Cafe | Avg Score | Grade | Best Recording |
|------|-----------|-------|----------------|
| Jacx & Co Food Hall | 47.74 | Fair | 50.85 (outside) |
| Blank Street Cafe | 47.14 | Fair | 55.51 (inside) |
| Rosecrans Cafe | 44.95 | Fair | 46.23 (outside) |
| Joe Coffee | 42.60 | Fair | 41.33 (inside) |
| Paris Baguette | 42.42 | Fair | 47.97 (inside) |
| Starbucks | 42.42 | Fair | 46.36 (outside) |
| Utopia Bagel | 40.56 | Fair | 46.10 (inside) |

All 7 cafes scored "Fair" — the full range is only 7.2 points (40.56–47.74).

### Inside vs. Outside Analysis

The most interesting finding: **Blank Street Cafe's inside recording scored 55.51** — the highest individual acoustic score, 15.2 points above its own outside recording (40.34). That gap is the largest across all venues, reflecting genuine acoustic isolation from the small enclosed space and soft materials. Jacx & Co Food Hall, by contrast, scored almost identically inside (50.81) and outside (50.85) — a food hall's open layout provides zero acoustic isolation.

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

1. **Domain gap** — UrbanSound8K lacks cafe-specific sounds (espresso machines, cutlery, conversation), so the CNN force-maps cafe audio onto intermediate classes. This compresses scores into the 40–55 range and is the single biggest limitation.
2. **All "Fair" clustering** — The 7.2-point spread is a symptom of the domain gap, not evidence that all cafes are equally study-friendly.
3. **Eatery API 403** — `eatery_count=0` for all cafes. Max impact: ~2.5 points.
4. **No CNN cross-validation** — Single held-out fold only. Compare 83.39% to single-fold baselines (72–73%), not CV averages.
5. **Small sample** — 7 cafes × 2 recordings demonstrates the pipeline, not statistical generalization.
6. **Subjective weights** — Distraction weights based on my judgment, not empirical measurement.
7. **Temporal snapshot** — ~2 minutes per recording at one time of day; cafe soundscapes vary throughout the day.

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

If I were advising someone studying in NYC cafes based on my data: Blank Street's inside space is genuinely quieter (55.51 score). For the rest, honestly, just check if there's construction on the block that day.

## References

1. Salamon, J., Jacoby, C., & Bello, J. P. (2014). A Dataset and Taxonomy for Urban Sound Research. *Proceedings of the 22nd ACM International Conference on Multimedia*.
2. Piczak, K. J. (2015). Environmental Sound Classification with Convolutional Neural Networks. *IEEE International Workshop on Machine Learning for Signal Processing*.
3. Salamon, J., & Bello, J. P. (2017). Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification. *IEEE Signal Processing Letters*, 24(3), 279–283.
4. Cartwright, M., et al. (2020). SONYC Urban Sound Tagging (SONYC-UST): A Multilabel Dataset from an Urban Acoustic Sensor Network. *Proceedings of the Detection and Classification of Acoustic Scenes and Events Workshop (DCASE)*.
