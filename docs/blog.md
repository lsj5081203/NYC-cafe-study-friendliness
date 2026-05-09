# Silence in the City: NYC Cafe Study-Friendliness from Sound

*Jun Lee — ML Class Project*

Code and notebooks: [github.com/lsj5081203/NYC-cafe-study-friendliness](https://github.com/lsj5081203/NYC-cafe-study-friendliness)

## Introduction

I study in cafes more than I probably should. My apartment is fine, but it is not always a good place to focus. So I look for rooms where I can open my laptop and stop noticing every sound.

What bothers me is that reviews do not really answer that question. Google Maps tells me coffee, seating, stars, maybe whether the place is crowded. Review pages rarely tell me what the room sounds like. A cafe can look perfect in photos and still be a bad place to read.

Noise level alone is not enough either. A low air conditioner hum can be fine. A car horn at the same average decibel level is not fine. One disappears after a minute. A horn cuts through whatever sentence I was trying to read.

So I made a small sound-based ranking system. I split cafe recordings into four-second chunks, trained a classifier on UrbanSound8K, and ran it on audio from seven New York cafes. Then I turned the predicted sound classes into a study-friendliness score. I also added two location signals: nearby public Wi-Fi and nearby eateries.

I expected a simple ranking. Quiet cafes at the top, chaotic ones at the bottom. Nope. All seven cafes landed in the "Fair" band, between 40.56 and 47.74 out of 100. At first I thought the scoring formula had failed. After looking at the predictions, the problem seemed more basic: the model knew street sounds, not cafe sounds.

## Background & Related Work

### Urban Sound Classification

I trained the classifier on UrbanSound8K, the dataset Salamon, Jacoby, and Bello introduced in 2014. UrbanSound8K has 8,732 labeled clips and ten classes: air conditioner, car horn, children playing, dog bark, drilling, engine idling, gun shot, jackhammer, siren, and street music. UrbanSound8K also comes with ten folds. I kept them because clips from the same source recording can sound very similar, and random splits would make testing too easy.

Early UrbanSound8K baselines used hand-built audio features, especially MFCCs. Piczak (2015) then used convolutional neural networks on spectrogram-like inputs. Salamon and Bello (2017) added data augmentation, including pitch shifting, time stretching, and background mixing. For my project, that history mattered because it pushed me toward mel-spectrograms instead of only summary features.

MFCCs still made sense as a baseline. They give a short description of spectral shape, and traditional classifiers can use them quickly. Mel-spectrograms keep more timing. A CNN can see a horn burst, a jackhammer rhythm, or a siren-like sweep that a mean MFCC vector may blur.

### Spatial Urban Data

For location, I used NYC Open Data. I pulled public Wi-Fi hotspot locations, and I tried to use food establishment records too. These datasets do not measure sound. They are more like crude hints about the block. Wi-Fi helps if you want to study. Many nearby food places may mean more foot traffic, more doors opening, more delivery bikes, more street noise. Not always, but often enough to test.

### SONYC-UST-V2

I also looked at SONYC-UST-V2. SONYC-UST-V2 is a New York sound dataset built from acoustic sensors, with 18,510 recordings and more labels than UrbanSound8K. I did not train on it because the label setup is different. Switching datasets late would have become a different project. If I were doing this again, I would look there before adding model complexity.

### Gap in Prior Work

Closest prior work sits in two areas: environmental sound classification and urban acoustic sensing. I did not find a project that turns cafe recordings plus city data into a study-friendliness score. I am not claiming a new architecture. I am asking a smaller question. Can sound labels tell me anything about where I might sit and work?

## Why Machine Learning?

A cheap version of this project would be a decibel meter. Record a cafe, average the volume, and call quiet places better. I thought about doing that. Clean, and easier to explain.

But loudness misses the thing I care about. Imagine two cafes with the same average volume. One has HVAC hum and soft conversation. Another has horns, delivery trucks, and a dog barking outside. Both can share the same average number. My attention does not treat them the same.

Classification gave me one extra layer. Instead of only asking "how loud is it?", I could ask "what kind of sound is it?" That is still a limited question, because UrbanSound8K has the wrong vocabulary for cafes. But without that step, I would only be building a noise meter with a nicer name.

## Data

### UrbanSound8K

UrbanSound8K has 8,732 clips. Most classes have about 1,000 examples, while car horn and gun shot are smaller. I used folds 1-8 for training, fold 9 for validation, and fold 10 for the reported test result. For the baseline models, I also ran 10-fold cross-validation.

![UrbanSound8K class distribution: ~1,000 clips per class with car horn and gun shot smaller.](./figures/class_distribution.png)
*Figure 1: UrbanSound8K class distribution. Most classes are close to balanced; car horn and gun shot are smaller but still distinctive.*

### NYC Cafe Recordings

I recorded seven cafes in Greenwich Village, the West Village, and Long Island City. My list was Blank Street Cafe, Jacx & Co Food Hall, Rosecrans Cafe, Joe Coffee, Paris Baguette, Starbucks, and Utopia Bagel. I picked places that looked plausible for studying: some seating, decent reviews, and locations I could reach.

At each cafe I recorded about two minutes inside and about two minutes outside on my phone. I ended up with 14 recordings. I kept the inside/outside split because a cafe can sit on a loud block and still feel calm once you are inside.

### Spatial Data

I counted NYC public Wi-Fi hotspots within 200 meters of each cafe. Joe Coffee had 6 nearby hotspots. Rosecrans had 5, Blank Street had 4, and Paris Baguette had 3. Jacx & Co and Starbucks each had 2. Utopia Bagel had 1.

I also wanted nearby eatery counts. Eatery counts broke. During inference, the NYC Open Data eatery endpoint returned HTTP 403, so I left `eatery_count=0` for every cafe. Since the eatery term only has a 5% weight, the final scores do not move much. Still, the spatial side of the project ended up thinner than planned.

## Methods

### Feature Extraction

For the Random Forest and SVM baselines, I represented each clip with 240 MFCC features. My feature vector includes 40 coefficients, their deltas, their delta-deltas, and the mean and standard deviation over time.

For the CNN, I used 128-bin log-mel spectrograms. A 4-second clip at 22,050 Hz becomes a 128 by 173 array. Frequency is vertical. Time is horizontal. Values represent energy.

### Baseline Classifiers

My baselines were a Random Forest with 200 trees and an RBF-kernel SVM with `C=10`. I scaled the SVM features. Random Forest did not need scaling.

### CNN Architecture

My CNN has four convolutional blocks: 32, 64, 128, and 256 channels. Each block uses a 3 by 3 convolution, batch normalization, ReLU, and max pooling. After that, the model uses adaptive average pooling and dropout. A linear layer maps 256 features to the 10 UrbanSound8K classes.

My model has about 390K parameters. I trained it for 30 epochs on a Colab A100 with Adam and cosine annealing. Best validation accuracy was 80.51% at epoch 24.

### Study-Friendliness Scoring

For each 4-second cafe window, the CNN predicts one UrbanSound8K class. I count those classes and convert the class mix into a distraction score. Steady sounds get low weights: air conditioner is 0.10 and engine idling is 0.15. Sharp or alarming sounds get high weights: siren is 0.85, car horn is 0.90, drilling and jackhammer are 0.95, and gun shot is 1.00.

Acoustic score:

```
acoustic_score = (1 - weighted_distraction) x 100
```

Then I add the spatial terms:

```
final_score = 0.9 x acoustic_score
            + 0.1 x min(wifi_count / 10, 1.0) x 100
            - 0.05 x min(eatery_count / 50, 1.0) x 100
```

I gave sound most of the weight because sound is the point of the project. My class weights are subjective. I chose them from my own sense of what breaks concentration, not from a survey.

## Experiments & Results

### Baseline Accuracy on UrbanSound8K

On the single fold-10 test split, the Random Forest reached 72.64% accuracy and the SVM reached 73.00%. In 10-fold cross-validation, the scores dropped: 67.36% +/- 4.49% for Random Forest and 68.24% +/- 5.54% for SVM. Some folds are just harder. So I do not ignore the official split.

![Side-by-side confusion matrices for Random Forest and SVM on fold 10.](./figures/baseline_confusion_matrices.png)
*Figure 2: Baseline confusion matrices on fold 10. The variable classes are harder; the sharper transient sounds are cleaner.*

### CNN Results

The CNN reached 83.39% test accuracy on fold 10. On that same split, it beat the SVM by about 10.4 percentage points. I did not run 10-fold cross-validation for the CNN, so I should not compare it to the baseline CV average. So I compare it to the single-split baseline.

![CNN training curves for loss and accuracy across 30 epochs.](./figures/cnn_training_curves.png)
*Figure 3: CNN training curves. Validation accuracy levels off around 80%, and the train/validation gap stays fairly small.*

### Summary Comparison

| Model | Features | Single Split (Fold 10) | 10-Fold CV |
|-------|----------|------------------------|------------|
| Random Forest | MFCC (240-dim) | 72.64% | 67.36% +/- 4.49% |
| SVM (RBF, C=10) | MFCC (240-dim) | 73.00% | 68.24% +/- 5.54% |
| CNN (4-block) | Mel-spectrogram | 83.39% | not run |

### Per-Class Performance (CNN)

Gun shot, engine idling, jackhammer, and car horn were easiest. Dog bark, air conditioner, siren, and children playing were harder. Looking at the confusion matrix, the pattern made sense. Some events have a clear shape. Others change a lot from clip to clip.

![CNN confusion matrix on fold 10 with 83.39% test accuracy.](./figures/cnn_confusion_matrix.png)
*Figure 4: CNN confusion matrix on the test fold. Most remaining errors are between classes with overlapping spectra or variable timing.*

### Cafe Scoring Results

When I applied the CNN to my cafe recordings, every cafe landed in the "Fair" range.

| Cafe | Avg Score | Grade | Best Recording |
|------|-----------|-------|----------------|
| Jacx & Co Food Hall | 47.74 | Fair | 50.85 outside |
| Blank Street Cafe | 47.14 | Fair | 55.51 inside |
| Rosecrans Cafe | 44.95 | Fair | 46.23 outside |
| Joe Coffee | 42.60 | Fair | 41.33 inside |
| Paris Baguette | 42.42 | Fair | 47.97 inside |
| Starbucks | 42.42 | Fair | 46.36 outside |
| Utopia Bagel | 40.56 | Fair | 46.10 inside |

![Horizontal bar chart of cafe study-friendliness scores, all in the Fair range.](./figures/cafe_scores_bar.png)
*Figure 5: Final cafe scores. The spread is only 7.2 points, which is too narrow for a recommendation system.*

### Inside vs. Outside Analysis

Blank Street Cafe had the largest inside/outside split. Its inside recording scored 55.51. Its outside recording scored 40.34. That 15-point gap matched what I remember hearing there: inside felt more sealed off than the sidewalk. Jacx & Co, which is a food hall, barely separated inside from outside.

![Stacked bar chart of predicted sound class distribution per cafe, comparing inside and outside.](./figures/class_distribution_per_cafe.png)
*Figure 6: Predicted class distribution for cafe recordings. The model forces many windows into mid-distraction UrbanSound8K classes because it has no cafe-specific labels.*

## Analysis & Discussion

### What Worked

My classifier learned UrbanSound8K well enough for the class benchmark. CNN beat the MFCC baselines, and the confusion matrices were readable. I could point to the mistakes instead of just reporting one accuracy number.

I also liked the inside/outside comparison. Inside/outside kept the project from becoming only a model leaderboard. Blank Street separated in the direction I expected, even with a limited model. After seeing that, the pipeline felt less like pure noise.

### What Didn't Work

Cafe ranking was the weak part. Seven cafes within 7.2 points is too compressed to guide an actual choice. If I were walking around with this score on my phone, it would not tell me much.

Label set caused the bigger problem. UrbanSound8K has no espresso machine class. No dishes. No overlapping conversation. No cash register, soft playlist, chair scrape, milk steamer, or quiet room tone. When the model hears those sounds, it still must choose an UrbanSound8K label. Cafe recordings get pulled toward the middle.

### The Domain Gap Problem

Oddly, the project became more useful to me when the ranking got worse. A classifier can recognize urban sound events and still miss cafe ambience. Training and cafe tasks look close enough to tempt you. Then the scores collapse, and the gap shows up.

I had read about this kind of applied ML failure. Seeing it in my own notebook felt different. Benchmark accuracy looked fine. Cafe audio had different categories. Warning sign was not subtle: all seven cafes landed in the same band.

## Ethical Considerations

Recording in public and semi-public spaces is the privacy issue I kept coming back to. I did not transcribe speech or analyze words, but raw audio can still catch pieces of conversation. A deployed version should process audio locally and store only class counts, probabilities, or summary scores.

Score design also has a fairness problem. My distraction weights are my preferences. Other students may react to sound differently. A single public score could hurt a small business for reasons outside its control, like construction, traffic, or one loud afternoon. If this ever became public-facing, it would need personalization and a plain explanation of what the score measures.

## Limitations

1. **Domain gap** — UrbanSound8K lacks cafe-specific classes, so cafe sounds are forced into the wrong vocabulary.
2. **All "Fair" clustering** — The narrow score range is a model limitation, not proof that the cafes are equivalent.
3. **Eatery API 403** — The eatery penalty was not applied because the endpoint failed.
4. **No CNN cross-validation** — The CNN result is one held-out fold, not a full CV estimate.
5. **Small sample** — Seven cafes and one visit per setting are enough for a prototype, not a city-wide conclusion.
6. **Subjective weights** — The distraction weights need calibration from actual users.
7. **Temporal snapshot** — Cafe sound changes by time of day, day of week, and nearby street activity.

## Future Work

With three more months, I would start by collecting cafe-specific labels. A few hundred 4-second clips would already help. I would label conversation, espresso machine, dishes, milk steamer, background music, traffic bleed, and quiet room tone. Less glamorous than changing the model, but probably more useful.

After that I would fix the location data and record each cafe at several times of day. Morning, lunch, late afternoon. A cafe can feel like a different place depending on the hour. I would also let users tune the distraction weights.

## Conclusion

I set out to rank NYC cafes by study-friendliness using sound classification. Classifier did well on its benchmark. My notebooks produce the figures, scores, and tables. But the cafe ranking was weak because the model did not have the right vocabulary for the room I cared about.

So no, I do not think I built a finished cafe-ranking tool. I built a pipeline that exposed a bad assumption. Code ran. Figures came out. UrbanSound8K accuracy looked respectable. And still, I would not ship the final score as advice on where to study.

If I had to choose from the seven cafes based only on this experiment, I would trust the Blank Street inside recording most. For everything else, I would treat the scores as a warning about domain mismatch, not as a guide.

## References

1. Salamon, J., Jacoby, C., & Bello, J. P. (2014). A Dataset and Taxonomy for Urban Sound Research. *Proceedings of the 22nd ACM International Conference on Multimedia*.
2. Piczak, K. J. (2015). Environmental Sound Classification with Convolutional Neural Networks. *IEEE International Workshop on Machine Learning for Signal Processing*.
3. Salamon, J., & Bello, J. P. (2017). Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification. *IEEE Signal Processing Letters*, 24(3), 279-283.
4. Cartwright, M., et al. (2020). SONYC Urban Sound Tagging (SONYC-UST): A Multilabel Dataset from an Urban Acoustic Sensor Network. *Proceedings of the Detection and Classification of Acoustic Scenes and Events Workshop (DCASE)*.
