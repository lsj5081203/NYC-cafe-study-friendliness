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

## References

1. Salamon, J., Jacoby, C., & Bello, J. P. (2014). A Dataset and Taxonomy for Urban Sound Research. *Proceedings of the 22nd ACM International Conference on Multimedia*.
2. Piczak, K. J. (2015). Environmental Sound Classification with Convolutional Neural Networks. *IEEE International Workshop on Machine Learning for Signal Processing*.
3. Salamon, J., & Bello, J. P. (2017). Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification. *IEEE Signal Processing Letters*, 24(3), 279-283.
