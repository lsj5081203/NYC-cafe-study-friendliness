# Datasets

## 1. UrbanSound8K (Training/Benchmark)

**8,732 labeled urban sound clips across 10 classes.**

### Download

1. Go to https://urbansounddataset.weebly.com/urbansound8k.html
2. Agree to the terms and download the dataset
3. Extract to `data/UrbanSound8K/`

Alternative mirror: https://zenodo.org/records/1203745

### Expected Structure

```
data/UrbanSound8K/
├── audio/
│   ├── fold1/
│   ├── fold2/
│   ├── ...
│   └── fold10/
└── metadata/
    └── UrbanSound8K.csv
```

### License

Creative Commons Attribution Non-Commercial 4.0 (CC BY-NC 4.0).

**Citation**: Salamon, J., Jacoby, C., & Bello, J. P. (2014). A Dataset and Taxonomy for Urban Sound Research. ACM MM 2014.

---

## 2. NYC Open Data (Spatial Context)

These datasets are downloaded automatically by `src/spatial_features.py` via the NYC Open Data SODA API.

### Wi-Fi Hotspot Locations

- **Source**: https://data.cityofnewyork.us/City-Government/NYC-Wi-Fi-Hotspot-Locations/yjub-udmw
- **API**: https://data.cityofnewyork.us/resource/yjub-udmw.json
- ~3,000 public Wi-Fi hotspots across NYC

### Directory of Eateries

- **Source**: https://data.cityofnewyork.us/Recreation/Directory-of-Eateries/8792-ebcp
- **API**: https://data.cityofnewyork.us/resource/8792-ebcp.json
- Restaurants, bars, and food locations

### License

NYC Open Data is made available under the public domain.

---

## 3. Cafe Field Recordings (Self-Collected)

Place your cafe recordings in `data/cafe_recordings/`.

### Recording Protocol

- **Duration**: 30-60 second clips per location
- **Locations**: ~10-15 NYC cafes, multiple clips per location
- **Device**: Smartphone or portable recorder
- **Privacy**: Capture ambient sound only. Avoid recording identifiable speech. Use short, non-identifying segments.

### Expected Structure

```
data/cafe_recordings/
├── cafe_metadata.csv          # name, latitude, longitude, datetime, filename
├── location_01_clip_01.wav
├── location_01_clip_02.wav
├── location_02_clip_01.wav
└── ...
```

### cafe_metadata.csv format

```csv
name,latitude,longitude,datetime,filename,notes
"Joe's Coffee",40.8075,-73.9626,2026-03-15 10:30,"location_01_clip_01.wav","Saturday morning, moderate crowd"
```
