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

## 2. SONYC-UST-V2 (NYC Urban Sound)

**18,510 real-world urban sound recordings from NYC with block-level geolocation.**

### Download

1. Go to https://zenodo.org/records/3966543
2. Download the dataset archive
3. Extract to `data/SONYC-UST-V2/`

### Expected Structure

```
data/SONYC-UST-V2/
├── audio/
│   └── *.wav
├── annotations.csv
└── README.md
```

### License

Creative Commons Attribution 4.0 (CC BY 4.0).

**Citation**: Cartwright, M., et al. (2020). SONYC Urban Sound Tagging (SONYC-UST): A Multilabel Dataset from an Urban Acoustic Sensor Network. DCASE 2020.

---

## 3. NYC Open Data (Spatial Context)

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

## 4. Cafe Field Recordings (Self-Collected)

Place your cafe recordings in `data/cafe_recordings/`.

### Recording Protocol

- **Duration**: ~2 minutes per recording
- **Method**: Two recordings per cafe — one **inside** and one **outside** — to capture both the indoor ambiance and the surrounding street-level soundscape
- **Locations**: 7 NYC cafes (14 total recordings)
- **Selection criteria**: Cafes with enough seating for studying and a 4+ star rating on Google Maps
- **Device**: Smartphone
- **Format**: .m4a
- **Privacy**: Ambient sound only; no identifiable speech recorded

### Cafes Recorded

| Cafe | Neighborhood | Lat | Lon |
|------|-------------|-----|-----|
| Blank Street Cafe | Greenwich Village | 40.7353 | -73.9984 |
| Jacx & Co Food Hall | Long Island City | 40.7489 | -73.9410 |
| Joe Coffee | West Village | 40.7333 | -74.0006 |
| Paris Baguette | Long Island City | 40.7481 | -73.9402 |
| Rosecrans Cafe | West Village | 40.7339 | -74.0009 |
| Starbucks | Long Island City | 40.7495 | -73.9408 |
| Utopia Bagel | Long Island City | 40.7470 | -73.9396 |

### Expected Structure

```
data/cafe_recordings/
├── cafe_metadata.csv
├── Blank street cafe inside.m4a
├── Blank street cafe outside.m4a
├── Jack&co inside.m4a
├── Jacx&co food hall outside.m4a
├── Joe coffee inside.m4a
├── Joe coffee outside.m4a
├── Paris Baguette inside.m4a
├── Paris Baguette outside.m4a
├── Rosecrans cafe outside.m4a
├── Rosecrans inside.m4a
├── Starbucks inside.m4a
├── Starbucks outside.m4a
├── Utopia bagle ouside.m4a
└── Utopia inside.m4a
```

### cafe_metadata.csv format

```csv
name,latitude,longitude,filename,recording_type,notes
Blank Street Cafe,40.735269,-73.998427,Blank street cafe inside.m4a,inside,Greenwich Village location; enough seats to study; 4+ stars on Google Maps
```
