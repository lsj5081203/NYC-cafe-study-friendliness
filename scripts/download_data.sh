#!/bin/bash
# Download NYC Open Data datasets
# UrbanSound8K must be downloaded manually (requires terms agreement)

set -e

DATA_DIR="$(dirname "$0")/../data"
mkdir -p "$DATA_DIR"

echo "=== NYC Open Data Downloads ==="

# Wi-Fi Hotspot Locations
echo "Downloading NYC Wi-Fi Hotspot Locations..."
curl -o "$DATA_DIR/nyc_wifi_hotspots.csv" \
  "https://data.cityofnewyork.us/api/views/yjub-udmw/rows.csv?accessType=DOWNLOAD"
echo "  Saved to $DATA_DIR/nyc_wifi_hotspots.csv"

# Directory of Eateries
echo "Downloading NYC Directory of Eateries..."
curl -o "$DATA_DIR/nyc_eateries.csv" \
  "https://data.cityofnewyork.us/api/views/8792-ebcp/rows.csv?accessType=DOWNLOAD"
echo "  Saved to $DATA_DIR/nyc_eateries.csv"

echo ""
echo "=== Done ==="
echo ""
echo "NOTE: UrbanSound8K must be downloaded manually:"
echo "  1. Go to https://urbansounddataset.weebly.com/urbansound8k.html"
echo "  2. Agree to terms and download"
echo "  3. Extract to $DATA_DIR/UrbanSound8K/"
