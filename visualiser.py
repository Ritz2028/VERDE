```python
import os
import numpy as np
import pandas as pd

# =====================================================
# VERDE - Jaipur Inference Grid Generator
# =====================================================

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Approximate Jaipur city bounding box
LAT_MIN, LAT_MAX = 26.75, 27.15
LON_MIN, LON_MAX = 75.65, 76.05
GRID_STEP = 0.005

# Generate latitude and longitude coordinates
lats = np.arange(LAT_MIN, LAT_MAX, GRID_STEP)
lons = np.arange(LON_MIN, LON_MAX, GRID_STEP)

# Create a grid of all coordinate combinations
latitude, longitude = np.meshgrid(lats, lons)
latitude = latitude.ravel()
longitude = longitude.ravel()

# Reproducible synthetic environmental inputs
rng = np.random.default_rng(seed=42)

# Small spatial variations for demonstration purposes
temperature = (
    27
    + 2.0 * (latitude - 26.95)
    + 1.5 * (longitude - 75.85)
    + rng.normal(0, 0.3, len(latitude))
)

precip = np.clip(
    0.2 + 0.1 * np.sin(latitude * 10)
    + rng.normal(0, 0.02, len(latitude)),
    0,
    None
)

# Keep the features consistent with model training
tmax = temperature + 5
tmin = temperature - 5

# Representative prediction date and hour
day = np.full(len(latitude), 15)
month = np.full(len(latitude), 6)
hour = np.zeros(len(latitude), dtype=int)

# Build inference dataframe
grid_df = pd.DataFrame({
    "latitude": latitude,
    "longitude": longitude,
    "temperature": temperature,
    "precip": precip,
    "tmax": tmax,
    "tmin": tmin,
    "day": day,
    "month": month,
    "hour": hour
})

# Save inference grids for both configured years
for year in (2019, 2022):
    output_path = os.path.join(
        OUTPUT_DIR,
        f"Jaipur_{year}_inference.csv"
    )
    grid_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({len(grid_df)} rows)")

print("\nInference grid generation complete.")
print(grid_df.head())
```
