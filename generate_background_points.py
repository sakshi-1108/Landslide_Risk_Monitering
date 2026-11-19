import pandas as pd
import numpy as np

# Load real GSI landslide data
landslides = pd.read_csv("ner_landslide_data.csv")

landslides["Latitude"] = pd.to_numeric(
    landslides["Latitude"], errors="coerce"
)

landslides["Longitude"] = pd.to_numeric(
    landslides["Longitude"], errors="coerce"
)

landslides = landslides.dropna(
    subset=["Latitude", "Longitude"]
)

# Use 3000 real landslide samples
positive = landslides.sample(
    n=3000,
    random_state=42
).copy()

positive["historical_landslide"] = 1

positive = positive[
    ["Latitude", "Longitude", "State", "District",
     "Material Involved", "Movement\nType",
     "History", "historical_landslide"]
]

# Generate 3000 candidate background points
np.random.seed(42)

negative = pd.DataFrame({
    "Latitude": np.random.uniform(21.5, 29.5, 3000),
    "Longitude": np.random.uniform(88.0, 97.5, 3000),
    "State": "Background",
    "District": "Background",
    "Material Involved": "None",
    "Movement\nType": "None",
    "History": "No GSI event at sampled point",
    "historical_landslide": 0
})

# Combine
rf_data = pd.concat(
    [positive, negative],
    ignore_index=True
)

# Shuffle
rf_data = rf_data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save
rf_data.to_csv(
    "rf_training_base.csv",
    index=False
)

print("RF training base created successfully.")
print("Total samples:", len(rf_data))

print("\nClass distribution:")
print(
    rf_data["historical_landslide"]
    .value_counts()
)

print("\nFirst 5 rows:")
print(rf_data.head())