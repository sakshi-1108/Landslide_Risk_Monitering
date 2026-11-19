import pandas as pd

# Load GSI landslide data
df = pd.read_csv("ner_landslide_data.csv")

# Convert coordinates to numbers
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

df = df.dropna(subset=["Latitude", "Longitude"])

# Sentinel-1 scene approximate geographic coverage
MIN_LAT = 26.703863
MAX_LAT = 28.640021
MIN_LON = 86.175363
MAX_LON = 88.997964

# Find points inside the satellite coverage
covered = df[
    (df["Latitude"] >= MIN_LAT) &
    (df["Latitude"] <= MAX_LAT) &
    (df["Longitude"] >= MIN_LON) &
    (df["Longitude"] <= MAX_LON)
].copy()

print("Total GSI landslide points:", len(df))
print("Points inside Sentinel-1 scene:", len(covered))
print("Points outside Sentinel-1 scene:", len(df) - len(covered))

print("\nStates covered:")
print(covered["State"].value_counts())

# Save the overlapping points
covered.to_csv("sar_covered_landslides.csv", index=False)

print("\nSaved as: sar_covered_landslides.csv")