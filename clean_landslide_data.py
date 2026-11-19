import pandas as pd

# Load raw extracted data
df = pd.read_csv("landslide_inventory_raw.csv", header=None)

print("Original shape:", df.shape)

# Row 2 contains the actual column names
df.columns = df.iloc[2]

# Remove first 3 rows
df = df.iloc[3:].reset_index(drop=True)

print("\nAfter removing extra rows:")
print("Shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 actual records:")
print(df.head())

# Save cleaned dataset
df.to_csv("landslide_inventory_clean.csv", index=False)

print("\nCleaned dataset saved successfully!")