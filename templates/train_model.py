import pandas as pd

print("THIS IS MY CURRENT TRAIN_MODEL.PY")

# Load actual GSI landslide data
df = pd.read_csv("ner_landslide_data.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nStates:")
print(df["State"].value_counts())