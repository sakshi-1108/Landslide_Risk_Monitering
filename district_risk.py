import pandas as pd

# Load GSI landslide data
df = pd.read_csv("ner_landslide_data.csv")

# Remove rows without district/state
df = df.dropna(subset=["State", "District"]).copy()

# Count historical landslides per district
district_risk = (
    df.groupby(["State", "District"])
    .size()
    .reset_index(name="historical_landslides")
)

# Classify historical density
def classify_density(count):

    if count >= 20:
        return "High"

    elif count >= 10:
        return "Moderate"

    else:
        return "Low"


district_risk["historical_risk"] = (
    district_risk["historical_landslides"]
    .apply(classify_density)
)

# Sort highest density first
district_risk = district_risk.sort_values(
    "historical_landslides",
    ascending=False
)

# Save result
district_risk.to_csv(
    "district_risk.csv",
    index=False
)

print("District-level historical risk created successfully.")

print("\nTotal districts:", len(district_risk))

print("\nRisk distribution:")
print(district_risk["historical_risk"].value_counts())

print("\nTop 15 districts:")
print(district_risk.head(15).to_string(index=False))