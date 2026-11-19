import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load NER landslide dataset
df = pd.read_csv("ner_landslide_data.csv")

print("Dataset loaded!")
print("Total records:", len(df))

# --------------------------------------------------
# Create target
# --------------------------------------------------
# These are real landslide inventory records,
# so target = 1 means a recorded landslide location.

df["risk_label"] = 1

# --------------------------------------------------
# Select features currently available
# --------------------------------------------------

features = [
    "Latitude",
    "Longitude"
]

X = df[features]
y = df["risk_label"]

# Remove missing values
data = pd.concat([X, y], axis=1)
data = data.dropna()

X = data[features]
y = data["risk_label"]

print("Records used for model:", len(X))

# --------------------------------------------------
# Temporary prototype model
# --------------------------------------------------

# NOTE:
# The current inventory contains only recorded landslide
# locations. Therefore this is an inventory-location
# prototype, NOT the final landslide risk model.

# Create a small background class using locations
# outside the recorded landslide points.
background = X.sample(
    n=min(len(X), 5000),
    random_state=42
).copy()

background["risk_label"] = 0

# Combine landslide and background samples
landslide_data = X.copy()
landslide_data["risk_label"] = 1

model_data = pd.concat(
    [landslide_data, background],
    ignore_index=True
)

X_final = model_data[features]
y_final = model_data["risk_label"]

# --------------------------------------------------
# Train Random Forest
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y_final,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("\nRandom Forest trained successfully!")
print("Accuracy:", accuracy)

# --------------------------------------------------
# Save model
# --------------------------------------------------

with open("landslide_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved as landslide_model.pkl")