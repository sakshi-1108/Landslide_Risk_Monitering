import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("rf_training_base.csv")

print("Dataset loaded.")
print("Total rows:", len(df))


# --------------------------------------------------
# 2. SELECT FEATURES
# --------------------------------------------------

features = [
    "Latitude",
    "Longitude"
]

target = "historical_landslide"

X = df[features]
y = df[target]


# --------------------------------------------------
# 3. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 4. RANDOM FOREST MODEL
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# --------------------------------------------------
# 5. PREDICTION
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 6. MODEL EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# 7. FEATURE IMPORTANCE
# --------------------------------------------------

print("\nFeature Importance:")

for feature, importance in zip(
    features,
    model.feature_importances_
):
    print(
        feature,
        ":",
        round(importance, 4)
    )


# --------------------------------------------------
# 8. SAVE MODEL
# --------------------------------------------------

import joblib

joblib.dump(
    model,
    "random_forest_model.pkl"
)

print("\nModel saved as random_forest_model.pkl")