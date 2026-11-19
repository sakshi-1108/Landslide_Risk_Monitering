from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
from pathlib import Path

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent


def read_csv_safe(filename):
    path = BASE_DIR / filename
    if not path.exists():
        print(f"Optional data file not found: {filename}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Could not read {filename}: {exc}")
        return pd.DataFrame()


def make_match_key(frame):
    dates = pd.to_datetime(frame["landslide_date"], errors="coerce")
    return (
        pd.to_numeric(frame["Latitude"], errors="coerce").round(6).astype("string").fillna("")
        + "_"
        + pd.to_numeric(frame["Longitude"], errors="coerce").round(6).astype("string").fillna("")
        + "_"
        + dates.dt.strftime("%Y-%m-%d").fillna("")
    )


df = read_csv_safe("ner_landslide_data.csv")
if df.empty:
    raise RuntimeError("ner_landslide_data.csv is missing or empty.")

df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
df = df.dropna(subset=["Latitude", "Longitude"]).copy()
df["landslide_date"] = pd.to_datetime(df["History"], errors="coerce").dt.strftime("%Y-%m-%d")
df["match_key"] = make_match_key(df)

for col in ["State", "District", "Material Involved", "Movement\nType", "History"]:
    if col in df.columns:
        df[col] = df[col].fillna("Not specified").astype(str).str.strip()

df["rainfall_24h"] = pd.NA
df["rainfall_7d"] = pd.NA
rain = read_csv_safe("rainfall_dataset.csv")
rain_required = {"Latitude", "Longitude", "landslide_date", "rainfall_24h", "rainfall_7d"}
if rain_required.issubset(rain.columns):
    rain["match_key"] = make_match_key(rain)
    for col in ["rainfall_24h", "rainfall_7d"]:
        rain[col] = pd.to_numeric(rain[col], errors="coerce")
    rain = rain.drop_duplicates("match_key", keep="last")
    df = df.drop(columns=["rainfall_24h", "rainfall_7d"]).merge(
        rain[["match_key", "rainfall_24h", "rainfall_7d"]],
        on="match_key", how="left", validate="many_to_one"
    )
else:
    print("Rainfall dataset missing required columns.")

df["soil_moisture"] = pd.NA
soil = read_csv_safe("soil_moisture_dataset.csv")
soil_required = {"Latitude", "Longitude", "landslide_date", "soil_moisture"}
if soil_required.issubset(soil.columns):
    soil["match_key"] = make_match_key(soil)
    soil["soil_moisture"] = pd.to_numeric(soil["soil_moisture"], errors="coerce")
    soil = soil.drop_duplicates("match_key", keep="last")
    df = df.drop(columns=["soil_moisture"]).merge(
        soil[["match_key", "soil_moisture"]],
        on="match_key", how="left", validate="many_to_one"
    )
else:
    print("Soil moisture dataset missing required columns.")

state_counts = df["State"].value_counts().sort_values(ascending=False).to_dict()
district_counts = df.groupby("State")["District"].nunique().to_dict()
top_districts = df["District"].value_counts().head(10).to_dict()
movement_counts = df["Movement\nType"].value_counts().head(8).to_dict() if "Movement\nType" in df else {}
material_counts = df["Material Involved"].value_counts().head(8).to_dict() if "Material Involved" in df else {}
history_counts = df["History"].value_counts().head(8).to_dict() if "History" in df else {}

risk = read_csv_safe("district_risk.csv")
if {"historical_risk", "State", "District", "historical_landslides"}.issubset(risk.columns):
    risk_counts = risk["historical_risk"].value_counts().reindex(["High", "Moderate", "Low"], fill_value=0).to_dict()
    top_risk_districts = risk[["State", "District", "historical_landslides", "historical_risk"]].head(10).to_dict(orient="records")
else:
    risk_counts = {"High": 0, "Moderate": 0, "Low": 0}
    top_risk_districts = []
sar_features = {}
try:
    with (BASE_DIR / "sar_features.json").open("r", encoding="utf-8") as f:
        sar_features = json.load(f)
except Exception as exc:
    print("SAR feature file could not be loaded:", exc)
    sar_features = {}


# LIVE 24-HOUR AI LANDSLIDE ALERT (see forecast.py)
from forecast import register_forecast
register_forecast(app, df)

# RL POLICY LAB (experimental, see rl_agent.py)
from rl_agent import register_rl
register_rl(app)

# HOME PAGE
@app.route("/")
def home():
    return render_template(
        "index.html",
        total_landslides=len(df),
        states=df["State"].nunique(),
        sar=sar_features,
        state_counts=state_counts,
        district_counts=district_counts,
        top_districts=top_districts,
        movement_counts=movement_counts,
        material_counts=material_counts,
        history_counts=history_counts,
        risk_counts=risk_counts,
        top_risk_districts=top_risk_districts
    )


# LANDSLIDE DATA API
@app.route("/api/landslides")
def landslides():
    records = []

    for _, row in df.iterrows():
        records.append({
            "latitude": float(row["Latitude"]),
            "longitude": float(row["Longitude"]),
            "state": str(row["State"]),
            "district": str(row["District"]),
            "landslide_date": (
                None if pd.isna(row["landslide_date"])
                else str(row["landslide_date"])
            ),
            "movement_type": str(row.get("Movement\nType", "")),
            "material": str(row.get("Material Involved", "")),
            "rainfall_24h": (
                None if pd.isna(row["rainfall_24h"])
                else float(row["rainfall_24h"])
            ),
            "rainfall_7d": (
                None if pd.isna(row["rainfall_7d"])
                else float(row["rainfall_7d"])
            ),
            "soil_moisture": (
                None if pd.isna(row["soil_moisture"])
                else float(row["soil_moisture"])
            )
        })

    return jsonify(records)


# ANALYTICS API
@app.route("/api/analytics")
def analytics():
    return jsonify({
        "total": int(len(df)),
        "states": state_counts,
        "districts": district_counts,
        "top_districts": top_districts,
        "movement_types": movement_counts,
        "materials": material_counts,
        "history": history_counts,
        "risk_counts": risk_counts,
        "top_risk_districts": top_risk_districts
    })


# SATELLITE DATA API
@app.route("/api/sar")
def sar():
    return jsonify(sar_features)


# AI RISK EXPLANATION API
@app.route("/api/risk-explanation")
def risk_explanation():
    selected_state = request.args.get("state", "").strip()
    selected_district = request.args.get("district", "").strip()

    if not selected_state or not selected_district:
        return jsonify({
            "error": "Please select both a state and a district."
        }), 400

    # Match the selected state and district against the inventory
    district_data = df[
        (df["State"].str.casefold() == selected_state.casefold()) &
        (df["District"].str.casefold() == selected_district.casefold())
    ]

    if district_data.empty:
        return jsonify({
            "error": "No landslide records found for this district."
        }), 404

    # Historical inventory evidence
    historical_count = int(len(district_data))

    # Count available environmental observations, excluding missing values
    rainfall24 = pd.to_numeric(
        district_data["rainfall_24h"], errors="coerce"
    ).dropna()

    rainfall7 = pd.to_numeric(
        district_data["rainfall_7d"], errors="coerce"
    ).dropna()

    soil_values = pd.to_numeric(
        district_data["soil_moisture"], errors="coerce"
    ).dropna()

    # Historical category based on the project's existing thresholds
    if historical_count >= 20:
        historical_risk = "High"
    elif historical_count >= 10:
        historical_risk = "Moderate"
    else:
        historical_risk = "Low"

    # Build explanations strictly from available evidence
    explanations = [
        f"The inventory contains {historical_count} landslide records "
        f"for {selected_district}, {selected_state}."
    ]

    explanations.append(
        f"The historical category is {historical_risk}, "
        "based on the project's record-count thresholds."
    )

    if len(rainfall24):
        explanations.append(
            f"24-hour rainfall data are available for "
            f"{len(rainfall24)} records. The mean of available values is "
            f"{rainfall24.mean():.2f} mm."
        )
    else:
        explanations.append(
            "No matched 24-hour rainfall observations are available "
            "for this district's inventory records."
        )

    if len(rainfall7):
        explanations.append(
            f"7-day rainfall data are available for {len(rainfall7)} "
            f"records. The mean of available values is "
            f"{rainfall7.mean():.2f} mm."
        )
    else:
        explanations.append(
            "No matched 7-day rainfall observations are available "
            "for this district's inventory records."
        )

    if len(soil_values):
        explanations.append(
            f"Soil-moisture data are available for {len(soil_values)} "
            "records; the mean available value is "
            f"{soil_values.mean():.2f} in the source dataset's units."
        )
    else:
        explanations.append(
            "No matched soil-moisture observations are available "
            "for this district's inventory records."
        )

    return jsonify({
        "state": selected_state,
        "district": selected_district,
        "historical_landslide_records": historical_count,
        "historical_risk": historical_risk,
        "environmental_data": {
            "rainfall_24h_count": int(len(rainfall24)),
            "rainfall_24h_mean_mm": (
                round(float(rainfall24.mean()), 2)
                if len(rainfall24) else None
            ),
            "rainfall_7d_count": int(len(rainfall7)),
            "rainfall_7d_mean_mm": (
                round(float(rainfall7.mean()), 2)
                if len(rainfall7) else None
            ),
            "soil_moisture_count": int(len(soil_values)),
            "soil_moisture_mean": (
                round(float(soil_values.mean()), 2)
                if len(soil_values) else None
            )
        },
        "explanations": explanations,
        "limitations": (
            "This is a historical evidence summary, not a real-time "
            "landslide forecast. Missing observations are excluded. "
            "Historical record counts do not establish the probability "
            "of a future landslide."
        )
    })


# START SERVER
if __name__ == "__main__":
    print("Starting TERRANEX...")
    app.run(debug=True)