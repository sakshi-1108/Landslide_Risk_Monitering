import pandas as pd
import requests
import os
import time

INPUT_FILE = "rainfall_dataset.csv"
OUTPUT_FILE = "soil_moisture_dataset.csv"

# Load rainfall dataset
df = pd.read_csv(INPUT_FILE)

# Make sure date is in proper format
df["landslide_date"] = pd.to_datetime(
    df["landslide_date"],
    errors="coerce"
)

# Add soil moisture column if it doesn't exist
if "soil_moisture" not in df.columns:
    df["soil_moisture"] = None

# If previous progress exists, load it
if os.path.exists(OUTPUT_FILE):
    old_df = pd.read_csv(OUTPUT_FILE)

    if "soil_moisture" in old_df.columns:
        df["soil_moisture"] = old_df["soil_moisture"]

print("================================")
print("SOIL MOISTURE PROCESSING")
print("================================")

processed = 0
skipped = 0

for i in range(len(df)):

    # Skip if soil moisture already exists
    if pd.notna(df.loc[i, "soil_moisture"]):
        skipped += 1
        continue

    latitude = df.loc[i, "Latitude"]
    longitude = df.loc[i, "Longitude"]
    date = df.loc[i, "landslide_date"]

    # Skip missing information
    if pd.isna(latitude) or pd.isna(longitude) or pd.isna(date):
        continue

    date_string = date.strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date_string,
        "end_date": date_string,
        "hourly": "soil_moisture_0_to_7cm",
        "timezone": "auto"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=30
        )

        if response.status_code != 200:
            print(
                f"API error at row {i}: "
                f"{response.status_code}"
            )
            continue

        data = response.json()

        values = data["hourly"]["soil_moisture_0_to_7cm"]

        # Remove missing values
        values = [
            value for value in values
            if value is not None
        ]

        if len(values) > 0:

            # Daily average
            average_soil_moisture = sum(values) / len(values)

            df.loc[i, "soil_moisture"] = average_soil_moisture

            processed += 1

            print(
                f"Row {i} | "
                f"Date {date_string} | "
                f"Soil moisture: "
                f"{average_soil_moisture:.3f}"
            )

        # Save progress every 25 records
        if processed > 0 and processed % 25 == 0:
            df.to_csv(
                OUTPUT_FILE,
                index=False
            )
            print("Progress saved.")

        # Small delay to avoid sending requests too quickly
        time.sleep(0.2)

    except Exception as e:

        print(
            f"Error at row {i}: {e}"
        )

# Final save
df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n================================")
print("SOIL MOISTURE COMPLETE")
print("================================")

print("Output file:", OUTPUT_FILE)

print(
    "Records with soil moisture:",
    df["soil_moisture"].notna().sum()
)

print(
    "Records without soil moisture:",
    df["soil_moisture"].isna().sum()
)