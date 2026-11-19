import pandas as pd
import requests
import time
import os

# --------------------------------------------------
# 1. Load preprocessed data
# --------------------------------------------------

input_file = "preprocessed_landslide_data.csv"
output_file = "rainfall_dataset.csv"

df = pd.read_csv(input_file)

# Convert date and coordinates
df["landslide_date"] = pd.to_datetime(
    df["landslide_date"],
    errors="coerce"
)

df["Latitude"] = pd.to_numeric(
    df["Latitude"],
    errors="coerce"
)

df["Longitude"] = pd.to_numeric(
    df["Longitude"],
    errors="coerce"
)

# --------------------------------------------------
# 2. Create rainfall columns if they don't exist
# --------------------------------------------------

if "rainfall_24h" not in df.columns:
    df["rainfall_24h"] = pd.NA

if "rainfall_7d" not in df.columns:
    df["rainfall_7d"] = pd.NA

# --------------------------------------------------
# 3. If previous output exists, load saved progress
# --------------------------------------------------

if os.path.exists(output_file):

    print("Previous rainfall dataset found.")
    print("Loading saved progress...")

    old_df = pd.read_csv(output_file)

    old_df["landslide_date"] = pd.to_datetime(
        old_df["landslide_date"],
        errors="coerce"
    )

    # Copy previous rainfall values
    if "rainfall_24h" in old_df.columns:
        df["rainfall_24h"] = old_df["rainfall_24h"]

    if "rainfall_7d" in old_df.columns:
        df["rainfall_7d"] = old_df["rainfall_7d"]

# --------------------------------------------------
# 4. Select only records with date + coordinates
# --------------------------------------------------

valid = df[
    df["landslide_date"].notna()
    & df["Latitude"].notna()
    & df["Longitude"].notna()
].copy()

print("\nTotal records:", len(df))
print("Records with usable date:", len(valid))

# --------------------------------------------------
# 5. Process rainfall
# --------------------------------------------------

processed = 0

for index, row in valid.iterrows():

    # Skip if rainfall was already collected
    if (
        pd.notna(df.at[index, "rainfall_24h"])
        and pd.notna(df.at[index, "rainfall_7d"])
    ):
        continue

    latitude = row["Latitude"]
    longitude = row["Longitude"]
    date = row["landslide_date"]

    # 7-day period including landslide date
    start_date = (
        date - pd.Timedelta(days=6)
    ).strftime("%Y-%m-%d")

    end_date = date.strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": "auto"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=30
        )

        if response.status_code == 200:

            data = response.json()

            rainfall = data["daily"]["precipitation_sum"]

            # Rainfall on landslide date
            rainfall_24h = rainfall[-1]

            # Total rainfall over 7 days
            rainfall_7d = sum(
                value
                for value in rainfall
                if value is not None
            )

            df.at[index, "rainfall_24h"] = rainfall_24h
            df.at[index, "rainfall_7d"] = rainfall_7d

            processed += 1

            print(
                f"{processed} | "
                f"Date: {date.strftime('%d-%b-%Y')} | "
                f"24h: {rainfall_24h} mm | "
                f"7d: {rainfall_7d} mm"
            )

        else:

            print(
                f"API error {response.status_code} "
                f"for row {index}"
            )

    except Exception as e:

        print(
            f"Error for row {index}: {e}"
        )

    # --------------------------------------------------
    # 6. Save progress every 50 records
    # --------------------------------------------------

    if processed > 0 and processed % 50 == 0:

        df.to_csv(
            output_file,
            index=False
        )

        print(
            f"\nProgress saved: {processed} new records\n"
        )

    # Small delay
    time.sleep(0.2)


# --------------------------------------------------
# 7. Final save
# --------------------------------------------------

df.to_csv(
    output_file,
    index=False
)

print("\n================================")
print("RAINfall PROCESSING COMPLETE")
print("================================")

print("Output file:", output_file)

print(
    "Records with rainfall:",
    df["rainfall_24h"].notna().sum()
)

print(
    "Records without rainfall:",
    df["rainfall_24h"].isna().sum()
)