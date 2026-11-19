import requests

latitude = 24.478972
longitude = 92.682861
date = "2016-05-17"

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": date,
    "end_date": date,
    "hourly": "soil_moisture_0_to_7cm",
    "timezone": "auto"
}

response = requests.get(url, params=params, timeout=30)

print("Status:", response.status_code)

data = response.json()

print("\nDates/times:")
print(data["hourly"]["time"])

print("\nSoil moisture:")
print(data["hourly"]["soil_moisture_0_to_7cm"])

values = [
    x for x in data["hourly"]["soil_moisture_0_to_7cm"]
    if x is not None
]

print("\nAverage soil moisture:", sum(values) / len(values))