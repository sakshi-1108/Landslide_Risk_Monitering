import requests

latitude = 24.478972
longitude = 92.682861

start_date = "2016-05-11"
end_date = "2016-05-17"

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": start_date,
    "end_date": end_date,
    "daily": "precipitation_sum",
    "timezone": "auto"
}

response = requests.get(url, params=params)

print("Status:", response.status_code)

data = response.json()

print("\nDates:")
print(data["daily"]["time"])

print("\nDaily rainfall (mm):")
print(data["daily"]["precipitation_sum"])

rainfall_24h = data["daily"]["precipitation_sum"][-1]
rainfall_7d = sum(
    x for x in data["daily"]["precipitation_sum"]
    if x is not None
)

print("\nRainfall on landslide date:", rainfall_24h, "mm")
print("Rainfall during previous 7 days:", rainfall_7d, "mm")