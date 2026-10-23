from glob import glob
import pandas as pd

daily_counts = []

for file in sorted(glob("data/raw/bluebikes/trips/*.csv")):
    trips = pd.read_csv(file, dtype=str)
    if "ride_id" in trips:
        trips = trips.drop_duplicates(subset="ride_id")
    else:
        trips = trips.drop_duplicates()
    trips = trips.rename(columns={"started_at": "starttime", "ended_at": "stoptime"})
    for column in ["starttime", "stoptime"]:
        trips[column] = pd.to_datetime(trips[column], format="mixed", errors="coerce")
    trips = trips[trips["stoptime"] > trips["starttime"]]
    daily_counts.append(trips.groupby(trips["starttime"].dt.strftime("%Y-%m-%d")).size())

daily = pd.concat(daily_counts).groupby(level=0).sum()
daily = daily.rename_axis("date").reset_index(name="ride_count")
daily.to_csv("data/processed/processed_ridership.csv", index=False)

weather = pd.read_csv("data/raw/weather/boston_weather_2022_2025.csv", skiprows=3)
weather.columns = ["date", "temp_mean", "temp_min", "temp_max", "precip_mm", "rain_mm", "snow_cm", "wind_kmh"]
merged = weather.merge(daily, on="date", how="left", validate="one_to_one")
merged = merged.replace(r"^\s*$", pd.NA, regex=True).dropna()
merged["ride_count"] = merged["ride_count"].astype(int)
merged.to_csv("data/processed/ridership_weather.csv", index=False)
