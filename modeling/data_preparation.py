import os
import pandas as pd

data = pd.read_csv("data/processed/ridership_weather.csv", parse_dates=["date"])
data = data.sort_values("date")
data["month"] = data["date"].dt.month
data["day_of_week"] = data["date"].dt.dayofweek
data["temp_squared"] = data["temp_mean"] ** 2
data["years_since_2022"] = (data["date"] - pd.Timestamp("2022-01-01")).dt.days / 365.25
data["split"] = "train"
data.loc[data["date"].dt.year == 2024, "split"] = "validation"
data.loc[data["date"].dt.year == 2025, "split"] = "test"
data.to_csv("data/processed/daily_features.csv", index=False)
os.makedirs("modeling/results", exist_ok=True)
