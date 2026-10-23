from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Plot temperature against daily rides.
folder = Path(__file__).resolve().parent
data = pd.read_csv(folder.parent / "data/processed/ridership_weather.csv")
plt.figure(figsize=(8, 5))
plt.scatter(data["temp_mean"], data["ride_count"], alpha=0.5, s=15)
plt.xlabel("Mean daily temperature (°C)")
plt.ylabel("Daily rides")
plt.title("Bluebikes Ridership vs. Temperature (2022–2025)")
plt.tight_layout()
plt.savefig(folder / "temperature_ridership.png", dpi=300)
plt.close()
