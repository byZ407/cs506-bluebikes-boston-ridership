import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from helpers import inputs, scores

data = pd.read_csv("data/processed/daily_features.csv", parse_dates=["date"])
train = data[data["split"] != "test"]
test = data[data["split"] == "test"]
experiments = pd.read_csv("modeling/results/experiments_results.csv")
selected = experiments.sort_values("MAE").groupby("model", sort=False).head(1)
selected.to_csv("modeling/results/model_comparison.csv", index=False)
models = [("Mean baseline", "basic", DummyRegressor())]

for row in selected.itertuples():
    if row.model == "Linear Regression":
        model = LinearRegression()
    elif row.model == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=int(row.n_estimators),
            max_depth=None if pd.isna(row.max_depth) else int(row.max_depth),
            min_samples_leaf=int(row.min_samples_leaf), random_state=42, n_jobs=-1,
        )
    else:
        model = XGBRegressor(
            n_estimators=int(row.n_estimators), learning_rate=row.learning_rate,
            max_depth=int(row.max_depth), objective="reg:squarederror",
            random_state=42, n_jobs=-1,
        )
    models.append((row.model, row.features, model))

results = []
predictions = test[["date", "ride_count"]].copy()

for name, feature_set, model in models:
    model.fit(inputs(train, feature_set), train["ride_count"])
    predicted = model.predict(inputs(test, feature_set))
    predictions[name] = predicted
    results.append({
        "model": name,
        "features": feature_set,
        "evaluation": "2025 follow-up",
        **scores(test["ride_count"], predicted),
    })

pd.DataFrame(results).to_csv("modeling/results/test_results.csv", index=False)
predictions.to_csv("modeling/results/predictions.csv", index=False)

plt.figure(figsize=(11, 5))
plt.plot(test["date"], test["ride_count"], label="Actual", color="black", linewidth=1)
for name in ["Linear Regression", "Random Forest", "XGBoost"]:
    plt.plot(test["date"], predictions[name], label=name, linewidth=1, alpha=0.7)
plt.title("Model Evaluation Using 2025 Daily Bluebikes Ridership")
plt.xlabel("Date")
plt.ylabel("Daily rides")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("visualization/evaluation.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 6))
for name in ["Linear Regression", "Random Forest", "XGBoost"]:
    plt.scatter(test["ride_count"], predictions[name], label=name, s=15, alpha=0.5)
limits = [min(test["ride_count"].min(), predictions[["Linear Regression", "Random Forest", "XGBoost"]].min().min()),
          max(test["ride_count"].max(), predictions[["Linear Regression", "Random Forest", "XGBoost"]].max().max())]
plt.plot(limits, limits, "k--", label="Perfect prediction")
plt.title("Actual Ridership vs. Predicted Ridership")
plt.xlabel("Actual daily rides")
plt.ylabel("Predicted daily rides")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("visualization/actual_vs_predicted_ridership.png", dpi=200)
plt.close()
