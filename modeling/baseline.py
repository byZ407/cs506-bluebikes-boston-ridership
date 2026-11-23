import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from helpers import inputs, scores

data = pd.read_csv("data/processed/daily_features.csv")
train = data[data["split"] == "train"]
validation = data[data["split"] == "validation"]
X_train, X_validation = inputs(train, "basic"), inputs(validation, "basic")
results = []

for name, model in [("Mean baseline", DummyRegressor()), ("Linear Regression", LinearRegression())]:
    model.fit(X_train, train["ride_count"])
    results.append({
        "model": name,
        "features": "basic",
        "train_MAE": scores(train["ride_count"], model.predict(X_train))["MAE"],
        **scores(validation["ride_count"], model.predict(X_validation)),
    })

pd.DataFrame(results).to_csv("modeling/results/baseline_results.csv", index=False)
