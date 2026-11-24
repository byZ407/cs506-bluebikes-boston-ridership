import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from helpers import FEATURES, inputs, scores

data = pd.read_csv("data/processed/daily_features.csv")
train = data[data["split"] == "train"]
validation = data[data["split"] == "validation"]
results = []

for feature_set in FEATURES:
    X_train, X_validation = inputs(train, feature_set), inputs(validation, feature_set)
    models = [("Linear Regression", LinearRegression())]
    for depth in [5, 10, 15, None]:
        for leaf in [1, 3, 5]:
            models.append(("Random Forest", RandomForestRegressor(
                n_estimators=100, max_depth=depth, min_samples_leaf=leaf,
                random_state=42, n_jobs=-1,
            )))
    for depth in [2, 3]:
        models.append(("XGBoost", XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=depth,
            objective="reg:squarederror", random_state=42, n_jobs=-1,
        )))
    for name, model in models:
        model.fit(X_train, train["ride_count"])
        parameters = model.get_params()
        results.append({
            "model": name,
            "features": feature_set,
            "n_estimators": parameters.get("n_estimators"),
            "learning_rate": parameters.get("learning_rate"),
            "max_depth": parameters.get("max_depth"),
            "min_samples_leaf": parameters.get("min_samples_leaf"),
            "train_MAE": scores(train["ride_count"], model.predict(X_train))["MAE"],
            **scores(validation["ride_count"], model.predict(X_validation)),
        })

results = pd.DataFrame(results).sort_values("MAE")
results.to_csv("modeling/results/experiments_results.csv", index=False)
comparison = results[
    (results["model"] == "Linear Regression")
    | ((results["model"] == "Random Forest") & (results["max_depth"] == 10) & (results["min_samples_leaf"] == 5))
    | ((results["model"] == "XGBoost") & (results["max_depth"] == 3))
]
comparison.to_csv("modeling/results/feature_comparison.csv", index=False)
results.groupby("model", sort=False).head(1).to_csv("modeling/results/validation_comparison.csv", index=False)
