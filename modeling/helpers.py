import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

FEATURES = {
    "basic": ["temp_mean", "precip_mm", "snow_cm", "wind_kmh", "month", "day_of_week"],
    "expanded": [
        "temp_mean", "precip_mm", "snow_cm", "wind_kmh", "month", "day_of_week",
        "temp_squared", "years_since_2022",
    ],
}


def inputs(data, feature_set):
    X = data[FEATURES[feature_set]].copy()
    X["month"] = pd.Categorical(X["month"], categories=range(1, 13))
    X["day_of_week"] = pd.Categorical(X["day_of_week"], categories=range(7))
    return pd.get_dummies(X, columns=["month", "day_of_week"], drop_first=True, dtype=int)


def scores(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": root_mean_squared_error(actual, predicted),
        "R2": r2_score(actual, predicted),
    }
