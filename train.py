"""Clean the Pune housing data, compare models, and save the best one."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path("data/pune_house_data.csv")
MODEL_DIR = Path("model")
FEATURES = ["total_sqft", "bhk", "bath", "balcony", "location"]


def to_sqft(value):
    """'1200' -> 1200.0, '2100 - 2850' -> 2475.0, '34.46Sq. Meter' -> NaN."""
    s = str(value).strip()
    try:
        if "-" in s:
            a, b = s.split("-")
            return (float(a) + float(b)) / 2
        return float(s)
    except ValueError:
        return np.nan


def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    loc_col = "site_location" if "site_location" in df.columns else "location"
    df = df.rename(columns={loc_col: "location"})
    print(f"Raw rows: {len(df)}")

    df["total_sqft"] = df["total_sqft"].apply(to_sqft)
    df["bhk"] = df["size"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    df["balcony"] = df["balcony"].fillna(df["balcony"].median())
    df["location"] = df["location"].astype(str).str.strip()
    df = df.dropna(subset=["total_sqft", "bhk", "bath", "price", "location"])

    # Rare locations -> "other" so the model does not memorise tiny groups
    counts = df["location"].value_counts()
    df.loc[df["location"].isin(counts[counts <= 10].index), "location"] = "other"

    # Remove unrealistic rows
    df = df[df["total_sqft"] / df["bhk"] >= 300]      # < 300 sqft per bedroom
    df = df[df["bath"] <= df["bhk"] + 2]               # too many bathrooms

    # Remove price-per-sqft outliers within each location
    df["ppsf"] = df["price"] * 100000 / df["total_sqft"]
    g = df.groupby("location")["ppsf"]
    mean, std, size = g.transform("mean"), g.transform("std"), g.transform("size")
    keep = (size < 10) | ((df["ppsf"] > mean - std) & (df["ppsf"] <= mean + std))
    df = df[keep]

    print(f"Clean rows: {len(df)}")
    return df[FEATURES + ["price"]].reset_index(drop=True)


def make_pipeline(regressor):
    pre = ColumnTransformer(
        [("loc", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["location"])],
        remainder="passthrough",
    )
    # Predict log(price): prices are skewed and errors are proportional
    return TransformedTargetRegressor(
        regressor=Pipeline([("pre", pre), ("model", regressor)]),
        func=np.log1p, inverse_func=np.expm1,
    )


def main():
    df = load_and_clean(DATA_PATH)
    X, y = df[FEATURES], df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "GradientBoosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=42),
    }

    results, best_name, best_cv = {}, None, -np.inf
    for name, reg in candidates.items():
        pipe = make_pipeline(reg)
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "cv_r2": round(float(cv), 4),
            "test_r2": round(float(r2_score(y_test, pred)), 4),
            "test_mae_lakhs": round(float(mean_absolute_error(y_test, pred)), 2),
        }
        print(f"{name:18s} {results[name]}")
        if cv > best_cv:
            best_name, best_cv = name, cv

    print(f"\nBest model: {best_name}")
    final = make_pipeline(candidates[best_name]).fit(X, y)  # refit on all data

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_DIR / "model.joblib", compress=("xz", 3))
    meta = {
        "best_model": best_name,
        "results": results,
        "locations": sorted(df["location"].unique().tolist()),
        "sqft_range": [int(df["total_sqft"].min()), int(df["total_sqft"].max())],
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Saved model/model.joblib and model/meta.json")


if __name__ == "__main__":
    main()
