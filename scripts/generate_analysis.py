#!/usr/bin/env python3
"""
End-to-end analysis script for the ICU mortality tutorial.

This script:
1. Loads the raw medical dataset and builds daily-level metrics.
2. Trains a random forest model to forecast daily mortality rates.
3. Outputs evaluation artifacts and visual assets consumed by the tutorial.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ADMISSIONS_PER_DAY = 500
TRAIN_TEST_SPLIT_DAYS = 30
RANDOM_STATE = 42
START_DATE = "2015-01-01"


@dataclass
class EvaluationResult:
    mae: float
    rmse: float
    r2: float
    current_day: str
    current_actual: float
    current_prediction: float


def load_data(path: Path) -> pd.DataFrame:
    """Load the raw feature CSV and coerce boolean columns to integers."""
    df = pd.read_csv(path)
    bool_columns = df.select_dtypes(include=["bool"]).columns
    if len(bool_columns):
        df[bool_columns] = df[bool_columns].astype(int)
    return df


def build_daily_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate admission-level records into daily metrics."""
    ordered = df.sort_values("hadm_id").reset_index(drop=True)
    ordered["day_index"] = (ordered.index // ADMISSIONS_PER_DAY).astype(int)

    daily = (
        ordered.groupby("day_index")
        .agg(
            daily_admissions=("hadm_id", "count"),
            unique_patients=("subject_id", "nunique"),
            mortality_rate=("hospital_expire_flag", "mean"),
            died_in_hospital_rate=("died_in_hospital", "mean"),
            los_hours_mean=("los_hours", "mean"),
            los_hours_median=("los_hours", "median"),
            home_discharge_rate=("discharge_location_HOME", "mean"),
            home_health_rate=("discharge_location_HOME HEALTH CARE", "mean"),
            transfer_from_hosp_rate=(
                "admission_location_TRANSFER FROM HOSPITAL",
                "mean",
            ),
            ed_stay_minutes_mean=("ed_stay_minutes", "mean"),
        )
        .reset_index()
    )

    daily["day"] = pd.to_datetime(START_DATE) + pd.to_timedelta(
        daily["day_index"], unit="D"
    )
    return daily


def add_temporal_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add lagged and rolling features for time-series modeling."""
    features = daily.copy()
    lag_sources = [
        "mortality_rate",
        "los_hours_mean",
        "home_discharge_rate",
        "ed_stay_minutes_mean",
        "died_in_hospital_rate",
    ]
    for col in lag_sources:
        for lag in (1, 7, 14):
            features[f"{col}_lag_{lag}"] = features[col].shift(lag)

    for window in (7, 14):
        features[f"mortality_rate_roll_mean_{window}"] = (
            features["mortality_rate"].rolling(window=window).mean()
        )

    enriched = features.dropna().reset_index(drop=True)
    return enriched


def train_and_evaluate(features: pd.DataFrame) -> tuple[EvaluationResult, pd.DataFrame]:
    """Train the model and return evaluation metrics plus predictions."""
    target_col = "mortality_rate"
    drop_cols = {"day_index", "day", target_col}
    feature_cols = [col for col in features.columns if col not in drop_cols]

    X = features[feature_cols]
    y = features[target_col]

    if len(features) <= TRAIN_TEST_SPLIT_DAYS:
        raise ValueError("可用的特征天数不足以进行时间序列划分。")

    split_point = len(features) - TRAIN_TEST_SPLIT_DAYS
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    day_series = features["day"].iloc[split_point:]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    result_frame = pd.DataFrame(
        {
            "day": day_series.values,
            "actual_mortality_rate": y_test.values,
            "predicted_mortality_rate": predictions,
        }
    )

    current_row = result_frame.iloc[-1]
    evaluation = EvaluationResult(
        mae=float(mae),
        rmse=float(rmse),
        r2=float(r2),
        current_day=str(current_row["day"].date()),
        current_actual=float(current_row["actual_mortality_rate"]),
        current_prediction=float(current_row["predicted_mortality_rate"]),
    )

    return evaluation, result_frame


def save_plot(predictions: pd.DataFrame, output_path: Path) -> None:
    """Plot predicted vs. actual mortality rates and persist the figure."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        predictions["day"],
        predictions["actual_mortality_rate"],
        label="Actual daily mortality rate",
        color="#1f77b4",
        linewidth=2,
    )
    ax.plot(
        predictions["day"],
        predictions["predicted_mortality_rate"],
        label="Model prediction",
        color="#ff7f0e",
        linewidth=2,
        linestyle="--",
    )
    ax.scatter(
        predictions["day"].iloc[-1],
        predictions["actual_mortality_rate"].iloc[-1],
        color="#2ca02c",
        s=80,
        zorder=5,
        label="Latest actual",
    )
    ax.scatter(
        predictions["day"].iloc[-1],
        predictions["predicted_mortality_rate"].iloc[-1],
        color="#d62728",
        s=80,
        zorder=5,
        label="Latest prediction",
        marker="X",
    )
    ax.set_title("Daily mortality rate: actual vs. predicted", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Mortality rate")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_artifacts(
    base_dir: Path,
    daily: pd.DataFrame,
    predictions: pd.DataFrame,
    evaluation: EvaluationResult,
) -> None:
    """Persist analysis artifacts for downstream documentation."""
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    daily.to_csv(artifacts_dir / "daily_metrics.csv", index=False)
    predictions.to_csv(artifacts_dir / "test_predictions.csv", index=False)
    with open(artifacts_dir / "evaluation.json", "w", encoding="utf-8") as fp:
        json.dump(asdict(evaluation), fp, ensure_ascii=False, indent=2)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "features_selected.csv"
    docs_assets = base_dir / "docs" / "assets"
    docs_assets.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    daily = build_daily_table(df)
    enriched = add_temporal_features(daily)
    evaluation, predictions = train_and_evaluate(enriched)

    plot_path = docs_assets / "mortality_forecast.png"
    save_plot(predictions, plot_path)
    save_artifacts(base_dir, daily, predictions, evaluation)

    print("Analysis complete:")
    print(json.dumps(asdict(evaluation), ensure_ascii=False, indent=2))
    print(f"Visualization saved to: {plot_path}")


if __name__ == "__main__":
    main()

