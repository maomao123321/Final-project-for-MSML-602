"""Data preprocessing and feature engineering pipeline for clinical CSV datasets.

This script loads the six CSV databases located in the working directory,
performs data cleaning steps tailored for structured data, engineers a set of
modeling features at the hospital admission level, runs a lightweight feature
selection routine, and saves both intermediate artefacts and the final
feature tables under the ``processed`` directory.

Outputs
-------
processed/
    clean_admissions.parquet (falls back to CSV if parquet engine missing)
    clean_patients.parquet (fallback CSV)
    icu_aggregates.parquet (fallback CSV)
    lab_aggregates.parquet (fallback CSV)
    omr_latest.parquet (fallback CSV)
    features_full.csv
    features_selected.csv
    feature_metadata.json

The feature selection step favours interpretable, information-rich fields by
combining missing-rate filtering, variance filtering, and mutual information
scores (if scikit-learn is available).

The script is idempotent and can be rerun safely.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:  
    from sklearn.feature_selection import mutual_info_classif

    SKLEARN_AVAILABLE = True
except ImportError:  
    SKLEARN_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILES = {
    "admissions": "admissions.csv",
    "patients": "patients.csv",
    "icustays": "icustays.csv",
    "labevents": "labevents.csv",
    "omr": "omr.csv",
    "diagnoses": "d_icd_diagnoses.csv",
}
PROCESSED_DIR = BASE_DIR / "processed"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _ensure_int(series: pd.Series) -> pd.Series:
    """Return a nullable integer series; preserves missing values."""

    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    """Convert object series to float when possible, otherwise NaN."""

    if series.dtype.kind in {"i", "u", "f"}:
        return series
    return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")


def _duration_hours(end: pd.Series, start: pd.Series) -> pd.Series:
    """Compute duration in hours between two datetime series."""

    return (end - start).dt.total_seconds() / 3600.0


def _duration_minutes(end: pd.Series, start: pd.Series) -> pd.Series:
    return (end - start).dt.total_seconds() / 60.0


def _recent_records(df: pd.DataFrame, group_cols: List[str], timestamp_col: str) -> pd.DataFrame:
    """Return the most recent row per group based on timestamp_col."""

    idx = df.groupby(group_cols)[timestamp_col].transform("idxmax")
    return df.loc[idx].reset_index(drop=True)


def _save_table(df: pd.DataFrame, filename: str) -> str:
    """Persist DataFrame as parquet if available; otherwise CSV.

    Returns the final filename that was written.
    """

    target = PROCESSED_DIR / filename
    suffix = target.suffix.lower()
    if suffix == ".parquet":
        try:
            df.to_parquet(target, index=False)
            return target.name
        except ImportError:
            csv_target = target.with_suffix(".csv")
            df.to_csv(csv_target, index=False)
            return csv_target.name
    elif suffix == ".csv":
        df.to_csv(target, index=False)
        return target.name
    else:
        raise ValueError(f"Unsupported file extension for {target}")


# ---------------------------------------------------------------------------
# Admissions
# ---------------------------------------------------------------------------


def load_and_clean_admissions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        parse_dates=[
            "admittime",
            "dischtime",
            "deathtime",
            "edregtime",
            "edouttime",
        ],
        na_values=["", "NA", "NaN", "?"],
    )

    df = df.drop_duplicates(subset=["hadm_id"], keep="last")
    df["subject_id"] = _ensure_int(df["subject_id"])
    df["hadm_id"] = _ensure_int(df["hadm_id"])

    # Feature engineering
    df["los_hours"] = _duration_hours(df["dischtime"], df["admittime"])
    df.loc[df["los_hours"] < 0, "los_hours"] = np.nan  # guard bad timestamps

    df["ed_wait_minutes"] = _duration_minutes(df["admittime"], df["edregtime"])
    df["ed_stay_minutes"] = _duration_minutes(df["edouttime"], df["edregtime"])

    df["died_in_hospital"] = df["hospital_expire_flag"].fillna(0).astype("Int64")

    # Harmonise key categoricals
    categorical_cols = [
        "admission_type",
        "admission_location",
        "discharge_location",
        "insurance",
        "language",
        "marital_status",
        "race",
    ]
    for col in categorical_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": np.nan, "?": np.nan, "": np.nan})
        )

    df["marital_status"] = df["marital_status"].fillna("UNKNOWN")
    df["language"] = df["language"].fillna("UNKNOWN")

    return df


# ---------------------------------------------------------------------------
# Patients
# ---------------------------------------------------------------------------


def load_and_clean_patients(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["dod"], na_values=["", "NA", "?"])
    df = df.drop_duplicates(subset=["subject_id"], keep="last")

    df["subject_id"] = _ensure_int(df["subject_id"])
    df["anchor_age"] = pd.to_numeric(df["anchor_age"], errors="coerce")
    df["anchor_year"] = pd.to_numeric(df["anchor_year"], errors="coerce")

    df["dod_available"] = df["dod"].notna().astype("Int64")

    age_bins = [0, 40, 60, 75, 200]
    age_labels = ["<40", "40-59", "60-74", "75+"]
    df["age_group"] = pd.cut(df["anchor_age"], bins=age_bins, labels=age_labels, right=False)

    df["gender"] = df["gender"].str.upper().str.strip()

    return df


# ---------------------------------------------------------------------------
# ICU stays
# ---------------------------------------------------------------------------


def load_and_aggregate_icustays(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        parse_dates=["intime", "outtime"],
        na_values=["", "NA", "NaN"],
    )

    df["subject_id"] = _ensure_int(df["subject_id"])
    df["hadm_id"] = _ensure_int(df["hadm_id"])
    df["stay_id"] = _ensure_int(df["stay_id"])

    df["los"] = pd.to_numeric(df["los"], errors="coerce")

    aggregations = {
        "stay_id": "count",
        "los": ["sum", "mean", "max"],
    }
    grouped = df.groupby("hadm_id").agg(aggregations)
    grouped.columns = [
        "icu_stay_count",
        "icu_los_hours_sum",
        "icu_los_hours_mean",
        "icu_los_hours_max",
    ]

    first_last_units = df.sort_values("intime").groupby("hadm_id").agg(
        first_careunit_first=("first_careunit", "first"),
        first_careunit_last=("first_careunit", "last"),
        last_careunit_first=("last_careunit", "first"),
        last_careunit_last=("last_careunit", "last"),
    )

    aggregated = grouped.join(first_last_units, how="left")
    aggregated.reset_index(inplace=True)

    return aggregated


# ---------------------------------------------------------------------------
# Lab events
# ---------------------------------------------------------------------------


@dataclass
class LabFeatureSpec:
    itemid: int
    name: str


def _select_lab_items(
    df: pd.DataFrame, eligible_hadm: Iterable[int], top_n: int = 15, min_coverage: float = 0.15
) -> List[LabFeatureSpec]:
    hadm_set = pd.Index(eligible_hadm)
    coverage = (
        df.dropna(subset=["hadm_id"])
        .groupby("itemid")["hadm_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    coverage_ratio = coverage / hadm_set.nunique()

    selected_itemids = coverage_ratio[coverage_ratio >= min_coverage].head(top_n).index.tolist()
    specs = [LabFeatureSpec(int(item), f"lab_{item}") for item in selected_itemids]
    return specs


def load_and_aggregate_labs(path: Path, eligible_hadm: Iterable[int]) -> Tuple[pd.DataFrame, List[LabFeatureSpec]]:
    df = pd.read_csv(
        path,
        parse_dates=["charttime", "storetime"],
        na_values=["", "NA", "NaN", "?", "___"],
    )

    df["subject_id"] = _ensure_int(df["subject_id"])
    df["hadm_id"] = _ensure_int(df["hadm_id"])

    # Prefer valuenum; fall back to parsed value
    df["value_num_clean"] = df["valuenum"]
    missing_mask = df["value_num_clean"].isna()
    df.loc[missing_mask, "value_num_clean"] = _safe_to_numeric(df.loc[missing_mask, "value"])

    df = df.dropna(subset=["hadm_id", "value_num_clean", "charttime"])

    specs = _select_lab_items(df, eligible_hadm)
    if not specs:
        return pd.DataFrame(columns=["hadm_id"]), []

    frames = []
    for spec in specs:
        subset = df[df["itemid"] == spec.itemid].copy()
        if subset.empty:
            continue
        subset.sort_values("charttime", inplace=True)

        agg = subset.groupby("hadm_id").agg(
            **{
                f"{spec.name}_count": ("value_num_clean", "count"),
                f"{spec.name}_mean": ("value_num_clean", "mean"),
                f"{spec.name}_std": ("value_num_clean", "std"),
                f"{spec.name}_min": ("value_num_clean", "min"),
                f"{spec.name}_max": ("value_num_clean", "max"),
            }
        )

        last_values = (
            subset.sort_values("charttime")
            .groupby("hadm_id", as_index=False)
            .tail(1)
            .set_index("hadm_id")["value_num_clean"]
        )
        agg[f"{spec.name}_last"] = agg.index.map(last_values)

        frames.append(agg)

    if not frames:
        return pd.DataFrame(columns=["hadm_id"]), []

    lab_features = pd.concat(frames, axis=1)
    lab_features.reset_index(inplace=True)
    lab_features = lab_features.loc[:, ~lab_features.columns.duplicated()]

    return lab_features, specs


# ---------------------------------------------------------------------------
# Outpatient (OMR) measurements
# ---------------------------------------------------------------------------


def _parse_blood_pressure(value: str) -> Tuple[float | None, float | None]:
    if not isinstance(value, str):
        return (np.nan, np.nan)
    parts = value.replace(" ", "").split("/")
    if len(parts) != 2:
        return (np.nan, np.nan)
    systolic = pd.to_numeric(parts[0], errors="coerce")
    diastolic = pd.to_numeric(parts[1], errors="coerce")
    return (systolic, diastolic)


def load_and_aggregate_omr(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["chartdate"], na_values=["", "NA", "NaN", "?"])
    df["subject_id"] = _ensure_int(df["subject_id"])

    df["result_name"] = df["result_name"].str.strip().str.upper()

    numeric_mask = df["result_name"].isin(
        [
            "HEIGHT (INCHES)",
            "WEIGHT (LBS)",
            "BMI (KG/M2)",
            "WEIGHT (LBS)",
        ]
    )
    df.loc[numeric_mask, "result_value_numeric"] = _safe_to_numeric(df.loc[numeric_mask, "result_value"])

    latest = _recent_records(df, ["subject_id", "result_name"], "chartdate")

    # Convert to feature columns
    features: Dict[str, pd.Series] = {"subject_id": latest["subject_id"]}

    def _assign_feature(name: str, mask: pd.Series, values: pd.Series) -> None:
        colname = f"omr_{name}"
        features[colname] = values.where(mask).groupby(latest["subject_id"]).first()

    height_mask = latest["result_name"] == "HEIGHT (INCHES)"
    height_cm = latest.loc[height_mask, "result_value_numeric"] * 2.54
    _assign_feature("height_cm", height_mask, height_cm)

    weight_mask = latest["result_name"] == "WEIGHT (LBS)"
    weight_kg = latest.loc[weight_mask, "result_value_numeric"] * 0.45359237
    _assign_feature("weight_kg", weight_mask, weight_kg)

    bmi_mask = latest["result_name"] == "BMI (KG/M2)"
    bmi_val = latest.loc[bmi_mask, "result_value_numeric"]
    _assign_feature("bmi", bmi_mask, bmi_val)

    bp_mask = latest["result_name"] == "BLOOD PRESSURE"
    bp_vals = latest.loc[bp_mask, "result_value"].apply(_parse_blood_pressure)
    if not bp_vals.empty:
        systolic = bp_vals.apply(lambda x: x[0])
        diastolic = bp_vals.apply(lambda x: x[1])
        _assign_feature("blood_pressure_systolic", bp_mask, systolic)
        _assign_feature("blood_pressure_diastolic", bp_mask, diastolic)

    result = pd.DataFrame(features).groupby("subject_id").first().reset_index()
    return result


# ---------------------------------------------------------------------------
# Feature assembly & selection
# ---------------------------------------------------------------------------


def assemble_feature_table(
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
    icu: pd.DataFrame,
    lab: pd.DataFrame,
    omr: pd.DataFrame,
) -> pd.DataFrame:
    df = admissions.merge(patients, on="subject_id", how="left", suffixes=("", "_pat"))

    if not icu.empty:
        df = df.merge(icu, on="hadm_id", how="left")

    if not lab.empty:
        df = df.merge(lab, on="hadm_id", how="left")

    if not omr.empty:
        df = df.merge(omr, on="subject_id", how="left")

    # Drop columns with excessive missingness (> 70%)
    missing_ratio = df.isna().mean()
    keep_cols = missing_ratio[missing_ratio <= 0.7].index.tolist()
    df = df[keep_cols]

    return df


def _prepare_numeric_matrix(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    feature_df = df.drop(columns=[target_col]).copy()
    target = df[target_col].astype("Int64").fillna(0)

    # Exclude identifiers from modeling features
    identifier_cols = [col for col in feature_df.columns if col in {"subject_id", "hadm_id"}]
    feature_df = feature_df.drop(columns=identifier_cols, errors="ignore")

    # Convert datetime and timedelta columns to numeric representations
    datetime_cols = feature_df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
    for col in datetime_cols:
        feature_df[col] = feature_df[col].apply(lambda x: x.value if pd.notna(x) else np.nan).astype(float)

    timedelta_cols = feature_df.select_dtypes(include=["timedelta64[ns]"]).columns
    for col in timedelta_cols:
        feature_df[col] = feature_df[col].dt.total_seconds()

    # One-hot encode categorical variables
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns
    feature_df = pd.get_dummies(feature_df, columns=categorical_cols, dummy_na=True)

    # Impute missing values with median (numeric only now)
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    # Drop constant columns
    nunique = feature_df.nunique()
    feature_df = feature_df.loc[:, nunique > 1]

    return feature_df, target


def select_features(df: pd.DataFrame, target_col: str, k: int = 20) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    X, y = _prepare_numeric_matrix(df, target_col)

    if X.empty:
        return df[["subject_id", "hadm_id", target_col]].copy(), [], {}

    feature_scores: Dict[str, float]

    if SKLEARN_AVAILABLE and y.nunique() > 1:
        scores = mutual_info_classif(X, y)
        feature_scores = dict(zip(X.columns, scores))
    else:
        # Fallback: absolute Pearson correlation
        corrs = {}
        for col in X.columns:
            series = X[col]
            values = series.to_numpy(dtype=float)
            if np.nanstd(values) == 0:
                corrs[col] = 0.0
                continue
            corrs[col] = abs(np.corrcoef(values, y)[0, 1])
        feature_scores = corrs

    top_features = sorted(feature_scores, key=feature_scores.get, reverse=True)[:k]

    selected_df = df[["subject_id", "hadm_id", target_col]].join(X[top_features])
    return selected_df, top_features, feature_scores


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    PROCESSED_DIR.mkdir(exist_ok=True)

    admissions = load_and_clean_admissions(BASE_DIR / RAW_FILES["admissions"])
    patients = load_and_clean_patients(BASE_DIR / RAW_FILES["patients"])
    icu = load_and_aggregate_icustays(BASE_DIR / RAW_FILES["icustays"])

    lab_path = BASE_DIR / RAW_FILES["labevents"]
    if lab_path.exists():
        labs, lab_specs = load_and_aggregate_labs(lab_path, admissions["hadm_id"].dropna().unique())
    else:
        labs = pd.DataFrame()
        lab_specs: List[LabFeatureSpec] = []

    omr_path = BASE_DIR / RAW_FILES["omr"]
    if omr_path.exists():
        omr = load_and_aggregate_omr(omr_path)
    else:
        omr = pd.DataFrame()

    # Persist intermediate datasets
    saved_files = {
        "admissions": _save_table(admissions, "clean_admissions.parquet"),
        "patients": _save_table(patients, "clean_patients.parquet"),
        "icu": _save_table(icu, "icu_aggregates.parquet"),
        "labs": _save_table(labs, "lab_aggregates.parquet") if not labs.empty else None,
        "omr": _save_table(omr, "omr_latest.parquet") if not omr.empty else None,
    }

    feature_table = assemble_feature_table(admissions, patients, icu, labs, omr)

    selected_df, top_features, feature_scores = select_features(feature_table, "hospital_expire_flag")

    full_features_filename = _save_table(feature_table, "features_full.csv")
    selected_features_filename = _save_table(selected_df, "features_selected.csv")

    metadata = {
        "lab_features": [spec.__dict__ for spec in lab_specs],
        "selected_feature_names": top_features,
        "feature_scores": feature_scores,
        "sklearn_used": SKLEARN_AVAILABLE,
        "rows": {
            "feature_full": int(feature_table.shape[0]),
            "feature_selected": int(selected_df.shape[0]),
        },
        "saved_files": {
            **saved_files,
            "features_full": full_features_filename,
            "features_selected": selected_features_filename,
        },
    }

    with open(PROCESSED_DIR / "feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Data preprocessing complete. Outputs saved under 'processed/'.")


if __name__ == "__main__":
    main()

