import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "location_aware_gis_leakage_dataset.csv"
REPORT_PATH = ROOT / "evaluation_report.json"

FEATURES = [
    "Pressure",
    "Flow_Rate",
    "Temperature",
    "Vibration",
]

LABEL = "Leakage_Flag"
GROUP = "Location_Code"

RANDOM_STATE = 42
N_ESTIMATORS = 300
VALIDATION_PERCENTILE = 99


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

print("\n========== LOADING DATASET ==========")

if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required_columns = FEATURES + [LABEL, GROUP]
missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

for col in FEATURES + [LABEL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(subset=required_columns).copy()

df = df[df[LABEL].isin([0, 1])].copy()
df[LABEL] = df[LABEL].astype(int)
df[GROUP] = df[GROUP].astype(str)

if set(df[LABEL].unique()) != {0, 1}:
    raise ValueError("Dataset must contain both normal and leakage labels.")

X = df[FEATURES].reset_index(drop=True)
y = df[LABEL].reset_index(drop=True)
groups = df[GROUP].reset_index(drop=True)

print(f"Total valid records: {len(df)}")
print(f"Normal records: {(y == 0).sum()}")
print(f"Leakage records: {(y == 1).sum()}")
print(f"Unique locations: {groups.nunique()}")


# ============================================================
# 3. OUTER GROUP SPLIT: 20% UNSEEN TEST LOCATIONS
# ============================================================

print("\n========== GROUP-AWARE DATA SPLITTING ==========")

outer_splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

train_val_idx, test_idx = next(
    outer_splitter.split(X, y, groups)
)


# ============================================================
# 4. INNER GROUP SPLIT: VALIDATION LOCATIONS
# ============================================================

inner_splitter = StratifiedGroupKFold(
    n_splits=4,
    shuffle=True,
    random_state=RANDOM_STATE + 1,
)

train_rel_idx, val_rel_idx = next(
    inner_splitter.split(
        X.iloc[train_val_idx],
        y.iloc[train_val_idx],
        groups.iloc[train_val_idx],
    )
)

train_idx = train_val_idx[train_rel_idx]
val_idx = train_val_idx[val_rel_idx]


# ============================================================
# 5. VERIFY LOCATION SEPARATION
# ============================================================

train_groups = set(groups.iloc[train_idx])
val_groups = set(groups.iloc[val_idx])
test_groups = set(groups.iloc[test_idx])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

print(f"Training locations: {len(train_groups)}")
print(f"Validation locations: {len(val_groups)}")
print(f"Test locations: {len(test_groups)}")

print("\nRecord distribution:")

for name, indices in [
    ("Training", train_idx),
    ("Validation", val_idx),
    ("Test", test_idx),
]:
    labels = y.iloc[indices]

    print(
        f"{name}: {len(indices)} records | "
        f"Normal: {(labels == 0).sum()} | "
        f"Leaks: {(labels == 1).sum()}"
    )


# ============================================================
# 6. TRAIN ONLY ON NORMAL TRAINING RECORDS
# ============================================================

print("\n========== TRAINING ISOLATION FOREST ==========")

y_train = y.iloc[train_idx].to_numpy()

normal_train_idx = train_idx[y_train == 0]

X_train_normal = X.iloc[normal_train_idx]

if len(X_train_normal) < 100:
    raise ValueError("Insufficient normal training records.")

print(f"Normal training records: {len(X_train_normal)}")

# Fit scaler only on normal training observations.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_normal)

model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(X_train_scaled)

print("Isolation Forest training completed.")


# ============================================================
# 7. CALIBRATE THRESHOLD ON VALIDATION NORMALS
# ============================================================

print("\n========== THRESHOLD CALIBRATION ==========")

y_val = y.iloc[val_idx].to_numpy()
X_val = X.iloc[val_idx]

val_normal_mask = y_val == 0

if val_normal_mask.sum() < 50:
    raise ValueError("Insufficient validation-normal records.")

X_val_normal = X_val.iloc[np.flatnonzero(val_normal_mask)]

X_val_normal_scaled = scaler.transform(X_val_normal)

val_normal_scores = -model.score_samples(
    X_val_normal_scaled
)

threshold = float(
    np.percentile(
        val_normal_scores,
        VALIDATION_PERCENTILE
    )
)

print(f"Validation normal records: {len(X_val_normal)}")
print(f"Threshold percentile: {VALIDATION_PERCENTILE}")
print(f"Selected anomaly threshold: {threshold:.6f}")


# ============================================================
# 8. EVALUATE ON UNTOUCHED TEST LOCATIONS
# ============================================================

print("\n========== FINAL TEST EVALUATION ==========")

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx].to_numpy()

X_test_scaled = scaler.transform(X_test)

# Higher scores indicate greater anomaly.
test_scores = -model.score_samples(X_test_scaled)

# Threshold fixed before looking at test labels.
y_pred = (test_scores > threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1],
).ravel()

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0,
)

accuracy = accuracy_score(y_test, y_pred)

false_positive_rate = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0.0
)

roc_auc = (
    roc_auc_score(y_test, test_scores)
    if len(np.unique(y_test)) == 2
    else None
)

average_precision = (
    average_precision_score(y_test, test_scores)
    if len(np.unique(y_test)) == 2
    else None
)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n========== CONFUSION MATRIX ==========")
print("                 Predicted")
print("                 Normal   Leak")
print(f"Actual Normal    {tn:6d}  {fp:6d}")
print(f"Actual Leak      {fn:6d}  {tp:6d}")

print("\n========== PERFORMANCE METRICS ==========")

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "false_positive_rate": float(false_positive_rate),
    "roc_auc": float(roc_auc) if roc_auc is not None else None,
    "average_precision": (
        float(average_precision)
        if average_precision is not None
        else None
    ),
}

for name, value in metrics.items():
    print(
        f"{name:25s}: "
        f"{value:.4f}" if value is not None
        else f"{name:25s}: unavailable"
    )


# ============================================================
# 10. SAVE EXPERIMENT REPORT
# ============================================================

report = {
    "experiment": "Location-grouped Isolation Forest evaluation",
    "model": "IsolationForest",
    "features": FEATURES,
    "training_method": "Normal-labelled records only",
    "split_method": "StratifiedGroupKFold",
    "threshold_method": (
        f"{VALIDATION_PERCENTILE}th percentile of "
        "validation-normal anomaly scores"
    ),
    "threshold": threshold,
    "random_state": RANDOM_STATE,
    "n_estimators": N_ESTIMATORS,
    "locations": {
        "training": len(train_groups),
        "validation": len(val_groups),
        "test": len(test_groups),
    },
    "records": {
        "training_normal": len(normal_train_idx),
        "validation": len(val_idx),
        "test": len(test_idx),
    },
    "confusion_matrix": {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    },
    "metrics": metrics,
    "production_artifacts_modified": False,
}

REPORT_PATH.write_text(
    json.dumps(report, indent=4),
    encoding="utf-8",
)

print(f"\nEvaluation report saved to: {REPORT_PATH}")
print("Production model artifacts were NOT modified.")
