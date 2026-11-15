import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "location_aware_gis_leakage_dataset.csv"
REPORT_PATH = ROOT / "model_comparison_report.json"

FEATURES = [
    "Pressure",
    "Flow_Rate",
    "Temperature",
    "Vibration"
]

LABEL = "Leakage_Flag"
GROUP = "Location_Code"

SEED = 42
N_ESTIMATORS = 300
MAX_FPR = 0.05


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

def calculate_metrics(y_true, scores, threshold):

    predictions = (scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true, predictions, zero_division=0
    )

    recall = recall_score(
        y_true, predictions, zero_division=0
    )

    f1 = f1_score(
        y_true, predictions, zero_division=0
    )

    f2 = (
        5 * precision * recall / (4 * precision + recall)
        if 4 * precision + recall else 0.0
    )

    fpr = fp / (fp + tn) if fp + tn else 0.0

    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "false_positive_rate": float(fpr)
    }


# ------------------------------------------------------------
# THRESHOLD CALIBRATION
# Validation labels only. Never use test labels here.
# ------------------------------------------------------------

def select_thresholds(y_val, val_scores):

    candidates = np.unique(np.concatenate([
        [np.nextafter(val_scores.min(), -np.inf)],
        val_scores,
        [np.nextafter(val_scores.max(), np.inf)]
    ]))

    results = [
        calculate_metrics(y_val, val_scores, t)
        for t in candidates
    ]

    normal_scores = val_scores[y_val == 0]

    threshold_p99 = float(
        np.percentile(normal_scores, 99)
    )

    best_f1 = max(
        results,
        key=lambda r: (r["f1"], r["recall"], r["precision"])
    )

    best_f2 = max(
        results,
        key=lambda r: (r["f2"], r["recall"], r["precision"])
    )

    eligible = [
        r for r in results
        if r["false_positive_rate"] <= MAX_FPR
    ]

    best_fpr = max(
        eligible,
        key=lambda r: (
            r["recall"],
            r["precision"],
            r["f1"]
        )
    )

    return {
        "validation_normal_p99": threshold_p99,
        "validation_best_f1": best_f1["threshold"],
        "validation_best_f2": best_f2["threshold"],
        "validation_max_recall_at_5pct_fpr": best_fpr["threshold"]
    }


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\n========== LOADING DATA ==========")

df = pd.read_csv(DATA_PATH)

required = FEATURES + [LABEL, GROUP]

missing = [
    col for col in required if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

for col in FEATURES + [LABEL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=required).copy()
df = df[df[LABEL].isin([0, 1])].copy()

df[LABEL] = df[LABEL].astype(int)
df[GROUP] = df[GROUP].astype(str)

X = df[FEATURES].reset_index(drop=True)
y = df[LABEL].reset_index(drop=True)
groups = df[GROUP].reset_index(drop=True)

print("Total records:", len(df))
print("Normal:", int((y == 0).sum()))
print("Leaks:", int((y == 1).sum()))
print("Locations:", groups.nunique())


# ------------------------------------------------------------
# GROUP-AWARE SPLITTING
# Same deterministic split as previous experiments.
# ------------------------------------------------------------

print("\n========== DATA SPLITTING ==========")

outer = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

train_val_idx, test_idx = next(
    outer.split(X, y, groups)
)

inner = StratifiedGroupKFold(
    n_splits=4,
    shuffle=True,
    random_state=SEED + 1
)

train_rel, val_rel = next(
    inner.split(
        X.iloc[train_val_idx],
        y.iloc[train_val_idx],
        groups.iloc[train_val_idx]
    )
)

train_idx = train_val_idx[train_rel]
val_idx = train_val_idx[val_rel]

train_groups = set(groups.iloc[train_idx])
val_groups = set(groups.iloc[val_idx])
test_groups = set(groups.iloc[test_idx])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

print("Training locations:", len(train_groups))
print("Validation locations:", len(val_groups))
print("Testing locations:", len(test_groups))

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx].to_numpy()

X_val = X.iloc[val_idx]
y_val = y.iloc[val_idx].to_numpy()

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx].to_numpy()


# ------------------------------------------------------------
# MODEL DEFINITIONS
# ------------------------------------------------------------

models = {

    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=SEED
        )
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1
    )
}


# ------------------------------------------------------------
# ISOLATION FOREST BASELINE
# Fit only on normal training observations.
# ------------------------------------------------------------

print("\n========== ISOLATION FOREST ==========")

normal_train_mask = y_train == 0
X_train_normal = X_train.iloc[
    np.flatnonzero(normal_train_mask)
]

if_scaler = StandardScaler()

X_normal_scaled = if_scaler.fit_transform(
    X_train_normal
)

if_model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination="auto",
    random_state=SEED,
    n_jobs=-1
)

if_model.fit(X_normal_scaled)

if_val_scores = -if_model.score_samples(
    if_scaler.transform(X_val)
)

if_test_scores = -if_model.score_samples(
    if_scaler.transform(X_test)
)

all_models = {
    "Isolation Forest": (
        if_val_scores,
        if_test_scores
    )
}


# ------------------------------------------------------------
# TRAIN SUPERVISED MODELS
# Both classes from training locations are used.
# ------------------------------------------------------------

for name, model in models.items():

    print(f"\n========== TRAINING {name} ==========")

    model.fit(X_train, y_train)

    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    all_models[name] = (
        val_scores,
        test_scores
    )


# ------------------------------------------------------------
# VALIDATION THRESHOLD SELECTION + TEST EVALUATION
# ------------------------------------------------------------

print("\n========== FINAL MODEL COMPARISON ==========")

report = {
    "experiment": "Location-grouped model comparison",
    "features": FEATURES,
    "seed": SEED,
    "estimators": N_ESTIMATORS,
    "locations": {
        "training": len(train_groups),
        "validation": len(val_groups),
        "test": len(test_groups)
    },
    "training_records": {
        "total": len(train_idx),
        "normal": int((y_train == 0).sum()),
        "leak": int((y_train == 1).sum())
    },
    "test_records": {
        "total": len(test_idx),
        "normal": int((y_test == 0).sum()),
        "leak": int((y_test == 1).sum())
    },
    "models": {}
}

strategies = [
    "validation_normal_p99",
    "validation_best_f1",
    "validation_best_f2",
    "validation_max_recall_at_5pct_fpr"
]

for name, (val_scores, test_scores) in all_models.items():

    thresholds = select_thresholds(y_val, val_scores)

    model_report = {
        "thresholds": thresholds,
        "validation": {},
        "test": {}
    }

    print(f"\n\n----- {name} -----")

    for strategy in strategies:

        threshold = thresholds[strategy]

        val_result = calculate_metrics(
            y_val, val_scores, threshold
        )

        test_result = calculate_metrics(
            y_test, test_scores, threshold
        )

        model_report["validation"][strategy] = val_result
        model_report["test"][strategy] = test_result

        print(f"\n{strategy}")
        print(
            f"Threshold: {threshold:.6f} | "
            f"TP={test_result['tp']} "
            f"FP={test_result['fp']} "
            f"TN={test_result['tn']} "
            f"FN={test_result['fn']}"
        )

        print(
            f"Precision={test_result['precision']:.4f} | "
            f"Recall={test_result['recall']:.4f} | "
            f"F1={test_result['f1']:.4f} | "
            f"F2={test_result['f2']:.4f} | "
            f"FPR={test_result['false_positive_rate']:.4f}"
        )

    model_report["ranking_metrics"] = {
        "roc_auc": float(
            roc_auc_score(y_test, test_scores)
        ),
        "average_precision": float(
            average_precision_score(y_test, test_scores)
        )
    }

    report["models"][name] = model_report


# ------------------------------------------------------------
# SAVE REPORT
# ------------------------------------------------------------

REPORT_PATH.write_text(
    json.dumps(report, indent=4),
    encoding="utf-8"
)

print("\nReport saved to:", REPORT_PATH)
print("Production model artifacts were NOT modified.")
