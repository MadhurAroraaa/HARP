import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "location_aware_gis_leakage_dataset.csv"
REPORT = ROOT / "threshold_analysis_report.json"

FEATURES = [
    "Pressure",
    "Flow_Rate",
    "Temperature",
    "Vibration",
]

SEED = 42
N_ESTIMATORS = 300
MAX_VALIDATION_FPR = 0.05


def metrics(y, scores, threshold):
    pred = (scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    f2 = (
        5 * precision * recall / (4 * precision + recall)
        if 4 * precision + recall else 0.0
    )

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
        "false_positive_rate": float(fpr),
    }


print("\n========== LOADING DATA ==========")

df = pd.read_csv(DATA)

df = df.dropna(
    subset=FEATURES + ["Leakage_Flag", "Location_Code"]
).copy()

for col in FEATURES + ["Leakage_Flag"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURES + ["Leakage_Flag"])

df = df[df["Leakage_Flag"].isin([0, 1])].copy()
df["Leakage_Flag"] = df["Leakage_Flag"].astype(int)

X = df[FEATURES].reset_index(drop=True)
y = df["Leakage_Flag"].reset_index(drop=True)
groups = df["Location_Code"].astype(str).reset_index(drop=True)


print("Records:", len(df))
print("Leakage:", int(y.sum()))
print("Normal:", int((y == 0).sum()))


# Same deterministic grouped split as evaluate_model.py.
outer = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

train_val_idx, test_idx = next(
    outer.split(X, y, groups)
)

inner = StratifiedGroupKFold(
    n_splits=4,
    shuffle=True,
    random_state=SEED + 1,
)

train_rel, val_rel = next(
    inner.split(
        X.iloc[train_val_idx],
        y.iloc[train_val_idx],
        groups.iloc[train_val_idx],
    )
)

train_idx = train_val_idx[train_rel]
val_idx = train_val_idx[val_rel]

# Verify group separation.
train_groups = set(groups.iloc[train_idx])
val_groups = set(groups.iloc[val_idx])
test_groups = set(groups.iloc[test_idx])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)


# Train on normal training records only.
normal_train_idx = train_idx[
    y.iloc[train_idx].to_numpy() == 0
]

scaler = StandardScaler()
X_train = scaler.fit_transform(X.iloc[normal_train_idx])

model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination="auto",
    random_state=SEED,
    n_jobs=-1,
)

model.fit(X_train)


# Obtain anomaly scores.
# Higher scores mean more anomalous.
val_scores = -model.score_samples(
    scaler.transform(X.iloc[val_idx])
)

test_scores = -model.score_samples(
    scaler.transform(X.iloc[test_idx])
)

y_val = y.iloc[val_idx].to_numpy()
y_test = y.iloc[test_idx].to_numpy()


# Candidate 1: Current conservative threshold.
normal_val_scores = val_scores[y_val == 0]

threshold_p99 = float(
    np.percentile(normal_val_scores, 99)
)


# Search candidate thresholds using validation labels only.
candidate_thresholds = np.unique(val_scores)

candidate_results = [
    metrics(y_val, val_scores, t)
    for t in candidate_thresholds
]


# Candidate 2: Best validation F1.
best_f1 = max(
    candidate_results,
    key=lambda r: (r["f1"], r["precision"])
)


# Candidate 3: Best validation F2.
best_f2 = max(
    candidate_results,
    key=lambda r: (r["f2"], r["precision"])
)


# Candidate 4: Highest recall under a 5% validation FPR.
eligible = [
    r for r in candidate_results
    if r["false_positive_rate"] <= MAX_VALIDATION_FPR
]

if not eligible:
    raise RuntimeError(
        "No threshold satisfies the validation FPR constraint."
    )

best_fpr = max(
    eligible,
    key=lambda r: (
        r["recall"],
        r["precision"],
        r["threshold"],
    )
)


thresholds = {
    "validation_normal_p99": threshold_p99,
    "validation_best_f1": best_f1["threshold"],
    "validation_best_f2": best_f2["threshold"],
    "validation_max_recall_at_5pct_fpr": best_fpr["threshold"],
}


print("\n========== VALIDATION THRESHOLD SELECTION ==========")

for name, threshold in thresholds.items():
    result = metrics(y_val, val_scores, threshold)

    print(f"\n{name}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1: {result['f1']:.4f}")
    print(f"F2: {result['f2']:.4f}")
    print(f"FPR: {result['false_positive_rate']:.4f}")


# Apply all pre-selected thresholds to test data.
# Test labels do not participate in threshold selection.
print("\n========== UNTOUCHED TEST EVALUATION ==========")

test_results = {}

for name, threshold in thresholds.items():
    result = metrics(y_test, test_scores, threshold)
    test_results[name] = result

    print(f"\n{name}")
    print(f"Threshold: {threshold:.6f}")
    print(
        f"TP={result['tp']} FP={result['fp']} "
        f"TN={result['tn']} FN={result['fn']}"
    )
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1: {result['f1']:.4f}")
    print(f"F2: {result['f2']:.4f}")
    print(
        f"False-positive rate: "
        f"{result['false_positive_rate']:.4f}"
    )


# Threshold-independent ranking metrics.
ranking = {
    "roc_auc": float(roc_auc_score(y_test, test_scores)),
    "average_precision": float(
        average_precision_score(y_test, test_scores)
    ),
    "test_leak_prevalence": float(y_test.mean()),
}

report = {
    "experiment": "Location-grouped threshold analysis",
    "model": "IsolationForest",
    "features": FEATURES,
    "training": "Normal-labelled training records only",
    "threshold_calibration": (
        "Validation locations only; test labels not used"
    ),
    "validation_locations": len(val_groups),
    "test_locations": len(test_groups),
    "thresholds": thresholds,
    "validation_selected_metrics": {
        "best_f1": best_f1,
        "best_f2": best_f2,
        "max_recall_at_5pct_fpr": best_fpr,
    },
    "test_metrics": test_results,
    "ranking_metrics": ranking,
    "production_artifacts_modified": False,
}

REPORT.write_text(
    json.dumps(report, indent=4),
    encoding="utf-8",
)

print("\nReport saved to:", REPORT)
print("Production model artifacts remain untouched.")
