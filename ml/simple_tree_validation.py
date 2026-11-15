import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data" / "location_aware_gis_leakage_dataset.csv"
REPORT = ROOT / "simple_tree_validation_report.json"

FEATURES = [
    "Pressure",
    "Flow_Rate",
    "Temperature",
    "Vibration"
]

SEEDS = [42, 123, 2026]
DEPTHS = [1, 2, 3]
MAX_FPR = 0.05


def calculate_metrics(y, scores, threshold):
    predictions = (scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y, predictions, zero_division=0
    )

    recall = recall_score(
        y, predictions, zero_division=0
    )

    f1 = f1_score(
        y, predictions, zero_division=0
    )

    fpr = fp / (fp + tn) if fp + tn else 0.0

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr)
    }


def select_threshold(y, scores):
    unique_scores = np.unique(scores)

    # Include the boundary immediately below every unique score.
    # This avoids missing score groups because predictions use >.
    candidates = np.concatenate([
        np.nextafter(unique_scores, -np.inf),
        [np.nextafter(unique_scores.max(), np.inf)]
    ])

    eligible = []

    for threshold in candidates:
        result = calculate_metrics(y, scores, threshold)

        if result["false_positive_rate"] <= MAX_FPR:
            eligible.append((threshold, result))

    if not eligible:
        raise RuntimeError("No threshold satisfies the FPR limit.")

    return max(
        eligible,
        key=lambda item: (
            item[1]["recall"],
            item[1]["precision"],
            item[1]["f1"]
        )
    )[0]


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA)

required = FEATURES + ["Leakage_Flag", "Location_Code"]

df = df.dropna(subset=required).copy()

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
print("Locations:", groups.nunique())


# --------------------------------------------------
# Repeated nested grouped cross-validation
# --------------------------------------------------

results = {
    depth: [] for depth in DEPTHS
}

for seed in SEEDS:

    outer = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    for fold, (train_val, test) in enumerate(
        outer.split(X, y, groups), start=1
    ):

        inner = StratifiedGroupKFold(
            n_splits=4,
            shuffle=True,
            random_state=seed + fold
        )

        train_rel, val_rel = next(
            inner.split(
                X.iloc[train_val],
                y.iloc[train_val],
                groups.iloc[train_val]
            )
        )

        train = train_val[train_rel]
        val = train_val[val_rel]

        train_groups = set(groups.iloc[train])
        val_groups = set(groups.iloc[val])
        test_groups = set(groups.iloc[test])

        assert train_groups.isdisjoint(val_groups)
        assert train_groups.isdisjoint(test_groups)
        assert val_groups.isdisjoint(test_groups)

        X_train = X.iloc[train]
        y_train = y.iloc[train].to_numpy()

        X_val = X.iloc[val]
        y_val = y.iloc[val].to_numpy()

        X_test = X.iloc[test]
        y_test = y.iloc[test].to_numpy()

        for depth in DEPTHS:

            model = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=seed
            )

            model.fit(X_train, y_train)

            val_scores = model.predict_proba(X_val)[:, 1]
            test_scores = model.predict_proba(X_test)[:, 1]

            threshold = select_threshold(
                y_val, val_scores
            )

            result = calculate_metrics(
                y_test,
                test_scores,
                threshold
            )

            result.update({
                "seed": seed,
                "fold": fold,
                "threshold": float(threshold)
            })

            results[depth].append(result)

        print(f"Completed seed {seed}, fold {fold}/5")


# --------------------------------------------------
# Summary
# --------------------------------------------------

report = {
    "experiment": "Repeated location-grouped decision tree validation",
    "features": FEATURES,
    "seeds": SEEDS,
    "depths": DEPTHS,
    "threshold_selection": "Maximum validation recall at FPR <= 5%",
    "production_artifacts_modified": False,
    "results": {}
}

metrics = [
    "precision",
    "recall",
    "f1",
    "false_positive_rate"
]

print("\n========== FINAL SUMMARY ==========")

for depth, fold_results in results.items():

    summary = {}

    print(f"\nDECISION TREE DEPTH {depth}")

    for metric in metrics:
        values = np.array([
            r[metric] for r in fold_results
        ])

        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1))
        }

        print(
            f"{metric:22s}: "
            f"{values.mean():.4f} "
            f"+/- {values.std(ddof=1):.4f}"
        )

    report["results"][str(depth)] = {
        "summary": summary,
        "fold_results": fold_results
    }

REPORT.write_text(
    json.dumps(report, indent=4),
    encoding="utf-8"
)

print("\nReport saved:", REPORT)
print("Production artifacts were not modified.")
