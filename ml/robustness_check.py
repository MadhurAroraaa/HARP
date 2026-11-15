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


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "location_aware_gis_leakage_dataset.csv"
REPORT_PATH = ROOT / "robustness_report.json"

FEATURES = [
    "Pressure",
    "Flow_Rate",
    "Temperature",
    "Vibration"
]

SEEDS = [42, 123, 2026]
N_ESTIMATORS = 300
MAX_FPR = 0.05


def calculate_metrics(y, scores, threshold):
    pred = (scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) else 0

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(
            average_precision_score(y, scores)
        )
    }


def select_threshold(y, scores):
    candidates = np.unique(np.concatenate([
        [np.nextafter(scores.min(), -np.inf)],
        scores,
        [np.nextafter(scores.max(), np.inf)]
    ]))

    eligible = []

    for threshold in candidates:
        result = calculate_metrics(y, scores, threshold)

        if result["false_positive_rate"] <= MAX_FPR:
            eligible.append((threshold, result))

    if not eligible:
        raise RuntimeError("No threshold meets the FPR constraint.")

    # Maximize recall, then precision, then F1.
    return max(
        eligible,
        key=lambda item: (
            item[1]["recall"],
            item[1]["precision"],
            item[1]["f1"],
            item[0]
        )
    )[0]


def build_supervised_models(seed):
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=seed
            )
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=N_ESTIMATORS,
            class_weight="balanced",
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1
        )
    }


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\n========== LOADING DATA ==========")

df = pd.read_csv(DATA_PATH)

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
print("Normal:", int((y == 0).sum()))
print("Leaks:", int((y == 1).sum()))
print("Locations:", groups.nunique())


# ------------------------------------------------------------
# REPEATED GROUPED CROSS-VALIDATION
# ------------------------------------------------------------

results = {
    name: []
    for name in [
        "Isolation Forest",
        "Logistic Regression",
        "Random Forest",
        "Extra Trees"
    ]
}

total_folds = 0

for seed in SEEDS:

    outer = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    for fold, (train_val_idx, test_idx) in enumerate(
        outer.split(X, y, groups), start=1
    ):

        total_folds += 1

        inner = StratifiedGroupKFold(
            n_splits=4,
            shuffle=True,
            random_state=seed + fold
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

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx].to_numpy()

        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx].to_numpy()

        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx].to_numpy()

        print(
            f"\nSeed {seed} | Fold {fold}/5 | "
            f"Train {len(train_idx)} | "
            f"Validation {len(val_idx)} | "
            f"Test {len(test_idx)}"
        )

        # ----------------------------------------------------
        # ISOLATION FOREST
        # ----------------------------------------------------

        normal_train = X_train.iloc[
            np.flatnonzero(y_train == 0)
        ]

        scaler = StandardScaler()
        X_normal_scaled = scaler.fit_transform(normal_train)

        model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination="auto",
            random_state=seed,
            n_jobs=-1
        )

        model.fit(X_normal_scaled)

        val_scores = -model.score_samples(
            scaler.transform(X_val)
        )

        test_scores = -model.score_samples(
            scaler.transform(X_test)
        )

        threshold = select_threshold(y_val, val_scores)

        result = calculate_metrics(
            y_test, test_scores, threshold
        )

        result.update({
            "seed": seed,
            "fold": fold,
            "threshold": float(threshold)
        })

        results["Isolation Forest"].append(result)

        # ----------------------------------------------------
        # SUPERVISED CLASSIFIERS
        # ----------------------------------------------------

        for name, model in build_supervised_models(seed).items():

            model.fit(X_train, y_train)

            val_scores = model.predict_proba(X_val)[:, 1]
            test_scores = model.predict_proba(X_test)[:, 1]

            threshold = select_threshold(y_val, val_scores)

            result = calculate_metrics(
                y_test, test_scores, threshold
            )

            result.update({
                "seed": seed,
                "fold": fold,
                "threshold": float(threshold)
            })

            results[name].append(result)

            print(
                f"  {name}: "
                f"Recall={result['recall']:.3f}, "
                f"Precision={result['precision']:.3f}, "
                f"FPR={result['false_positive_rate']:.3f}"
            )


# ------------------------------------------------------------
# AGGREGATE RESULTS
# ------------------------------------------------------------

print("\n========== ROBUSTNESS SUMMARY ==========")

report = {
    "experiment": "3-repeat, 5-fold grouped cross-validation",
    "features": FEATURES,
    "seeds": SEEDS,
    "total_test_folds": total_folds,
    "threshold_selection": "Validation recall maximized at FPR <= 5%",
    "models": {}
}

metric_names = [
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "roc_auc",
    "average_precision"
]

for name, fold_results in results.items():

    summary = {}

    print(f"\n{name}")

    for metric in metric_names:
        values = np.array([
            row[metric] for row in fold_results
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

    report["models"][name] = {
        "summary": summary,
        "fold_results": fold_results
    }


REPORT_PATH.write_text(
    json.dumps(report, indent=4),
    encoding="utf-8"
)

print("\nReport saved:", REPORT_PATH)
print("Production artifacts were NOT modified.")
