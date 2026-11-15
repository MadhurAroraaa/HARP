import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data" / "location_aware_gis_leakage_dataset.csv"
REPORT = ROOT / "rule_stability_report.json"

FEATURES = [
    "Pressure",
    "Flow_Rate",
    "Temperature",
    "Vibration"
]

SEEDS = [42, 123, 2026]

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

rules = Counter()
split_counts = Counter()
thresholds = {feature: [] for feature in FEATURES}
importance_values = {feature: [] for feature in FEATURES}

fold = 0

for seed in SEEDS:

    outer = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    for fold_number, (train_val, test) in enumerate(
        outer.split(X, y, groups), start=1
    ):

        inner = StratifiedGroupKFold(
            n_splits=4,
            shuffle=True,
            random_state=seed + fold_number
        )

        train_rel, val_rel = next(
            inner.split(
                X.iloc[train_val],
                y.iloc[train_val],
                groups.iloc[train_val]
            )
        )

        train = train_val[train_rel]

        train_groups = set(groups.iloc[train])
        test_groups = set(groups.iloc[test])

        assert train_groups.isdisjoint(test_groups)

        model = DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=seed
        )

        model.fit(X.iloc[train], y.iloc[train])

        rule = export_text(
            model,
            feature_names=FEATURES,
            decimals=2
        )

        rules[rule] += 1

        for index, feature_index in enumerate(
            model.tree_.feature
        ):

            if feature_index >= 0:
                feature = FEATURES[feature_index]

                split_counts[feature] += 1

                thresholds[feature].append(
                    float(model.tree_.threshold[index])
                )

        for i, feature in enumerate(FEATURES):
            importance_values[feature].append(
                float(model.feature_importances_[i])
            )

        fold += 1

        print(f"Inspected fold {fold}/15")


print("\n========== FEATURE SPLIT FREQUENCY ==========")

for feature in FEATURES:
    print(
        feature,
        ":",
        split_counts[feature],
        "splits"
    )

print("\n========== FEATURE IMPORTANCE ==========")

importance_summary = {}

for feature in FEATURES:
    values = np.array(importance_values[feature])

    importance_summary[feature] = {
        "mean": float(values.mean()),
        "std": float(values.std())
    }

    print(
        f"{feature:15s}: "
        f"{values.mean():.4f} +/- {values.std():.4f}"
    )


print("\n========== UNIQUE LEARNED RULES ==========")

for index, (rule, count) in enumerate(
    rules.most_common(), start=1
):

    print(f"\nRULE PATTERN {index}: FOUND IN {count}/15 FOLDS")
    print(rule)


report = {
    "experiment": "Decision Tree Rule Stability",
    "folds": fold,
    "model": "DecisionTreeClassifier",
    "max_depth": 2,
    "min_samples_leaf": 20,
    "feature_split_frequency": dict(split_counts),
    "feature_importance": importance_summary,
    "unique_rule_patterns": [
        {
            "occurrences": count,
            "rule": rule
        }
        for rule, count in rules.most_common()
    ]
}

REPORT.write_text(
    json.dumps(report, indent=4),
    encoding="utf-8"
)

print("\nReport saved:", REPORT)
print("No production model artifacts were modified.")
