
"""
HARP — Hybrid Anomaly Recognition and Path Reconfiguration
Machine Learning API

Current live detection model: Isolation Forest.
LSTM integration is a separate future development task.

Input format:
{
    "window": [
        [pressure, flow_rate, temperature, vibration],
        ...
    ]
}
"""

from __future__ import annotations
from features.feature_engineering import create_features
from data.data_loader import load_data

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats as sp_stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# 1. PROJECT PATHS AND LOCAL IMPORTS
# ============================================================

ML_DIR = Path(__file__).resolve().parents[1]

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))


DATA_DIR = ML_DIR / "data"

DATA_PATH = DATA_DIR / "location_aware_gis_leakage_dataset.csv"

MODEL_PATH = ML_DIR / "model.pkl"
SCALER_PATH = ML_DIR / "scaler.pkl"
THRESHOLD_PATH = ML_DIR / "threshold.npy"
METADATA_PATH = ML_DIR / "model_metadata.json"


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | HARP-ML | %(levelname)s | %(message)s"
)

logger = logging.getLogger("HARP-ML")


# ============================================================
# 3. MODEL CONFIGURATION
# ============================================================

INPUT_COLUMNS = [
    "pressure",
    "flow_rate",
    "temperature",
    "vibration"
]

FEATURES = [
    "pressure",
    "flow_rate",
    "temperature",
    "vibration",
    "pressure_diff",
    "flow_diff",
    "rolling_mean",
    "rolling_std",
    "pressure_rate"
]

WINDOW_SIZE = 30
MIN_WINDOW_SIZE = 10
MAX_WINDOW_SIZE = 1000
MIN_TRAINING_ROWS = 30

CONTAMINATION = 0.01
N_ESTIMATORS = 200
RANDOM_STATE = 42

THRESHOLD_PERCENTILE = 99
ARTIFACT_SCHEMA_VERSION = 1

EPSILON = 1e-8


# ============================================================
# 4. MODEL STATE
# ============================================================

model = None
scaler = None
threshold: float | None = None
training_rows = 0
model_error: str | None = None


# ============================================================
# 5. REQUEST VALIDATION
# ============================================================

class PredictionRequest(BaseModel):
    window: list[list[float]] = Field(
        min_length=MIN_WINDOW_SIZE,
        max_length=MAX_WINDOW_SIZE
    )


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

def build_feature_frame(sensor_data: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw sensor readings into the exact nine features
    used during model training.

    IMPORTANT:
    Training and prediction must use the same transformations.
    """

    missing_columns = [
        col for col in INPUT_COLUMNS
        if col not in sensor_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing sensor columns: {missing_columns}"
        )

    df = sensor_data.loc[:, INPUT_COLUMNS].copy()

    for col in INPUT_COLUMNS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if df.isna().any().any():
        raise ValueError(
            "Sensor data contains missing or invalid readings."
        )

    # Apply the same feature engineering used during training.
    df = create_features(df.copy())

    # Handle infinity from percentage changes and
    # missing values from rolling calculations.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0.0)

    feature_frame = df.loc[:, FEATURES].astype(np.float64)

    if not np.isfinite(feature_frame.to_numpy()).all():
        raise ValueError(
            "Feature engineering produced non-finite values."
        )

    return feature_frame


# ============================================================
# 7. SAVE MODEL ARTIFACTS
# ============================================================

def save_artifacts(
    trained_model,
    trained_scaler,
    trained_threshold: float,
    rows: int
) -> None:
    """
    Saves model, scaler, threshold and metadata.

    Metadata makes it possible to detect incompatible
    or outdated model artifacts during future startups.
    """

    metadata = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "input_columns": INPUT_COLUMNS,
        "feature_names": FEATURES,
        "sklearn_version": sklearn.__version__,
        "training_rows": rows,
        "contamination": CONTAMINATION,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "threshold_method": "99th percentile of training anomaly scores",
        "model_type": "IsolationForest"
    }

    model_tmp = Path(str(MODEL_PATH) + ".tmp")
    scaler_tmp = Path(str(SCALER_PATH) + ".tmp")
    threshold_tmp = Path(str(THRESHOLD_PATH) + ".tmp")
    metadata_tmp = Path(str(METADATA_PATH) + ".tmp")

    # Remove the old metadata marker before replacing artifacts.
    METADATA_PATH.unlink(missing_ok=True)

    try:
        joblib.dump(trained_model, model_tmp)
        joblib.dump(trained_scaler, scaler_tmp)

        with threshold_tmp.open("wb") as file:
            np.save(file, trained_threshold)

        metadata_tmp.write_text(
            json.dumps(metadata, indent=4),
            encoding="utf-8"
        )

        os.replace(model_tmp, MODEL_PATH)
        os.replace(scaler_tmp, SCALER_PATH)
        os.replace(threshold_tmp, THRESHOLD_PATH)
        os.replace(metadata_tmp, METADATA_PATH)

    finally:
        for temporary_file in (
            model_tmp,
            scaler_tmp,
            threshold_tmp,
            metadata_tmp
        ):
            temporary_file.unlink(missing_ok=True)


# ============================================================
# 8. LOAD EXISTING MODEL
# ============================================================

def load_saved_artifacts() -> bool:
    """
    Loads artifacts only when their metadata confirms
    that they match the current feature configuration
    and scikit-learn version.
    """

    global model, scaler, threshold, training_rows

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        THRESHOLD_PATH,
        METADATA_PATH
    ]

    if not all(path.is_file() for path in required_files):
        logger.warning(
            "Complete compatible artifact set not found."
        )
        return False

    try:
        metadata = json.loads(
            METADATA_PATH.read_text(encoding="utf-8")
        )

        if metadata.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            logger.warning("Artifact schema mismatch.")
            return False

        if metadata.get("input_columns") != INPUT_COLUMNS:
            logger.warning("Input column configuration mismatch.")
            return False

        if metadata.get("feature_names") != FEATURES:
            logger.warning("Feature configuration mismatch.")
            return False

        if metadata.get("sklearn_version") != sklearn.__version__:
            logger.warning(
                "Scikit-learn version mismatch. Retraining required."
            )
            return False

        loaded_model = joblib.load(MODEL_PATH)
        loaded_scaler = joblib.load(SCALER_PATH)

        loaded_threshold = float(
            np.load(THRESHOLD_PATH, allow_pickle=False)
        )

        if not hasattr(loaded_model, "score_samples"):
            logger.warning("Invalid model artifact.")
            return False

        if getattr(loaded_model, "n_features_in_", None) != len(FEATURES):
            logger.warning("Model feature dimension mismatch.")
            return False

        if getattr(loaded_scaler, "n_features_in_", None) != len(FEATURES):
            logger.warning("Scaler feature dimension mismatch.")
            return False

        if not hasattr(loaded_scaler, "feature_names_in_"):
            logger.warning("Scaler feature metadata unavailable.")
            return False

        if list(loaded_scaler.feature_names_in_) != FEATURES:
            logger.warning("Scaler feature order mismatch.")
            return False

        if not np.isfinite(loaded_threshold) or loaded_threshold <= 0:
            logger.warning("Invalid saved threshold.")
            return False

        model = loaded_model
        scaler = loaded_scaler
        threshold = loaded_threshold
        training_rows = int(metadata.get("training_rows", 0))

        logger.info(
            "Model artifacts loaded successfully. Threshold: %.6f",
            threshold
        )

        return True

    except Exception:
        logger.exception(
            "Could not load model artifacts. Fresh training will be attempted."
        )
        return False


# ============================================================
# 9. TRAIN MODEL
# ============================================================

def train_on_startup() -> bool:
    """
    Trains Isolation Forest using the available pipeline dataset.

    Training and inference both use the same feature engineering.
    """

    global model, scaler, threshold, training_rows

    logger.info("Starting model training...")

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_PATH}"
        )

    logger.info("Loading dataset: %s", DATA_PATH.name)

    raw_df = load_data(DATA_PATH)

    missing_columns = [
        col for col in INPUT_COLUMNS
        if col not in raw_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    raw_df = raw_df.loc[:, INPUT_COLUMNS].copy()

    for col in INPUT_COLUMNS:
        raw_df[col] = pd.to_numeric(
            raw_df[col],
            errors="coerce"
        )

    raw_df = raw_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    original_rows = len(raw_df)

    raw_df = raw_df.dropna().reset_index(drop=True)

    dropped_rows = original_rows - len(raw_df)

    logger.info("Dataset rows: %d", original_rows)
    logger.info("Invalid rows removed: %d", dropped_rows)

    if len(raw_df) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Insufficient training data. "
            f"At least {MIN_TRAINING_ROWS} valid rows are required; "
            f"received {len(raw_df)}."
        )

    # Build the nine model features.
    feature_df = build_feature_frame(raw_df)

    X = feature_df

    logger.info(
        "Training Isolation Forest on %d rows and %d features.",
        len(X),
        len(FEATURES)
    )

    # Standardise features.
    trained_scaler = StandardScaler()

    X_scaled = trained_scaler.fit_transform(X)

    # Fit anomaly detection model.
    trained_model = IsolationForest(
        contamination=CONTAMINATION,
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    trained_model.fit(X_scaled)

    # Higher values represent more anomalous observations.
    scores = -trained_model.score_samples(X_scaled)

    if not np.isfinite(scores).all():
        raise ValueError(
            "Training produced invalid anomaly scores."
        )

    trained_threshold = float(
        np.percentile(scores, THRESHOLD_PERCENTILE)
    )

    if not np.isfinite(trained_threshold) or trained_threshold <= 0:
        raise ValueError("Invalid threshold generated during training.")

    # Persist the new compatible artifact set.
    save_artifacts(
        trained_model,
        trained_scaler,
        trained_threshold,
        len(raw_df)
    )

    # Update application state only after successful training and saving.
    model = trained_model
    scaler = trained_scaler
    threshold = trained_threshold
    training_rows = len(raw_df)

    logger.info("Training completed successfully.")
    logger.info("Training rows: %d", training_rows)
    logger.info("Threshold: %.6f", threshold)

    return True


# ============================================================
# 10. INITIALISE MODEL
# ============================================================

def initialise_model() -> None:
    global model_error

    model_error = None

    if load_saved_artifacts():
        logger.info("Using existing compatible model.")
        return

    logger.info(
        "Existing artifacts are missing or incompatible. Retraining."
    )

    try:
        train_on_startup()

    except Exception as exc:
        model_error = str(exc)

        logger.exception(
            "ML model initialisation failed."
        )


# ============================================================
# 11. FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting HARP ML service...")

    initialise_model()

    yield

    logger.info("Stopping HARP ML service...")


app = FastAPI(
    title="HARP — Machine Learning Service",
    description=(
        "Real-time pipeline anomaly detection, "
        "sensor analysis and fault localisation."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# 12. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


# ============================================================
# 13. PRE-FAILURE TREND ANALYSIS
# ============================================================

def detect_pressure_trend(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Detects statistically significant downward pressure trends.

    ETA is deliberately not calculated because the project
    does not yet define a validated critical-pressure threshold.
    """

    warnings = []

    pressure_series = df["pressure"].to_numpy(dtype=np.float64)

    if len(pressure_series) < MIN_WINDOW_SIZE:
        return warnings

    if not np.isfinite(pressure_series).all():
        return warnings

    try:
        slope, _, r_value, p_value, _ = sp_stats.linregress(
            np.arange(len(pressure_series)),
            pressure_series
        )

        if not all(
            np.isfinite(value)
            for value in (slope, r_value, p_value)
        ):
            return warnings

        r_squared = float(r_value ** 2)

        if (
            slope < -0.05
            and p_value < 0.05
            and r_squared > 0.5
        ):
            warnings.append({
                "node": 0,
                "feature": "pressure",
                "slope": round(float(slope), 4),
                "r2": round(r_squared, 3),
                "eta_seconds": None,
                "eta_note": (
                    "ETA not calculated: a validated critical "
                    "pressure threshold is not configured."
                )
            })

    except (ValueError, FloatingPointError):
        logger.exception("Pressure trend analysis failed.")

    return warnings


# ── POST /predict ──────────────────────────────────────────────

@app.post("/predict")
def predict(data: dict):
    global model, scaler, threshold

    # Check whether the model is available
    if model is None or scaler is None:
        return {
            "error": "ML model is not available",
            "leak_detected": False
        }

    # Get sensor window from Node.js
    window = data.get("window", [])

    # Validate minimum window size
    if not isinstance(window, list) or len(window) < 10:
        return {
            "anomaly_score": 0.0,
            "threshold": round(float(threshold), 6),
            "leak_detected": False,
            "leak_feature": 0,
            "confidence": 0.0,
            "error": "At least 10 sensor readings are required"
        }

    try:
        # Backend sends:
        # [pressure, flow_rate, temperature, vibration]
        df = pd.DataFrame(
            window,
            columns=INPUT_COLUMNS
        )

        # Convert values to numeric
        df = df.astype(float)

        # Reject invalid sensor readings
        if not np.isfinite(df.values).all():
            raise ValueError("Sensor readings contain invalid values")

        # Apply feature engineering
        df = create_features(df)

        # Keep the exact nine features used during training
        df = df[FEATURES]

        # Standardise features using the training scaler
        X = scaler.transform(df)

        # Calculate anomaly scores
        all_scores = -model.score_samples(X)

        # Latest reading represents the current sensor state
        latest_score = float(all_scores[-1])
        anomaly_score = round(latest_score, 6)

        # Compare against trained threshold
        leak_detected = anomaly_score > threshold

        # Anomaly ratio (not a calibrated probability)
        confidence = round(
            min(
                max(anomaly_score, 0.0) / (threshold + 1e-8),
                3.0
            ),
            3
        )

        # Calculate per-feature anomaly contributions
        feature_errors = (X[-1] ** 2).tolist()

        # Localise using only the four original sensor features
        # Indices correspond to FEATURE_NODE_MAP in Node.js
        leak_feature = int(
            np.argmax(feature_errors[:len(INPUT_COLUMNS)])
        )

        # Detect statistically significant pressure decline
        pre_failure_warnings = []

        pressure_series = df["pressure"].values

        if len(pressure_series) >= 10:
            slope, _, r_value, pval, _ = sp_stats.linregress(
                range(len(pressure_series)),
                pressure_series
            )

            if (
                slope < -0.05
                and pval < 0.05
                and r_value ** 2 > 0.5
            ):
                pre_failure_warnings.append({
                    "node": 0,
                    "feature": "pressure",
                    "slope": round(float(slope), 4),
                    "r2": round(float(r_value ** 2), 3),
                    "eta_seconds": None
                })

        # Return prediction result
        return {
            "anomaly_score": anomaly_score,
            "threshold": round(float(threshold), 6),
            "leak_detected": bool(leak_detected),
            "leak_feature": leak_feature,
            "confidence": confidence,
            "confidence_interpretation": (
                "Anomaly ratio, not a calibrated probability"
            ),
            "feature_errors": feature_errors,
            "feature_names": FEATURES,
            "pre_failure_warnings": pre_failure_warnings,
            "dynamic_threshold_active": False
        }

    except Exception as exc:
        logger.exception("Prediction failed")

        return {
            "error": str(exc),
            "leak_detected": False
        }


# ============================================================
# 15. GET /health
# ============================================================

@app.get("/health")
def health():

    model_ready = (
        model is not None
        and scaler is not None
        and threshold is not None
    )

    return {
        "service": "HARP ML Service",

        "status": "ok" if model_ready else "untrained",

        "model_ready": model_ready,

        "model_type": "IsolationForest",

        "threshold": (
            round(float(threshold), 6)
            if threshold is not None
            else None
        ),

        "threshold_method": (
            "99th percentile of training anomaly scores"
        ),

        "n_features": len(FEATURES),

        "input_columns": INPUT_COLUMNS,

        "training_rows": training_rows,

        "error": model_error
    }


# ============================================================
# 16. GET /debug/features
# ============================================================

@app.get("/debug/features")
def debug_features():

    return {
        "input_columns": INPUT_COLUMNS,

        "feature_names": FEATURES,

        "n_input_features": len(INPUT_COLUMNS),

        "n_model_features": len(FEATURES),

        "window_size": WINDOW_SIZE,

        "minimum_window_size": MIN_WINDOW_SIZE,

        "model_ready": (
            model is not None
            and scaler is not None
        ),

        "threshold": (
            round(float(threshold), 6)
            if threshold is not None
            else None
        ),

        "model_type": "IsolationForest",

        "sklearn_version": sklearn.__version__,

        "numpy_version": np.__version__,

        "pandas_version": pd.__version__,

        "scipy_version": scipy.__version__,

        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,

        "data_file_exists": DATA_PATH.is_file(),

        "model_artifact_exists": MODEL_PATH.is_file(),

        "scaler_artifact_exists": SCALER_PATH.is_file(),

        "threshold_artifact_exists": THRESHOLD_PATH.is_file(),

        "metadata_artifact_exists": METADATA_PATH.is_file()
    }
