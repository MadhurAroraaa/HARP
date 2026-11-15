import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.prepare_data import prepare_pipeline
from sklearn.ensemble import IsolationForest
from lstm_model import create_lstm_model, prepare_sequences
import numpy as np

# load processed data (pipeline already includes features)
df = prepare_pipeline("ml/data/location_aware_gis_leakage_dataset.csv")

features = [
    "pressure", "flow_rate", "temperature", "vibration",
    "pressure_diff", "flow_diff", "rolling_mean", "rolling_std", "pressure_rate"
]

# ---------- Isolation Forest ----------
if_model = IsolationForest(contamination=0.01, random_state=42)
if_model.fit(df[features])

df["if_anomaly"] = if_model.predict(df[features])
df["if_anomaly"] = df["if_anomaly"].map({1: 0, -1: 1})

# ---------- LSTM ----------
data = df[features].values
X = prepare_sequences(data)

lstm = create_lstm_model((X.shape[1], X.shape[2]))
lstm.fit(X, X[:, -1, 0], epochs=3, batch_size=32, verbose=0)

preds = lstm.predict(X)
errors = np.abs(preds.flatten() - X[:, -1, 0])

threshold = np.mean(errors) + 2*np.std(errors)

df["lstm_anomaly"] = 0
df.loc[10:, "lstm_anomaly"] = (errors > threshold).astype(int)

# ---------- HYBRID ----------
df["final_anomaly"] = (
    (df["if_anomaly"] == 1) | (df["lstm_anomaly"] == 1)
).astype(int)

# output
print(df[["if_anomaly", "lstm_anomaly", "final_anomaly"]].head())
print("Total anomalies:", df["final_anomaly"].sum())