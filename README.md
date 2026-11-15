# 🛡️ Pipeline Sentinel — Real-Time Leak Detection & Intelligent Rerouting System

![Status](https://img.shields.io/badge/status-active-success)
![AI](https://img.shields.io/badge/AI-Hybrid%20Model-blue)
![Model](https://img.shields.io/badge/Model-LSTM%20%2B%20Isolation%20Forest-purple)
![Backend](https://img.shields.io/badge/backend-Node.js%20%2B%20Express-green)
![Frontend](https://img.shields.io/badge/frontend-React%20%2B%20Vite-orange)
![ML](https://img.shields.io/badge/ML%20service-Python%20FastAPI-yellow)
![Graph](https://img.shields.io/badge/routing-Dijkstra%20Algorithm-red)

---

## 🌍 Overview

**Pipeline Sentinel** is an advanced real-time infrastructure monitoring platform that detects pipeline leaks, localises faults to specific network nodes, and automatically computes bypass routing paths — all within seconds of anomaly onset.

The system combines a **hybrid AI detection engine** (LSTM Autoencoder + Isolation Forest) with a **Dijkstra-based graph routing engine**, streaming live sensor data through a WebSocket-connected full-stack architecture. It is designed for critical infrastructure scenarios where response latency directly translates to human safety and economic impact.

---

## 💀 Why This Exists — Real-World Catastrophes

Pipeline failures are not hypothetical. They are recurring, catastrophic events with consequences measured in human lives, environmental devastation, and geopolitical destabilisation. The case for real-time automated detection and rerouting is written in history.

### ☢️ Bhopal Gas Leak — December 1984, India
The Union Carbide pesticide plant in Bhopal, Madhya Pradesh suffered a catastrophic failure when water entered a storage tank containing methyl isocyanate (MIC), triggering a runaway exothermic reaction. Over **40 tonnes of toxic gas** were released into a densely populated area. Conservative estimates place the immediate death toll at **3,787**, with later government estimates exceeding **15,000** fatalities and over **500,000** survivors suffering permanent injuries including blindness, respiratory failure, and neurological damage.

**What went wrong:** Multiple safety systems were offline or malfunctioning. There was no real-time anomaly detection on tank pressure and temperature. There was no automated rerouting or emergency isolation protocol. Manual monitoring failed entirely. A system like Pipeline Sentinel — monitoring pressure differential, temperature spikes, and flow anomalies in real time — would have flagged the event minutes before criticality, triggering isolation before the release became unsurvivable.

### ⚔️ Abadan Refinery Attacks — Iran, 1980–1988
During the Iran-Iraq War, the Abadan refinery — once the world's largest — was repeatedly struck by Iraqi air and artillery strikes beginning in September 1980. Pipeline infrastructure connecting the refinery to export terminals was systematically destroyed. Iran lost an estimated **$50 billion** in oil infrastructure and production capacity. Fires burned for months. The inability to rapidly detect damaged sections and reroute flow to intact pipeline segments compounded losses far beyond the initial strike damage.

More recently, the September 2019 drone and cruise missile strikes on Saudi Aramco's **Abqaiq processing facility and Khurais oil field** — attributed to Houthi forces aligned with Iran — temporarily knocked out approximately **5.7 million barrels per day** of Saudi oil production, representing roughly **5% of global daily supply**. Smoke was visible from satellite imagery for days. The facility had no real-time automated rerouting capability capable of isolating damaged segments and redirecting flow through intact infrastructure within the critical first hour.

**What Pipeline Sentinel addresses:** Rapid automated detection of pressure loss and flow anomaly at any node, immediate graph-theoretic computation of bypass paths through undamaged segments, and sub-second broadcasting of rerouting instructions to operators — capabilities that directly reduce the window between strike and catastrophic total loss.

### 🌊 Colonial Pipeline Ransomware Attack — May 2021, USA
A ransomware attack on the Colonial Pipeline's IT systems caused operators to **manually shut down 5,500 miles** of pipeline carrying 45% of the US East Coast's fuel supply. The shutdown lasted six days, causing widespread fuel shortages across the southeastern United States. The FBI attributed the attack to the DarkSide ransomware group. The operator paid approximately **$4.4 million** in ransom.

The core failure was not just a cybersecurity gap — it was the absence of a real-time operational monitoring layer that could continue functioning independently of compromised IT systems, maintaining pipeline integrity visibility and rerouting capability even under adversarial conditions.

---

## ⚡ Key Features

| Feature | Description |
|---|---|
| 🔍 Real-time Leak Detection | Continuous 600ms-interval monitoring of pressure, flow rate, temperature, and vibration |
| 📍 Node-level Fault Localisation | Pinpoints the specific pipeline network node experiencing failure |
| 🧠 Hybrid AI Model | Dual-layer: LSTM Autoencoder (temporal) + Isolation Forest (statistical) |
| 🛣️ Dynamic Rerouting | Dijkstra shortest-path engine finds optimal bypass around any failed node |
| 📊 Live Dashboard | Real-time WebSocket-fed React dashboard with SVG pipeline topology visualisation |
| ⚠️ Severity Classification | Critical / Moderate / Minor tiering with estimated flow loss in L/min |
| 🔄 Auto-reconnect WebSocket | Frontend reconnects automatically if connection drops |
| 📡 REST + WebSocket API | Full REST API alongside real-time WebSocket data channel |
| 🧪 Synthetic Data Mode | Backend streams realistic synthetic sensor data when no CSV is present |

---

## 🧠 Hybrid AI Model Architecture

Traditional systems rely on a single detection methodology. Pipeline Sentinel uses a two-layer hybrid approach that catches anomalies both instantaneously and over time.

### Layer 1 — LSTM Autoencoder (Temporal Learning)
The LSTM Autoencoder is trained on normal pipeline behaviour sequences. During inference, it attempts to reconstruct the incoming sensor window. A high **reconstruction error** indicates the current pattern deviates from learned normal behaviour — signalling a potential anomaly developing over time.

- Input: sliding window of 30 rows × 4 features
- Captures slow-developing anomalies: gradual pressure decay, progressive flow reduction
- Strong against gradual corrosion-induced leaks and thermal expansion failures

### Layer 2 — Isolation Forest (Statistical Outlier Detection)
The Isolation Forest algorithm identifies anomalous readings by isolating data points that require fewer splits in a random partition tree — outliers are isolated faster than normal points.

- Input: same 30-row feature window with engineered features
- Captures sudden anomalies: burst failures, valve ruptures, strike damage
- Works well with high-dimensional, noisy sensor data

### Feature Engineering (Python FastAPI — server side)
The following features are computed from the raw 4-column window inside the ML service before inference. Node.js sends raw features only — the FastAPI service handles all engineering:

```python
def create_features(df):
    df["pressure_diff"]  = df["pressure"].diff()
    df["flow_diff"]      = df["flow_rate"].diff()
    df["rolling_mean"]   = df["pressure"].rolling(10).mean()
    df["rolling_std"]    = df["pressure"].rolling(10).std()
    df["pressure_rate"]  = df["pressure"].pct_change()
    df.fillna(0, inplace=True)
    return df
```

### Combined Output
Both model scores are fused into a single `anomaly_score`. When this score exceeds the learned `threshold`, `leak_detected: true` is returned alongside:
- `confidence` — normalised [0,1] model certainty
- `leak_node` — pipeline node ID where the fault is localised
- `anomaly_score` and `threshold` — raw values for dashboard display

---

## 🛣️ Graph Routing Engine

When a leak is detected at a node, that node is removed from the pipeline graph and Dijkstra's shortest path algorithm computes the optimal bypass route from source to delivery point through remaining intact segments.

**Edge weight:** `length / diameter` — shorter, wider pipes are preferred.

**Bottleneck capacity:** The minimum-capacity edge along the bypass path is reported. If it falls below required flow (default 60 L/min), a `capacity_warning` is issued.

**Node resolution order** (for leak localisation):
1. FastAPI returns `leak_node` directly — used as-is (any integer, no constraints)
2. FastAPI returns `leak_sensor_id` — looked up in `SENSOR_NODE_MAP`
3. FastAPI returns `leak_feature` (column index) — looked up in `FEATURE_NODE_MAP`

The graph is not constrained to any fixed number of nodes and can represent pipelines of arbitrary size. Replace the `PIPELINE_GRAPH` object in `graphRouter.js` with a JSON loader for production-scale deployments.

---

## 📂 Project Structure

```
Pipeline-Leak-Detection-System/
│
├── Backend/                        # Node.js + Express + WebSocket
│   ├── routes/
│   │   ├── graph.js                # GET /api/graph, POST /api/bypass
│   │   └── status.js               # GET /api/status (uptime, WS clients)
│   │
│   ├── services/
│   │   ├── dataStreamer.js          # Sensor tick loop — CSV or synthetic
│   │   ├── graphRouter.js           # Dijkstra bypass routing engine
│   │   └── mlClient.js             # Axios client for Python FastAPI
│   │
│   ├── server.js                   # Express + WebSocket server (port 3000)
│   ├── test-ws.js                  # WebSocket + REST integration test
│   ├── package.json
│   └── package-lock.json
│
├── Frontend/
│   └── vihaan/
│       ├── public/
│       ├── src/
│       │   ├── App.jsx             # Main real-time dashboard
│       │   ├── Landing.jsx         # Landing page
│       │   ├── main.jsx            # React entry point
│       │   ├── App.css
│       │   └── index.css
│       ├── index.html
│       ├── vite.config.js
│       ├── package.json
│       └── package-lock.json
│
├── ml/
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py                  # FastAPI service (port 8000)
│   │
│   ├── data/
│   │   ├── clean_data.py
│   │   ├── data_loader.py
│   │   ├── prepare_data.py
│   │   ├── location_aware_gis.py   # GIS-based sensor-to-node mapping
│   │   └── test.py
│   │
│   ├── features/
│   │   └── feature_engineering.py  # create_features() — diff, rolling, pct_change
│   │
│   └── model/
│       ├── anomaly_model.py        # Isolation Forest logic
│       ├── lstm_model.py           # LSTM Autoencoder
│       ├── run_model.py            # Inference pipeline
│       ├── model.pkl               # Trained model weights
│       ├── scaler.pkl              # Feature scaler
│       └── threshold.npy           # Decision threshold
│
└── README.md
```

---

## 🔌 Data Flow

```
CSV / Synthetic Data
        ↓  600ms tick
  dataStreamer.js
  [pressure, flow_rate, temperature, vibration]  ← raw 4 features per row
        ↓  sliding window (30 rows × 4 features)
  mlClient.js  →  POST /predict  →  FastAPI (port 8000)
                                          ↓
                                  create_features()
                                  LSTM Autoencoder
                                  Isolation Forest
                                          ↓
                              { anomaly_score, threshold,
                                confidence, leak_detected,
                                leak_node }
        ↓
  dataStreamer.js
  [if leak_detected] → findBypassPath(leak_node) → Dijkstra
        ↓
  broadcast(payload)  →  WebSocket  →  React Dashboard
```

---

## 📡 API Reference

### WebSocket — `ws://localhost:3000`

**On connection:** Server immediately sends:
```json
{ "type": "init", "graph": { "nodes": [...], "edges": [...] } }
```

**Every 600ms tick:**
```json
{
  "timestamp": 1718000000000,
  "readings": {
    "pressure": 98.4,
    "flow_rate": 52.1,
    "temperature": 71.3,
    "vibration": 0.42
  },
  "anomaly_score": 0.000412,
  "threshold": 0.000380,
  "confidence": 0.73,
  "leak_detected": true,
  "leak_node": 4,
  "severity": "moderate",
  "estimatedLossLpm": 38,
  "bypass": {
    "feasible": true,
    "path": [0, 6, 7, 5],
    "pathDisplay": "0 → 6 → 7 → 5",
    "hops": 3,
    "bottleneckCapacity": 80,
    "capacityWarning": false,
    "status": "optimal",
    "edgesUsed": ["e4", "e5", "e6"]
  }
}
```

### REST Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/graph` | Full pipeline graph (nodes + edges) |
| `POST` | `/api/bypass` | Compute bypass for given `{ leakNode, source, target, requiredFlow }` |
| `GET` | `/api/health` | Express status + ML service reachability |
| `GET` | `/api/status` | Server uptime, timestamp, connected WS clients |

---

## 🚀 Setup & Running

### Prerequisites
- Node.js v18+
- Python 3.9+
- npm

### 1. Clone the repository
```bash
git clone https://github.com/binaryperson/vihaanieee.git
cd vihaanieee
```

### 2. Start the ML service (Python FastAPI — port 8000)
```bash
cd ml
pip install fastapi uvicorn numpy pandas scikit-learn tensorflow
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### 3. Start the backend (Node.js — port 3000)
```bash
cd Backend
npm install
npm run dev       # or: npm start
```

> If `data/sensor_data.csv` is present, it will stream from the CSV.
> If not, it falls back to synthetic 4-feature sensor data automatically.

### 4. Start the frontend (React + Vite — port 5173)
```bash
cd Frontend/vihaan
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

### 5. Test the backend independently (no browser needed)
```bash
cd Backend
node test-ws.js
```
This connects via WebSocket, prints 10 live ticks, then runs all REST endpoint checks.

---

## 🧪 Testing Without ML Service

The system is designed to operate gracefully without the Python ML service running:

- `dataStreamer.js` catches all ML errors per tick and continues streaming with `leak_detected: false`
- `mlClient.js` returns `null` on any network failure — the stream never crashes
- The backend can be fully tested using `test-ws.js` with synthetic data and no FastAPI instance

---

## ⚙️ Configuration Reference

### Backend — `services/dataStreamer.js`

| Constant | Default | Description |
|---|---|---|
| `WINDOW_SIZE` | `30` | Rows sent to ML per prediction — must match training |
| `TICK_INTERVAL_MS` | `600` | Milliseconds between sensor ticks |
| `FEATURE_COLS` | `["pressure","flow_rate","temperature","vibration"]` | Column order sent to FastAPI |
| `SEVERITY.critical` | `0.8` | Confidence threshold for critical severity |
| `SEVERITY.moderate` | `0.5` | Confidence threshold for moderate severity |

### Backend — `services/graphRouter.js`

| Constant | Default | Description |
|---|---|---|
| `PIPELINE_GRAPH` | Demo 10-node graph | Replace with real topology for production |
| `requiredFlow` | `60` L/min | Minimum flow for bypass capacity warning |

### Backend — `services/mlClient.js`

| Constant | Default | Description |
|---|---|---|
| `ML_BASE_URL` | `http://localhost:8000` | FastAPI service URL |
| `PREDICT_TIMEOUT` | `5000` ms | Per-request timeout for `/predict` |
| `HEALTH_TIMEOUT` | `3000` ms | Per-request timeout for `/health` |

---

## 🗺️ Scaling to Real Pipelines

The prototype uses a 10-node demo graph. For production:

1. **Replace `PIPELINE_GRAPH`** in `graphRouter.js` with a JSON loader:
   ```js
   import { readFileSync } from "fs";
   export const PIPELINE_GRAPH = JSON.parse(readFileSync("./data/pipeline.json", "utf8"));
   ```

2. **Populate `SENSOR_NODE_MAP`** in `dataStreamer.js` with your real sensor-to-node deployment:
   ```js
   const SENSOR_NODE_MAP = {
     "P-101": 14,   // pressure sensor at node 14
     "F-202": 27,   // flow sensor at node 27
   };
   ```

3. **Update `SYNTHETIC_RANGES`** to match your actual hardware sensor ranges if using synthetic mode.

4. **Calibrate severity thresholds** after initial real-world model integration — `SEVERITY.critical` and `SEVERITY.moderate` in `dataStreamer.js` depend on your model's actual confidence score distribution.

---

## 🤝 Team

| Domain | Stack |
|---|---|
| ML / Data Science | Python, TensorFlow/Keras, scikit-learn, pandas, FastAPI |
| Backend | Node.js, Express, WebSocket (`ws`), Axios |
| Frontend | React, Vite, SVG |
| Graph Routing | Dijkstra (custom ES module implementation) |

---

## 📄 License

This project is developed as a prototype for academic and demonstration purposes.
# HARP
