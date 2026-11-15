/**
 * services/dataStreamer.js
 *
 * ── Leak node localization ───────────────────────────────────────────────────
 * The ML model detects anomalies in sensor features (pressure, flow_rate,
 * temperature, vibration). Translating that to a physical pipeline node
 * requires a sensor deployment map: which node each sensor is installed at.
 *
 * Resolution order (first one that works wins):
 *   1. FastAPI returns `leak_node` directly (preferred — ML team uses training labels)
 *   2. FastAPI returns `leak_sensor_id` → looked up in SENSOR_NODE_MAP below
 *   3. FastAPI returns `leak_feature` (column index 0–3) → looked up in FEATURE_NODE_MAP
 *   4. Simulated: z-score deviation from baselines picks the most anomalous feature
 *
 * No modulo arithmetic. No hardcoded node count. Node IDs can be any value.
 *
 * ── Window shape sent to FastAPI ────────────────────────────────────────────
 *   { window: [[pressure, flow_rate, temperature, vibration], ...] }  →  30 × 4
 *   FastAPI runs create_features() internally before prediction.
 */

import { createReadStream, existsSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { parse } from "csv-parse";
import { predict } from "./mlClient.js";
import { findBypassPath } from "./graphRouter.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CSV_PATH = path.resolve(__dirname, "../data/location_aware_gis_leakage_dataset.csv");

const FEATURE_COLS = ["Pressure", "Flow_Rate", "Temperature", "Vibration"];

// ─── Config ───────────────────────────────────────────────────────────────────

const WINDOW_SIZE      = 30;   // rows — must match what ML team trained on
const TICK_INTERVAL_MS = 1000; // ms between ticks

// Non-sensor CSV columns to ignore
const SKIP_COLS = new Set(["timestamp", "label", "leak_node"]);

// ─── Sensor → Node mapping ────────────────────────────────────────────────────
const SENSOR_NODE_MAP = {
  // "P-101": 3,
  // "F-202": 7,
  // "T-305": 12,
};

const FEATURE_NODE_MAP = {
  0: 2,  // Pressure    → node 2
  1: 6,  // Flow_Rate   → node 6
  2: 4,  // Temperature → node 4
  3: 7,  // Vibration   → node 7
};

const SEVERITY = { critical: 2.0, moderate: 1.3 };

// ─── SYNTHETIC data ranges ────────────────────────────────────────────────────
// NOTE: This entire block is only used when no CSV is found.
// Comment it out or remove it once real sensor data is available.

// const SYNTHETIC = {
//   pressure:    { base: 100, spread: 10   },
//   flow_rate:   { base: 55,  spread: 8    },
//   temperature: { base: 70,  spread: 5    },
//   vibration:   { base: 0.4, spread: 0.15 },
// };

// ─── Helpers ──────────────────────────────────────────────────────────────────

// NOTE: rnd() is only used by syntheticRow() — can be removed with it.
// function rnd(base, spread) {
//   return parseFloat((base + (Math.random() - 0.5) * 2 * spread).toFixed(4));
// }

// NOTE: syntheticRow() generates fake sensor readings when no CSV is present.
// This is the core synthetic data generator — comment out to disable.
// function syntheticRow() {
//   return {
//     timestamp:   Date.now(),
//     pressure:    rnd(SYNTHETIC.pressure.base,    SYNTHETIC.pressure.spread),
//     flow_rate:   rnd(SYNTHETIC.flow_rate.base,   SYNTHETIC.flow_rate.spread),
//     temperature: rnd(SYNTHETIC.temperature.base, SYNTHETIC.temperature.spread),
//     vibration:   rnd(SYNTHETIC.vibration.base,   SYNTHETIC.vibration.spread),
//   };
// }

// ─── Simulated leak node ──────────────────────────────────────────────────────
//
// Used as Strategy 4 when ML returns no leak location info.
// Computes a z-score for each feature against its expected baseline.
// The feature with the highest deviation is treated as the anomaly source
// and mapped to a pipeline node via FEATURE_NODE_MAP.
//
// To calibrate: run this in Python against your real dataset:
//   df[["Pressure","Flow_Rate","Temperature","Vibration"]].agg(["mean","std"])
// Then paste the real values into BASELINES below.

const BASELINES = {
  Pressure:    { mean: 100, std: 10   },
  Flow_Rate:   { mean: 55,  std: 8    },
  Temperature: { mean: 70,  std: 5    },
  Vibration:   { mean: 0.4, std: 0.15 },
};

function simulateLeakNode(readings) {
  if (!readings) {
    console.warn("[dataStreamer] simulateLeakNode: no readings — defaulting to node 2");
    return 2;
  }

  // Compute z-score for each feature — highest absolute deviation wins
  const scores = FEATURE_COLS.map((col, idx) => {
    const baseline = BASELINES[col];
    if (!baseline) return { idx, col, score: 0 };
    const val   = readings[col] ?? baseline.mean;
    const score = Math.abs((val - baseline.mean) / baseline.std);
    return { idx, col, score };
  });

  scores.sort((a, b) => b.score - a.score);
  const dominant = scores[0];
  const node     = FEATURE_NODE_MAP[dominant.idx];

  console.log(
    `[dataStreamer] simulateLeakNode: dominant feature "${dominant.col}" ` +
    `(z=${dominant.score.toFixed(2)}) → node ${node}`
  );

  return node ?? 2; // node 2 as last resort
}

function toFeatureVector(row) {
  const vec = [];
  for (const col of FEATURE_COLS) {
    const val = Number(row[col]);
    if (row[col] === undefined || isNaN(val)) {
      console.warn(`[dataStreamer] Missing feature "${col}" — tick skipped`);
      return null;
    }
    vec.push(val);
  }
  return vec;
}

function resolveLeakNode(result) {
  // Strategy 1: FastAPI returns leak_node directly
  if (result.leak_node !== undefined && result.leak_node !== null) {
    return result.leak_node;
  }

  // Strategy 2: FastAPI returns a sensor ID string → look up in deployment map
  if (result.leak_sensor_id !== undefined) {
    const node = SENSOR_NODE_MAP[result.leak_sensor_id];
    if (node !== undefined) return node;
    console.warn(`[dataStreamer] leak_sensor_id "${result.leak_sensor_id}" not in SENSOR_NODE_MAP — add it`);
  }

  // Strategy 3: FastAPI returns feature column index → look up in feature map
  if (result.leak_feature !== undefined) {
    const node = FEATURE_NODE_MAP[result.leak_feature];
    if (node !== undefined) return node;
    console.warn(`[dataStreamer] leak_feature ${result.leak_feature} not in FEATURE_NODE_MAP`);
  }

  // Strategy 4 is handled at call site via simulateLeakNode()
  return null;
}

// ─── Entry point ──────────────────────────────────────────────────────────────

export function startStreaming(broadcast) {
  if (existsSync(CSV_PATH)) {
    console.log(`[dataStreamer] CSV found — loading ${CSV_PATH}`);
    loadCSV(broadcast);
  } else {
    // NOTE: This else branch only runs when CSV is missing.
    // It falls back to synthetic data — remove once real data pipeline is set up.
    console.warn("[dataStreamer] No CSV found — using synthetic 4-feature data");
    stream([], broadcast);
  }
}

// ─── CSV loader ───────────────────────────────────────────────────────────────

function loadCSV(broadcast) {
  const rows = [];
  createReadStream(CSV_PATH)
    .on("error", (err) => {
      // NOTE: On CSV read failure, falls back to synthetic — remove fallback once stable.
      console.error(`[dataStreamer] Cannot open CSV: ${err.message} — falling back to synthetic`);
      stream([], broadcast);
    })
    .pipe(parse({ columns: true, cast: true }))
    .on("data",  (row) => rows.push(row))
    .on("error", (err) => console.error(`[dataStreamer] CSV parse error: ${err.message}`))
    .on("end", () => {
      if (rows.length === 0) {
        // NOTE: Empty CSV also falls back to synthetic — remove once data is reliable.
        console.error("[dataStreamer] CSV empty — falling back to synthetic");
        stream([], broadcast);
        return;
      }
      console.log(`[dataStreamer] Loaded ${rows.length} rows. Stream starting…`);
      stream(rows, broadcast);
    });
}

// ─── Core tick loop ───────────────────────────────────────────────────────────

async function stream(csvRows, broadcast) {
  const slidingBuffer = [];
  let rowIndex = 0;
  const useCSV = csvRows.length > 0;

  async function tick() {
    let row, featureVec;

    if (useCSV) {
      row = csvRows[rowIndex];
      rowIndex = (rowIndex + 1) % csvRows.length;
      featureVec = toFeatureVector(row);
      if (!featureVec) { setTimeout(tick, TICK_INTERVAL_MS); return; }
    } else {
      // NOTE: This else block is the synthetic data path.
      // It runs when useCSV is false (no CSV loaded).
      // Remove or replace with a real data source when ready.
      // row = syntheticRow();
      // featureVec = FEATURE_COLS.map((f) => row[f]);
      console.error("[dataStreamer] No CSV and synthetic data is disabled — tick skipped");
      setTimeout(tick, TICK_INTERVAL_MS);
      return;
    }

    const timestamp = row.timestamp ?? Date.now();
    const readings  = Object.fromEntries(FEATURE_COLS.map((f, i) => [f, featureVec[i]]));

    slidingBuffer.push(featureVec);
    if (slidingBuffer.length > WINDOW_SIZE) slidingBuffer.shift();

    const payload = {
      timestamp,
      readings,
      anomaly_score: 0,
      threshold:     0,
      confidence:    0,
      leak_detected: false,
    };

    if (slidingBuffer.length === WINDOW_SIZE) {
      try {
        const result = await predict(slidingBuffer);

        if (result) {
          payload.anomaly_score = result.anomaly_score ?? 0;
          payload.threshold     = result.threshold     ?? 0;
          payload.confidence    = result.confidence    ?? 0;
          payload.leak_detected = result.leak_detected ?? false;

          if (payload.leak_detected) {
            // Try ML-provided node first (strategies 1–3),
            // fall back to z-score simulation (strategy 4)
            const leakNode = resolveLeakNode(result) ?? simulateLeakNode(readings);

            if (leakNode !== null) {
              payload.leak_node = leakNode;

              payload.severity =
                payload.confidence >= SEVERITY.critical ? "critical" :
                payload.confidence >= SEVERITY.moderate ? "moderate" : "minor";

              payload.estimatedLossLpm = Math.round(
                Math.max(0, payload.anomaly_score - payload.threshold) * 1200
              );

              payload.bypass = findBypassPath(leakNode);
            } else {
              payload.leak_node        = null;
              payload.severity         = "unknown";
              payload.estimatedLossLpm = 0;
              payload.bypass           = { feasible: false, reason: "Node could not be resolved from ML output" };
            }
          }
        }
      } catch (mlErr) {
        console.error(`[dataStreamer] ML tick error: ${mlErr.message}`);
      }
    }

    broadcast(payload);
    setTimeout(tick, TICK_INTERVAL_MS);
  }

  setTimeout(tick, TICK_INTERVAL_MS);
}