// /**
//  * services/mlClient.js
//  *
//  * Thin HTTP client that talks to the Python FastAPI ML service running on port 8000.
//  * Exposes two functions:
//  *   predict(window)    — POST a 30-reading sliding window, get anomaly scores back.
//  *   checkMLHealth()    — GET /health to verify the ML service is reachable.
//  *
//  * Both functions are wrapped in try/catch so callers never have to worry about
//  * network errors crashing the stream.
//  */

import axios from "axios";

const ML_BASE_URL = process.env.ML_BASE_URL || "http://localhost:8000";

// /**
//  * predict(window)
//  * Sends a sliding window of sensor readings to the ML service for anomaly detection.
//  *
//  * @param {Array<Array<number>>} window - Up to 30 rows of sensor values
//  * @returns {Promise<object|null>} Prediction result, or null on failure
//  *
//  * Expected response shape from FastAPI:
//  * {
//  *   anomaly_score: number,
//  *   threshold:     number,
//  *   confidence:    number,
//  *   leak_detected: boolean,
//  *   leak_feature:  number   // index hint used to derive leak_node
//  * }
//  */
export async function predict(window) {
  try {
  const response = await axios.post(`${ML_BASE_URL}/predict`, { window });
   return response.data;
  } catch (err) {
    const msg = err.response
      ? `ML /predict returned ${err.response.status}: ${JSON.stringify(err.response.data)}`
      : `ML /predict unreachable: ${err.message}`;
    console.error(`[mlClient] ${msg}`);
     return null;
   }
 }

// /**
//  * checkMLHealth()
//  * Probes the ML service health endpoint.
//  *
//  * @returns {Promise<object|null>} Health payload from the ML service, or null if down
//  */
export async function checkMLHealth() {
   try {
     const response = await axios.get(`${ML_BASE_URL}/health`, { timeout: 3000 });
     return response.data;
   } catch (err) {
     console.error(`[mlClient] Health check failed: ${err.message}`);
     return null;
   }
 }

// export async function predict(window) {
//   // Simulate occasional leak detection
//   const anomaly_score = Math.random() * 0.0008;
//   const threshold = 0.00038;
//   const leak_detected = anomaly_score > threshold;

//   return {
//     anomaly_score,
//     threshold,
//     confidence: leak_detected ? 1.0 + Math.random() * 1.5 : 0,
//     leak_detected,
//     leak_feature: Math.floor(Math.random() * 30), // maps to leak_node via % 10
//   };
// }

// export async function checkMLHealth() {
//   return { status: "mocked" };
// }