/**
 * test-ws.js
 *
 * Quick local test script — connects to the WebSocket, prints every message,
 * and runs a few REST endpoint checks. No frontend needed.
 *
 * Usage:  node test-ws.js
 * Testing Websockets
 */

import WebSocket from "ws";

const WS_URL   = "ws://localhost:3000";
const REST_URL = "http://localhost:3000";

let msgCount = 0;
const MAX_MESSAGES = 10;

// ─── WebSocket Test ───────────────────────────────────────────────────────────

console.log("\n=== WebSocket Test ===");
console.log(`Connecting to ${WS_URL}...\n`);

const ws = new WebSocket(WS_URL);

ws.on("open", () => {
  console.log("✅ Connected to WebSocket\n");
});

ws.on("message", (data) => {
  const payload = JSON.parse(data);
  msgCount++;

  if (payload.type === "init") {
    console.log("📡 INIT message received — pipeline graph:");
    console.log(`   Nodes : ${payload.graph.nodes.length}`);
    console.log(`   Edges : ${payload.graph.edges.length}\n`);
    return;
  }

  if (msgCount <= MAX_MESSAGES) {
    console.log(`📦 Tick #${msgCount}`);
    console.log(`   timestamp     : ${payload.timestamp}`);
    console.log(`   anomaly_score : ${payload.anomaly_score?.toFixed(6)}`);
    console.log(`   threshold     : ${payload.threshold?.toFixed(6)}`);
    console.log(`   leak_detected : ${payload.leak_detected}`);

    if (payload.leak_detected) {
      console.log(`   ⚠️  leak_node      : ${payload.leak_node}`);
      console.log(`   ⚠️  severity       : ${payload.severity}`);
      console.log(`   ⚠️  estimatedLoss  : ${payload.estimatedLossLpm} L/min`);
      console.log(`   ⚠️  bypass path    : ${payload.bypass?.pathDisplay}`);
      console.log(`   ⚠️  bypass status  : ${payload.bypass?.status}`);
    }
    console.log();
  }

  if (msgCount === MAX_MESSAGES) {
    console.log(`--- Received ${MAX_MESSAGES} ticks, stopping WebSocket output ---\n`);
    ws.close();
    runRestTests(); // kick off REST checks after WS is verified
  }
});

ws.on("close", () => {
  console.log("🔌 WebSocket closed\n");
});

ws.on("error", (err) => {
  console.error("❌ WebSocket error:", err.message);
  console.error("   Is the server running? (npm run dev)\n");
  process.exit(1);
});

// ─── REST Tests ───────────────────────────────────────────────────────────────

async function runRestTests() {
  console.log("=== REST Endpoint Tests ===\n");

  // 1. GET /api/health
  try {
    const res  = await fetch(`${REST_URL}/api/health`);
    const data = await res.json();
    console.log("GET /api/health");
    console.log(`   express    : ${data.express}`);
    console.log(`   mlService  : ${data.mlService}\n`);
  } catch (e) {
    console.error("❌ /api/health failed:", e.message, "\n");
  }

  // 2. GET /api/graph
  try {
    const res  = await fetch(`${REST_URL}/api/graph`);
    const data = await res.json();
    console.log("GET /api/graph");
    console.log(`   nodes : ${data.nodes.length}`);
    console.log(`   edges : ${data.edges.length}\n`);
  } catch (e) {
    console.error("❌ /api/graph failed:", e.message, "\n");
  }

  // 3. GET /api/status
  try {
    const res  = await fetch(`${REST_URL}/api/status`);
    const data = await res.json();
    console.log("GET /api/status");
    console.log(`   uptime           : ${data.uptime}s`);
    console.log(`   connectedClients : ${data.connectedClients}\n`);
  } catch (e) {
    console.error("❌ /api/status failed:", e.message, "\n");
  }

  // 4. POST /api/bypass  (test with node 3 as the leak)
  try {
    const res  = await fetch(`${REST_URL}/api/bypass`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ leakNode: 3, source: 0, target: 5, requiredFlow: 60 }),
    });
    const data = await res.json();
    console.log("POST /api/bypass  (leakNode=3)");
    console.log(`   feasible          : ${data.feasible}`);
    console.log(`   path              : ${data.pathDisplay}`);
    console.log(`   bottleneck        : ${data.bottleneckCapacity} L/min`);
    console.log(`   status            : ${data.status}\n`);
  } catch (e) {
    console.error("❌ /api/bypass failed:", e.message, "\n");
  }

  console.log("=== All tests done ===\n");
}