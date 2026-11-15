/**
 * server.js
 *
 * Entry point for the pipeline leak-detection backend.
 *
 * Responsibilities:
 *   • Serves a REST API (Express) on port 3000.
 *   • Attaches a WebSocketServer to the *same* HTTP server (no extra port).
 *   • Broadcasts real-time sensor/anomaly payloads to all connected WS clients.
 *   • On WS connection, immediately sends an `init` message containing the full
 *     pipeline graph so clients can render the topology right away.
 *   • Starts the CSV-driven sensor data stream via dataStreamer.
 *
 * REST endpoints:
 *   GET  /api/graph         — Full pipeline graph
 *   POST /api/bypass        — Compute bypass path given a leaking node
 *   GET  /api/health        — Express + ML service health
 *   GET  /api/status        — Uptime, ti
 * mestamp, connected WS clients
 */

import { createServer }    from "http";
import express             from "express";
import cors                from "cors";
import { WebSocketServer } from "ws";

import { PIPELINE_GRAPH, findBypassPath } from "./services/graphRouter.js";
import { checkMLHealth }                  from "./services/mlClient.js";
import { startStreaming }                 from "./services/dataStreamer.js";
import graphRouter                        from "./routes/graph.js";
import statusRouter, { setWss }          from "./routes/status.js";

const PORT = Number(process.env.PORT) || 3000;
// ─── Express App ──────────────────────────────────────────────────────────────

const app    = express();
const server = createServer(app);

app.use(cors());
app.use(express.json());

// ─── WebSocket Server (shares the HTTP server) ────────────────────────────────

const wss = new WebSocketServer({ server });

// Give the status router a reference to the WS server for client counting
setWss(wss);

/**
 * broadcast(data)
 * Serialises `data` to JSON and sends it to every open WebSocket client.
 * Individual sends are wrapped in try/catch so a single dead socket never
 * breaks the loop.
 *
 * @param {object} data
 */
function broadcast(data) {
  const message = JSON.stringify(data);

  for (const client of wss.clients) {
    if (client.readyState === 1 /* WebSocket.OPEN */) {
      try {
        client.send(message);
      } catch (err) {
        console.warn(`[broadcast] Failed to send to client: ${err.message}`);
      }
    }
  }
}

// Handle new WebSocket connections
wss.on("connection", (ws, req) => {
  console.log(`[ws] Client connected from ${req.socket.remoteAddress}`);

  // Immediately push the pipeline graph so the client can render topology
  try {
    ws.send(JSON.stringify({ type: "init", graph: PIPELINE_GRAPH }));
  } catch (err) {
    console.warn(`[ws] Could not send init message: ${err.message}`);
  }

  ws.on("close", () => {
    console.log("[ws] Client disconnected");
  });

  ws.on("error", (err) => {
    console.error(`[ws] Socket error: ${err.message}`);
  });
});

// ─── REST Routes ──────────────────────────────────────────────────────────────

// Modular routers
app.use("/api/graph",  graphRouter);
app.use("/api/status", statusRouter);

// GET /api/graph — also reachable directly on the root router for convenience
app.get("/api/graph", (req, res) => {
  res.json(PIPELINE_GRAPH);
});

// POST /api/bypass
app.post("/api/bypass", (req, res) => {
  const {
    leakNode,
    source       = 0,
    target       = 5,
    requiredFlow = 60,
  } = req.body;

  if (leakNode === undefined || leakNode === null) {
    return res.status(400).json({ error: "leakNode is required." });
  }

  const result = findBypassPath(
    Number(leakNode),
    Number(source),
    Number(target),
    Number(requiredFlow)
  );

  res.json(result);
});

// GET /api/health
app.get("/api/health", async (req, res) => {
  const mlDetails = await checkMLHealth();

  res.json({
    express:   "ok",
    mlService: mlDetails ? "ok" : "unreachable",
    mlDetails: mlDetails ?? null,
  });
});

// ─── Start Sensor Stream ──────────────────────────────────────────────────────

startStreaming(broadcast);

// ─── Listen ───────────────────────────────────────────────────────────────────

server.listen(PORT, "0.0.0.0", () => {
  console.log(`[server] Express + WebSocket listening on http://localhost:${PORT}`);
  console.log(`[server] ML service expected at http://localhost:8000`);
});
