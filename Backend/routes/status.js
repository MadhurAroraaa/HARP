/**
 * routes/status.js
 *
 * Express router that exposes server health/status information.
 *   GET /api/status — Returns server uptime (seconds), current ISO timestamp,
 *                     and the number of currently connected WebSocket clients.
 *
 */

import { Router } from "express";

const router = Router();

// Will be set by server.js after the WebSocketServer is created
let _wss = null;

/**
 * setWss(wss)
 * Called once from server.js to give this router a reference to the WS server. (mock web socket setup for testing)
 *
 * @param {import("ws").WebSocketServer} wss
 */
export function setWss(wss) {
  _wss = wss;
}

// GET /api/status
router.get("/", (req, res) => {
  const clientCount = _wss ? _wss.clients.size : 0;

  res.json({
    status:           "ok",
    uptime:           Math.floor(process.uptime()),
    timestamp:        new Date().toISOString(),
    connectedClients: clientCount,
  });
});

export default router;