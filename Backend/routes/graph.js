/**
 * routes/graph.js
 *
 * Express router that exposes the pipeline graph over REST.
 *   GET  /api/graph         — Returns the full PIPELINE_GRAPH object.
 *   POST /api/graph/bypass  — Accepts { leakNode, source, target, requiredFlow }
 *                             and returns the computed bypass path.
 * this route is to find shortest route
 */


import { Router } from "express";
import { PIPELINE_GRAPH, findBypassPath } from "../services/graphRouter.js";

const router = Router();

// GET /api/graph
router.get("/", (req, res) => {
  res.json(PIPELINE_GRAPH);
});

// POST /api/graph/bypass
router.post("/bypass", (req, res) => {
  const {
    leakNode,
    source      = 0,
    target      = 5,
    requiredFlow = 60,
  } = req.body;

  if (leakNode === undefined || leakNode === null) {
    return res.status(400).json({ error: "leakNode is required in the request body." });
  }

  const result = findBypassPath(
    Number(leakNode),
    Number(source),
    Number(target),
    Number(requiredFlow)
  );

  res.json(result);
});

export default router;