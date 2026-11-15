
/**
 * HARP - Pipeline Graph Router
 *
 * Responsibilities:
 * 1. Load pipeline topology from Backend/data/pipeline.json.
 * 2. Validate nodes, edges, capacities and dimensions.
 * 3. Find shortest feasible routes using Dijkstra's algorithm.
 * 4. Isolate a leaking node and all its incident edges.
 * 5. Exclude edges that cannot satisfy the required flow.
 * 6. Calculate the bottleneck capacity of the selected route.
 *
 * IMPORTANT:
 * The included fallback graph is simulated demonstration data.
 * It is not a real pipeline engineering topology.
 *
 * Edge weight:
 *     weight = length / diameter
 *
 * Routing objective:
 * Find the minimum-cost path among paths whose every edge
 * has capacity >= requiredFlow.
 *
 * This is NOT a maximum-flow algorithm.
 */

import { readFileSync, existsSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";

// ─── Paths ──────────────────────────────────────────────────────────────────

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const GRAPH_PATH = path.resolve(
  __dirname,
  "../data/pipeline.json"
);

// ─── Graph validation ───────────────────────────────────────────────────────

function validateGraph(graph) {
  if (!graph || typeof graph !== "object") {
    throw new Error("Pipeline graph must be a valid JSON object.");
  }

  if (!Array.isArray(graph.nodes) || graph.nodes.length === 0) {
    throw new Error("graph.nodes is missing or empty.");
  }

  if (!Array.isArray(graph.edges)) {
    throw new Error("graph.edges is missing or invalid.");
  }

  if (!Number.isInteger(graph.source)) {
    throw new Error("graph.source must be an integer.");
  }

  if (!Number.isInteger(graph.target)) {
    throw new Error("graph.target must be an integer.");
  }

  const nodeIds = new Set();

  // Validate nodes.
  for (const node of graph.nodes) {
    if (!node || !Number.isInteger(node.id)) {
      throw new Error(
        `Invalid node ID: ${JSON.stringify(node)}`
      );
    }

    if (nodeIds.has(node.id)) {
      throw new Error(`Duplicate node ID: ${node.id}`);
    }

    nodeIds.add(node.id);
  }

  // Validate source and target.
  if (!nodeIds.has(graph.source)) {
    throw new Error(
      `Source node ${graph.source} does not exist.`
    );
  }

  if (!nodeIds.has(graph.target)) {
    throw new Error(
      `Target node ${graph.target} does not exist.`
    );
  }

  if (graph.source === graph.target) {
    throw new Error("Source and target cannot be the same node.");
  }

  const edgeIds = new Set();

  // Validate edges.
  for (const edge of graph.edges) {
    if (!edge || typeof edge !== "object") {
      throw new Error("Invalid edge definition.");
    }

    if (
      typeof edge.id !== "string" ||
      edge.id.trim() === ""
    ) {
      throw new Error(
        `Invalid edge ID: ${JSON.stringify(edge)}`
      );
    }

    if (edgeIds.has(edge.id)) {
      throw new Error(`Duplicate edge ID: ${edge.id}`);
    }

    edgeIds.add(edge.id);

    if (
      !Number.isInteger(edge.from) ||
      !Number.isInteger(edge.to)
    ) {
      throw new Error(
        `Edge ${edge.id} must have integer from/to node IDs.`
      );
    }

    if (!nodeIds.has(edge.from)) {
      throw new Error(
        `Edge ${edge.id} references missing node ${edge.from}.`
      );
    }

    if (!nodeIds.has(edge.to)) {
      throw new Error(
        `Edge ${edge.id} references missing node ${edge.to}.`
      );
    }

    if (
      !Number.isFinite(edge.capacity) ||
      edge.capacity <= 0
    ) {
      throw new Error(
        `Edge ${edge.id} must have a positive capacity.`
      );
    }

    if (
      !Number.isFinite(edge.diameter) ||
      edge.diameter <= 0
    ) {
      throw new Error(
        `Edge ${edge.id} must have a positive diameter.`
      );
    }

    if (
      !Number.isFinite(edge.length) ||
      edge.length < 0
    ) {
      throw new Error(
        `Edge ${edge.id} must have a non-negative length.`
      );
    }
  }
}

// ─── Fallback demonstration graph ──────────────────────────────────────────

/**
 * Used only when pipeline.json is unavailable or invalid.
 *
 * This graph is simulated.
 *
 * Sensor mapping:
 * Node 2 = Pressure
 * Node 4 = Temperature
 * Node 6 = Flow Rate
 * Node 7 = Vibration
 */

function getDemoGraph() {
  return {
    metadata: {
      name: "HARP Fallback Demonstration Pipeline",
      topologyType: "simulated",
      directed: true,
      description:
        "Illustrative network for anomaly isolation and bypass routing."
    },

    source: 0,
    target: 5,

    nodes: [
      { id: 0, label: "Source Pump Station" },
      { id: 1, label: "Junction A" },
      { id: 2, label: "Junction B - Pressure Sensor" },
      { id: 3, label: "Junction C" },
      { id: 4, label: "Junction D - Temperature Sensor" },
      { id: 5, label: "Delivery Point" },
      { id: 6, label: "Junction E - Flow Sensor" },
      { id: 7, label: "Junction F - Vibration Sensor" },
      { id: 8, label: "Bypass Node G" },
      { id: 9, label: "Bypass Node H" }
    ],

    edges: [
      {
        id: "e0",
        from: 0,
        to: 1,
        capacity: 120,
        diameter: 6,
        length: 150
      },
      {
        id: "e1",
        from: 1,
        to: 2,
        capacity: 100,
        diameter: 5,
        length: 200
      },
      {
        id: "e2",
        from: 2,
        to: 4,
        capacity: 90,
        diameter: 5,
        length: 180
      },
      {
        id: "e3",
        from: 3,
        to: 5,
        capacity: 85,
        diameter: 4,
        length: 220
      },
      {
        id: "e4",
        from: 0,
        to: 6,
        capacity: 110,
        diameter: 6,
        length: 130
      },
      {
        id: "e5",
        from: 6,
        to: 7,
        capacity: 95,
        diameter: 5,
        length: 170
      },
      {
        id: "e6",
        from: 7,
        to: 5,
        capacity: 80,
        diameter: 4,
        length: 190
      },
      {
        id: "e7",
        from: 1,
        to: 3,
        capacity: 75,
        diameter: 4,
        length: 260
      },
      {
        id: "e8",
        from: 1,
        to: 6,
        capacity: 70,
        diameter: 4,
        length: 160
      },
      {
        id: "e9",
        from: 6,
        to: 9,
        capacity: 75,
        diameter: 4,
        length: 195
      },
      {
        id: "e10",
        from: 9,
        to: 5,
        capacity: 65,
        diameter: 3,
        length: 175
      },
      {
        id: "e11",
        from: 0,
        to: 8,
        capacity: 65,
        diameter: 4,
        length: 210
      },
      {
        id: "e12",
        from: 8,
        to: 3,
        capacity: 60,
        diameter: 3,
        length: 140
      },
      {
        id: "e13",
        from: 2,
        to: 8,
        capacity: 60,
        diameter: 3,
        length: 155
      },
      {
        id: "e14",
        from: 4,
        to: 3,
        capacity: 90,
        diameter: 5,
        length: 180
      }
    ]
  };
}

// ─── Load graph ─────────────────────────────────────────────────────────────

function loadGraph() {
  if (!existsSync(GRAPH_PATH)) {
    console.warn(
      `[graphRouter] pipeline.json not found at ${GRAPH_PATH}. Using fallback graph.`
    );

    return getDemoGraph();
  }

  try {
    const raw = readFileSync(GRAPH_PATH, "utf8");
    const graph = JSON.parse(raw);

    validateGraph(graph);

    console.log(
      `[graphRouter] Loaded pipeline graph: ` +
      `${graph.nodes.length} nodes, ${graph.edges.length} edges`
    );

    return graph;

  } catch (error) {
    console.error(
      `[graphRouter] Failed to load pipeline.json: ${error.message}`
    );

    console.warn(
      "[graphRouter] Using fallback demonstration graph."
    );

    return getDemoGraph();
  }
}

// ─── Initialize graph ───────────────────────────────────────────────────────

export const PIPELINE_GRAPH = loadGraph();

const VALID_NODES = new Set(
  PIPELINE_GRAPH.nodes.map((node) => node.id)
);

const DEFAULT_SOURCE = PIPELINE_GRAPH.source;
const DEFAULT_TARGET = PIPELINE_GRAPH.target;

// ─── Edge weight function ──────────────────────────────────────────────────

/**
 * Lower weight means a preferred route.
 *
 * Current cost model:
 *     weight = length / diameter
 *
 * This is a simplified routing heuristic, not a hydraulic
 * pressure-loss equation.
 */

function weightFn(edge) {
  return edge.length / edge.diameter;
}

// ─── Adjacency list ─────────────────────────────────────────────────────────

/**
 * Builds a directed adjacency list.
 *
 * Excludes:
 * 1. The isolated/leaking node.
 * 2. Every edge connected to that node.
 * 3. Every edge whose capacity is below requiredFlow.
 *
 * Edges are directed from `from` to `to`.
 * Reverse movement requires a separately defined reverse edge.
 */

function buildAdjacency(
  excludeNode = null,
  requiredFlow = 0
) {
  const adj = new Map();

  // Create an empty list for every node.
  for (const { id } of PIPELINE_GRAPH.nodes) {
    adj.set(id, []);
  }

  // Add only valid, usable edges.
  for (const edge of PIPELINE_GRAPH.edges) {
    const {
      from,
      to,
      capacity,
      id: edgeId
    } = edge;

    // Exclude leaking node and its incident edges.
    if (
      excludeNode !== null &&
      (from === excludeNode || to === excludeNode)
    ) {
      continue;
    }

    // Enforce minimum required flow.
    if (capacity < requiredFlow) {
      continue;
    }

    adj.get(from).push({
      to,
      weight: weightFn(edge),
      capacity,
      edgeId
    });
  }

  return adj;
}

// ─── Dijkstra shortest-path algorithm ───────────────────────────────────────

/**
 * Finds the lowest-cost path in a directed weighted graph.
 *
 * Returns:
 * {
 *   path: [node IDs],
 *   edges: [selected edge objects],
 *   totalCost: number
 * }
 *
 * Returns null if no path exists.
 */

function dijkstra(adj, source, target) {
  const dist = new Map();
  const previousNode = new Map();
  const previousEdge = new Map();
  const visited = new Set();

  // Initialize distances.
  for (const nodeId of adj.keys()) {
    dist.set(nodeId, Infinity);
  }

  dist.set(source, 0);

  // Priority queue: [distance, node]
  const priorityQueue = [
    [0, source]
  ];

  while (priorityQueue.length > 0) {
    // Extract node with smallest distance.
    priorityQueue.sort((a, b) => a[0] - b[0]);

    const [currentDistance, currentNode] =
      priorityQueue.shift();

    // Ignore stale queue entries.
    if (currentDistance > dist.get(currentNode)) {
      continue;
    }

    if (visited.has(currentNode)) {
      continue;
    }

    visited.add(currentNode);

    // Destination reached.
    if (currentNode === target) {
      break;
    }

    const neighbors = adj.get(currentNode) || [];

    for (const edge of neighbors) {
      const nextNode = edge.to;

      if (visited.has(nextNode)) {
        continue;
      }

      const newDistance =
        currentDistance + edge.weight;

      if (newDistance < dist.get(nextNode)) {
        dist.set(nextNode, newDistance);

        previousNode.set(nextNode, currentNode);
        previousEdge.set(nextNode, edge);

        priorityQueue.push([
          newDistance,
          nextNode
        ]);
      }
    }
  }

  // No route exists.
  if (!Number.isFinite(dist.get(target))) {
    return null;
  }

  // Reconstruct path and selected edges.
  const path = [target];
  const edges = [];

  let currentNode = target;

  while (currentNode !== source) {
    const edge = previousEdge.get(currentNode);
    const parent = previousNode.get(currentNode);

    if (edge === undefined || parent === undefined) {
      return null;
    }

    edges.unshift(edge);
    path.unshift(parent);

    currentNode = parent;
  }

  return {
    path,
    edges,
    totalCost: dist.get(target)
  };
}

// ─── Find bypass path ───────────────────────────────────────────────────────

/**
 * findBypassPath(leakNode, source, target, requiredFlow)
 *
 * Finds the minimum-cost path from source to target while:
 * 1. Excluding the leaking node.
 * 2. Excluding all edges connected to the leaking node.
 * 3. Excluding edges below the required capacity.
 *
 * @param {number} leakNode
 *   Node that must be isolated.
 *
 * @param {number} source
 *   Starting node. Defaults to graph source.
 *
 * @param {number} target
 *   Delivery node. Defaults to graph target.
 *
 * @param {number} requiredFlow
 *   Minimum capacity required on every edge.
 *   Defaults to 60.
 *
 * @returns {object}
 *   Feasible path, capacity information and diagnostics.
 */

export function findBypassPath(
  leakNode,
  source = DEFAULT_SOURCE,
  target = DEFAULT_TARGET,
  requiredFlow = 60
) {

  // ── 1. Validate required flow ──────────────────────────────────────────────

  if (
    !Number.isFinite(requiredFlow) ||
    requiredFlow < 0
  ) {
    return {
      feasible: false,
      reason: "requiredFlow must be a non-negative finite number."
    };
  }

  // ── 2. Validate graph nodes ───────────────────────────────────────────────

  if (!VALID_NODES.has(leakNode)) {
    return {
      feasible: false,
      reason:
        `Node ${leakNode} does not exist in the pipeline graph.`
    };
  }

  if (!VALID_NODES.has(source)) {
    return {
      feasible: false,
      reason:
        `Source node ${source} does not exist in the pipeline graph.`
    };
  }

  if (!VALID_NODES.has(target)) {
    return {
      feasible: false,
      reason:
        `Target node ${target} does not exist in the pipeline graph.`
    };
  }

  if (source === target) {
    return {
      feasible: false,
      reason: "Source and target must be different nodes."
    };
  }

  // ── 3. Check whether source or target is leaking ──────────────────────────

  if (
    leakNode === source ||
    leakNode === target
  ) {
    return {
      feasible: false,
      reason:
        `Leak is at source or delivery node (${leakNode}). Full shutdown required.`
    };
  }

  // ── 4. Build capacity-constrained adjacency list ──────────────────────────

  const adj = buildAdjacency(
    leakNode,
    requiredFlow
  );

  // ── 5. Run Dijkstra ───────────────────────────────────────────────────────

  const route = dijkstra(
    adj,
    source,
    target
  );

  if (!route) {
    return {
      feasible: false,

      reason:
        `No alternate path from node ${source} to node ${target} ` +
        `can satisfy required flow ${requiredFlow} ` +
        `after isolating node ${leakNode}.`,

      isolatedNode: leakNode,
      requiredFlow
    };
  }

  // ── 6. Calculate bottleneck capacity ─────────────────────────────────────

  const edgesUsed = route.edges.map(
    (edge) => edge.edgeId
  );

  const bottleneckCapacity = route.edges.reduce(
    (minimum, edge) =>
      Math.min(minimum, edge.capacity),
    Infinity
  );

  // Defensive validation.
  if (bottleneckCapacity < requiredFlow) {
    return {
      feasible: false,
      reason:
        "Selected route does not satisfy the required flow constraint.",
      isolatedNode: leakNode,
      requiredFlow,
      bottleneckCapacity
    };
  }

  // ── 7. Return successful bypass result ────────────────────────────────────

  return {
    feasible: true,

    path: route.path,

    pathDisplay: route.path.join(" → "),

    hops: route.path.length - 1,

    totalCost: Number(route.totalCost.toFixed(4)),

    bottleneckCapacity,

    requiredFlow,

    capacityWarning: false,

    status: "optimal",

    edgesUsed,

    isolatedNode: leakNode
  };
}
