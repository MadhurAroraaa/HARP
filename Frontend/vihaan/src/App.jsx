import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ||
  `${window.location.protocol}//${window.location.hostname}:3000`;

const WS_URL =
  import.meta.env.VITE_WS_URL ||
  `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.hostname}:3000`;

const ML_HEALTH_URL =
  import.meta.env.VITE_ML_HEALTH_URL ||
  `${API_BASE}/api/health`;

const SENSOR_CONFIG = [
  { key: "Pressure", label: "Pressure", unit: "dataset scale", short: "P" },
  { key: "Flow_Rate", label: "Flow rate", unit: "L/min", short: "F" },
  { key: "Temperature", label: "Temperature", unit: "dataset scale", short: "T" },
  { key: "Vibration", label: "Vibration", unit: "dataset scale", short: "V" },
];

const NODE_LAYOUT = {
  0: { x: 82, y: 220 },
  1: { x: 280, y: 105 },
  2: { x: 475, y: 105 },
  3: { x: 670, y: 105 },
  4: { x: 475, y: 220 },
  5: { x: 875, y: 220 },
  6: { x: 280, y: 220 },
  7: { x: 670, y: 220 },
  8: { x: 475, y: 350 },
  9: { x: 670, y: 350 },
};

function Icon({ name, size = 18, strokeWidth = 1.7 }) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth,
    strokeLinecap: "round",
    strokeLinejoin: "round",
    "aria-hidden": "true",
  };

  const paths = {
    activity: <><path d="M3 12h4l3-8 4 16 3-8h4" /></>,
    arrow: <><path d="M5 12h13" /><path d="m13 6 6 6-6 6" /></>,
    check: <><path d="m5 12 4 4L19 6" /></>,
    chevron: <><path d="m9 18 6-6-6-6" /></>,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    gauge: <><path d="M4 18a8 8 0 1 1 16 0" /><path d="m12 14 4-4" /><path d="M7 18h10" /></>,
    grid: <><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></>,
    layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 12 9 5 9-5" /><path d="m3 16 9 5 9-5" /></>,
    route: <><circle cx="6" cy="18" r="2" /><circle cx="18" cy="6" r="2" /><path d="M8 18h3a4 4 0 0 0 4-4V10a4 4 0 0 1 4-4" /></>,
    server: <><rect x="3" y="4" width="18" height="6" rx="1.5" /><rect x="3" y="14" width="18" height="6" rx="1.5" /><path d="M7 7h.01M7 17h.01M11 7h6M11 17h6" /></>,
    wave: <><path d="M2 12h3l2-6 4 12 3-9 2 3h6" /></>,
    warning: <><path d="M12 3 22 20H2L12 3Z" /><path d="M12 9v5M12 17h.01" /></>,
  };

  return <svg {...common}>{paths[name] || paths.activity}</svg>;
}

function StatusPill({ status = "connecting", children }) {
  const labels = {
    connected: "Live connection",
    connecting: "Connecting",
    disconnected: "Disconnected",
    error: "Connection error",
    ok: "Service ready",
    untrained: "Model unavailable",
  };
  const tone = status === "connected" || status === "ok" ? "good" : status === "connecting" ? "waiting" : "bad";
  return (
    <span className={`status-pill status-${tone}`}>
      <span className="status-dot" />
      {children || labels[status] || status}
    </span>
  );
}

function formatValue(value, digits = 2) {
  if (value === null || value === undefined || value === "" || !Number.isFinite(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function getReading(readings, key) {
  if (!readings) return null;
  const aliases = {
    Pressure: ["Pressure", "pressure"],
    Flow_Rate: ["Flow_Rate", "flow_rate", "flowRate"],
    Temperature: ["Temperature", "temperature"],
    Vibration: ["Vibration", "vibration"],
  };
  for (const name of aliases[key] || [key]) {
    const value = readings[name];
    if (value !== undefined && value !== null && Number.isFinite(Number(value))) return Number(value);
  }
  return null;
}

function formatTime(timestamp) {
  if (!timestamp) return "Waiting for first reading";
  const date = typeof timestamp === "number" ? new Date(timestamp) : new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "Recent update";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function makeNodePositions(nodes) {
  const positions = {};
  const unknown = nodes.filter((node) => !NODE_LAYOUT[node.id]);
  nodes.forEach((node, index) => {
    if (NODE_LAYOUT[node.id]) {
      positions[node.id] = NODE_LAYOUT[node.id];
      return;
    }
    const row = Math.floor(index / 4);
    const col = index % 4;
    positions[node.id] = { x: 240 + col * 180, y: 120 + row * 110 };
  });
  if (unknown.length) {
    unknown.forEach((node, index) => {
      const row = Math.floor(index / 4);
      const col = index % 4;
      positions[node.id] = { x: 190 + col * 185, y: 390 + row * 40 };
    });
  }
  return positions;
}

function PipelineMap({ graph, payload }) {
  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];
  const positions = useMemo(() => makeNodePositions(nodes), [nodes]);
  const leakNode = payload?.leak_node ?? null;
  const bypassPath = payload?.bypass?.feasible ? payload.bypass.path || [] : [];
  const bypassEdges = new Set();

  for (let i = 0; i < bypassPath.length - 1; i += 1) {
    bypassEdges.add(`${bypassPath[i]}-${bypassPath[i + 1]}`);
    bypassEdges.add(`${bypassPath[i + 1]}-${bypassPath[i]}`);
  }

  if (!nodes.length) {
    return (
      <div className="map-empty">
        <div className="map-empty-mark"><Icon name="grid" size={22} /></div>
        <strong>Topology is loading</strong>
        <span>Waiting for the backend graph configuration.</span>
      </div>
    );
  }

  return (
    <svg className="network-svg" viewBox="0 0 960 460" role="img" aria-label="Pipeline network topology">
      <defs>
        <pattern id="mapDots" width="24" height="24" patternUnits="userSpaceOnUse">
          <circle cx="1.2" cy="1.2" r="1" fill="#c9d2c8" />
        </pattern>
        <marker id="edgeArrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
          <path d="M0 0 L7 3.5 L0 7" fill="none" stroke="#a7b4a9" strokeWidth="1.2" />
        </marker>
        <marker id="routeArrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
          <path d="M0 0 L7 3.5 L0 7" fill="none" stroke="#b45e3f" strokeWidth="1.4" />
        </marker>
      </defs>
      <rect x="0" y="0" width="960" height="460" fill="url(#mapDots)" opacity="0.58" />

      {edges.map((edge) => {
        const from = positions[edge.from];
        const to = positions[edge.to];
        if (!from || !to) return null;
        const key = `${edge.from}-${edge.to}`;
        const onRoute = bypassEdges.has(key);
        const touchesLeak = leakNode !== null && (edge.from === leakNode || edge.to === leakNode);
        return (
          <g key={edge.id || key}>
            <line
              x1={from.x} y1={from.y} x2={to.x} y2={to.y}
              className={`network-edge ${onRoute ? "edge-route" : ""} ${touchesLeak && !onRoute ? "edge-isolated" : ""}`}
              markerEnd={onRoute ? "url(#routeArrow)" : "url(#edgeArrow)"}
            />
            {edge.capacity !== undefined && (
              <text x={(from.x + to.x) / 2} y={(from.y + to.y) / 2 - 8} className="edge-capacity">
                {edge.capacity}
              </text>
            )}
          </g>
        );
      })}

      {nodes.map((node) => {
        const point = positions[node.id];
        if (!point) return null;
        const isSource = Number(node.id) === Number(graph.source ?? 0);
        const isDestination = Number(node.id) === Number(graph.target ?? 5);
        const isLeak = Number(node.id) === Number(leakNode);
        const onRoute = bypassPath.map(Number).includes(Number(node.id));
        const label = node.label || `Node ${node.id}`;
        return (
          <g key={node.id} className="network-node">
            {isLeak && <circle className="node-halo" cx={point.x} cy={point.y} r="31" />}
            <circle
              cx={point.x} cy={point.y} r="22"
              className={`node-circle ${isSource ? "node-source" : ""} ${isDestination ? "node-destination" : ""} ${isLeak ? "node-leak" : ""} ${onRoute && !isLeak ? "node-route" : ""}`}
            />
            <text x={point.x} y={point.y + 4} className="node-id">{String(node.id).padStart(2, "0")}</text>
            <text x={point.x} y={point.y + 40} className="node-label">{label.length > 22 ? `${label.slice(0, 20)}…` : label}</text>
            <title>{`${label} · Node ${node.id}`}</title>
          </g>
        );
      })}
    </svg>
  );
}

function SensorCard({ sensor, value, active }) {
  return (
    <article className={`sensor-card ${active ? "sensor-card-active" : ""}`}>
      <div className="sensor-card-top">
        <span className="sensor-symbol">{sensor.short}</span>
        <span className="sensor-state">{value === null ? "NO DATA" : active ? "REVIEW" : "STREAMING"}</span>
      </div>
      <div className="sensor-name">{sensor.label}</div>
      <div className="sensor-value-row">
        <strong>{formatValue(value, 2)}</strong>
        <span>{sensor.unit}</span>
      </div>
      <div className="sensor-footnote">Live sensor channel</div>
    </article>
  );
}

function ScoreHistory({ history, threshold }) {
  const width = 620;
  const height = 112;
  const pad = 10;
  if (history.length < 2) {
    return <div className="history-empty">A trend will appear as readings arrive.</div>;
  }
  const values = history.map((entry) => entry.score);
  const maxValue = Math.max(threshold || 0, ...values, 0.000001) * 1.12;
  const pointString = values.map((value, index) => {
    const x = pad + (index / Math.max(1, values.length - 1)) * (width - pad * 2);
    const y = height - pad - (value / maxValue) * (height - pad * 2);
    return `${x},${y}`;
  }).join(" ");
  const thresholdY = height - pad - ((threshold || 0) / maxValue) * (height - pad * 2);
  const last = values[values.length - 1];
  const lastX = width - pad;
  const lastY = height - pad - (last / maxValue) * (height - pad * 2);

  return (
    <svg className="history-chart" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label="Anomaly score history with threshold line">
      {[28, 56, 84].map((y) => <line key={y} x1="0" y1={y} x2={width} y2={y} className="history-gridline" />)}
      <line x1="0" y1={thresholdY} x2={width} y2={thresholdY} className="history-threshold" />
      <polyline points={pointString} className="history-line" />
      <circle cx={lastX} cy={lastY} r="4" className="history-point" />
    </svg>
  );
}

function MetricCard({ label, value, detail, icon, accent = "green" }) {
  return (
    <article className={`metric-card metric-${accent}`}>
      <div className="metric-topline">
        <span>{label}</span>
        <span className="metric-icon"><Icon name={icon} size={17} /></span>
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-detail">{detail}</div>
    </article>
  );
}

export default function App() {
  const [wsStatus, setWsStatus] = useState("connecting");
  const [payload, setPayload] = useState(null);
  const [graph, setGraph] = useState(null);
  const [health, setHealth] = useState(null);
  const [tickCount, setTickCount] = useState(0);
  const [history, setHistory] = useState([]);
  const [events, setEvents] = useState([]);
  const [lastMessageAt, setLastMessageAt] = useState(null);
  const socketRef = useRef(null);
  const reconnectRef = useRef(null);
  const previousLeakRef = useRef({ active: false, node: null });

  useEffect(() => {
    let disposed = false;
    let healthTimer;

    const loadGraph = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/graph`);
        if (response.ok) setGraph(await response.json());
      } catch {
        // The WebSocket init packet is a second source for graph data.
      }
    };

    const loadHealth = async () => {
      try {
        const response = await fetch(ML_HEALTH_URL);
        if (response.ok) {
          const payload = await response.json();
          setHealth(payload.mlDetails ?? payload);
        }
      } catch {
        setHealth(null);
      }
    };

    const connect = () => {
      if (disposed) return;
      setWsStatus((current) => current === "connected" ? current : "connecting");
      const socket = new WebSocket(WS_URL);
      socketRef.current = socket;

      socket.onopen = () => {
        if (!disposed) setWsStatus("connected");
      };

      socket.onmessage = (event) => {
        if (disposed) return;
        let data;
        try {
          data = JSON.parse(event.data);
        } catch {
          return;
        }

        if (data.type === "init") {
          if (data.graph) setGraph(data.graph);
          return;
        }

        setPayload(data);
        setTickCount((count) => count + 1);
        setLastMessageAt(Date.now());
        setHistory((current) => [
          ...current.slice(-47),
          { score: Number(data.anomaly_score) || 0, time: data.timestamp || Date.now() },
        ]);

        const leakActive = Boolean(data.leak_detected);
        const node = data.leak_node ?? null;
        const previous = previousLeakRef.current;
        if (leakActive && (!previous.active || previous.node !== node)) {
          setEvents((current) => [
            {
              id: `${data.timestamp || Date.now()}-${node ?? "unknown"}`,
              time: formatTime(data.timestamp || Date.now()),
              node,
              severity: data.severity || "unclassified",
              path: data.bypass?.pathDisplay || null,
              score: data.anomaly_score,
            },
            ...current,
          ].slice(0, 8));
        }
        previousLeakRef.current = { active: leakActive, node };
      };

      socket.onerror = () => {
        if (!disposed) setWsStatus("error");
      };

      socket.onclose = () => {
        if (disposed) return;
        setWsStatus("disconnected");
        reconnectRef.current = window.setTimeout(connect, 2500);
      };
    };

    loadGraph();
    loadHealth();
    healthTimer = window.setInterval(loadHealth, 15000);
    connect();

    return () => {
      disposed = true;
      window.clearTimeout(reconnectRef.current);
      window.clearInterval(healthTimer);
      socketRef.current?.close();
    };
  }, []);

  const readings = payload?.readings || null;
  const isLeak = Boolean(payload?.leak_detected);
  const anomalyScore = Number(payload?.anomaly_score) || 0;
  const threshold = Number(payload?.threshold) || 0;
  const thresholdRatio = threshold > 0 ? anomalyScore / threshold : 0;
  const leakFeature = Number.isInteger(payload?.leak_feature) ? SENSOR_CONFIG[payload.leak_feature]?.label : null;
  const bypass = payload?.bypass;
  const modelReady = health?.model_ready ?? null;
  const systemStatus = !payload ? "Awaiting data" : isLeak ? "Anomaly flagged" : "Within threshold";
  const systemTone = !payload ? "neutral" : isLeak ? "warning" : "normal";

  return (
    <div className="dashboard-page">
      <header className="app-topbar">
        <div className="topbar-inner">
          <Link to="/" className="brand-lockup" aria-label="HARP home">
            <span className="brand-mark">H</span>
            <span className="brand-word">HARP</span>
            <span className="brand-divider" />
            <span className="brand-descriptor">Pipeline intelligence / research prototype</span>
          </Link>
          <div className="topbar-right">
            <div className="topbar-service"><span className={`service-dot ${modelReady === false ? "dot-off" : ""}`} />ML service {modelReady === false ? "unavailable" : modelReady ? "ready" : "checking"}</div>
            <StatusPill status={wsStatus} />
            <span className="topbar-clock"><Icon name="clock" size={15} /> {lastMessageAt ? formatTime(lastMessageAt) : "No stream yet"}</span>
          </div>
        </div>
      </header>

      <main className="dashboard-main">
        <section className="page-heading">
          <div>
            <div className="eyebrow"><span className="eyebrow-line" /> NETWORK OPERATIONS <span className="eyebrow-muted">/ OVERVIEW</span></div>
            <h1>Network overview</h1>
            <p>A live view of sensor activity, anomaly scoring and alternate route computation.</p>
          </div>
          <div className="heading-meta">
            <div className={`state-marker state-${systemTone}`}><span />{systemStatus}</div>
            <div className="sample-label">CSV-backed demonstration stream</div>
          </div>
        </section>

        <section className="metric-grid" aria-label="System summary">
          <MetricCard
            label="Detection state"
            value={isLeak ? "Flagged" : payload ? "Nominal" : "Waiting"}
            detail={payload ? (isLeak ? `Node ${payload.leak_node ?? "not resolved"} · ${payload.severity || "severity unclassified"}` : "Latest score is below threshold") : "Waiting for first sensor window"}
            icon="activity"
            accent={isLeak ? "orange" : "green"}
          />
          <MetricCard
            label="Anomaly score"
            value={payload ? formatValue(anomalyScore, 4) : "—"}
            detail="Latest Isolation Forest score"
            icon="gauge"
            accent="blue"
          />
          <MetricCard
            label="Detection threshold"
            value={threshold ? formatValue(threshold, 4) : "—"}
            detail="99th percentile of training scores"
            icon="layers"
            accent="sand"
          />
          <MetricCard
            label="Messages received"
            value={tickCount.toLocaleString()}
            detail="WebSocket sensor updates this session"
            icon="wave"
            accent="slate"
          />
        </section>

        <section className="workspace-grid">
          <article className="surface topology-panel">
            <div className="surface-header">
              <div>
                <div className="section-kicker">NETWORK MAP</div>
                <h2>Pipeline topology</h2>
              </div>
              <div className="map-header-meta">
                <span>{graph?.nodes?.length ?? "—"} nodes</span>
                <span className="meta-separator">·</span>
                <span>{graph?.edges?.length ?? "—"} links</span>
              </div>
            </div>
            <div className="topology-canvas">
              <PipelineMap graph={graph} payload={payload} />
            </div>
            <div className="map-legend">
              <span><i className="legend-dot legend-source" /> Source</span>
              <span><i className="legend-dot legend-destination" /> Delivery</span>
              <span><i className="legend-dot legend-leak" /> Flagged node</span>
              <span><i className="legend-line legend-route" /> Computed route</span>
            </div>
            {isLeak && leakFeature && (
              <div className="map-note"><Icon name="warning" size={15} /> The model's highest contributing sensor feature is <strong>{leakFeature}</strong>. The backend maps that feature to node {payload.leak_node ?? "—"}; this is a heuristic localisation, not a confirmed physical leak location.</div>
            )}
          </article>

          <aside className="side-stack">
            <article className={`surface assessment-panel ${isLeak ? "assessment-alert" : ""}`}>
              <div className="surface-header">
                <div>
                  <div className="section-kicker">MODEL OUTPUT</div>
                  <h2>Detection assessment</h2>
                </div>
                <span className={`assessment-tag ${isLeak ? "tag-alert" : "tag-calm"}`}>{isLeak ? "Above threshold" : payload ? "Below threshold" : "No reading"}</span>
              </div>
              <div className="assessment-score-row">
                <div>
                  <div className="big-score">{payload ? formatValue(anomalyScore, 4) : "—"}</div>
                  <div className="score-caption">ANOMALY SCORE</div>
                </div>
                <div className="score-ratio-block">
                  <span className="score-ratio-value">{payload && threshold ? `${(thresholdRatio * 100).toFixed(0)}%` : "—"}</span>
                  <span className="score-caption">OF THRESHOLD</span>
                </div>
              </div>
              <div className="threshold-track" aria-label="Anomaly score relative to threshold">
                <div className="threshold-fill" style={{ width: `${Math.min(thresholdRatio * 100, 100)}%` }} />
                <span className="threshold-marker" />
              </div>
              <div className="threshold-scale"><span>0</span><span>100% = detection threshold</span><span>200%+</span></div>
              <div className="assessment-details">
                <div><span>Threshold</span><strong>{threshold ? formatValue(threshold, 6) : "—"}</strong></div>
                <div><span>Model</span><strong>Isolation Forest</strong></div>
                <div><span>Feature signal</span><strong>{leakFeature || "Not identified"}</strong></div>
                <div><span>Backend severity</span><strong>{isLeak ? (payload?.severity || "Unclassified") : "—"}</strong></div>
              </div>
              <p className="fine-print">The ratio is a comparison with the model threshold, not a probability of a leak or a calibrated risk percentage.</p>
            </article>

            <article className="surface service-panel">
              <div className="surface-header compact-header">
                <div>
                  <div className="section-kicker">CONNECTIONS</div>
                  <h2>Service status</h2>
                </div>
                <Icon name="server" size={18} />
              </div>
              <div className="service-row">
                <span className="service-name">WebSocket stream</span>
                <StatusPill status={wsStatus} />
              </div>
              <div className="service-row">
                <span className="service-name">ML model</span>
                <StatusPill status={modelReady === true ? "ok" : modelReady === false ? "untrained" : "connecting"} />
              </div>
              <div className="service-row">
                <span className="service-name">Graph topology</span>
                <span className={`plain-state ${graph?.nodes?.length ? "plain-good" : ""}`}>{graph?.nodes?.length ? "Loaded" : "Waiting"}</span>
              </div>
            </article>
          </aside>
        </section>

        <section className="sensor-section">
          <div className="section-heading-row">
            <div>
              <div className="section-kicker">INSTRUMENTATION</div>
              <h2>Sensor channels</h2>
            </div>
            <span className="section-aside">Four raw channels · latest received row</span>
          </div>
          <div className="sensor-grid">
            {SENSOR_CONFIG.map((sensor) => (
              <SensorCard
                key={sensor.key}
                sensor={sensor}
                value={getReading(readings, sensor.key)}
                active={isLeak && SENSOR_CONFIG[payload?.leak_feature]?.key === sensor.key}
              />
            ))}
          </div>
        </section>

        <section className="lower-grid">
          <article className="surface trend-panel">
            <div className="surface-header">
              <div>
                <div className="section-kicker">RECENT SIGNAL</div>
                <h2>Anomaly score history</h2>
              </div>
              <span className="trend-count">Last {history.length} updates</span>
            </div>
            <div className="trend-chart-wrap"><ScoreHistory history={history} threshold={threshold} /></div>
            <div className="trend-legend"><span><i className="trend-key-score" /> Anomaly score</span><span><i className="trend-key-threshold" /> Model threshold</span></div>
          </article>

          <article className="surface route-panel">
            <div className="surface-header">
              <div>
                <div className="section-kicker">ROUTE COMPUTATION</div>
                <h2>Alternate path</h2>
              </div>
              <Icon name="route" size={19} />
            </div>
            {isLeak ? (
              bypass?.feasible ? (
                <>
                  <div className="route-status"><span className="route-status-dot" /> Feasible bypass returned</div>
                  <div className="route-path">{bypass.pathDisplay || (bypass.path || []).join(" → ")}</div>
                  <div className="route-facts">
                    <div><span>Hops</span><strong>{bypass.hops ?? "—"}</strong></div>
                    <div><span>Bottleneck capacity</span><strong>{bypass.bottleneckCapacity ?? "—"} <small>L/min</small></strong></div>
                    <div><span>Capacity check</span><strong className={bypass.capacityWarning ? "text-alert" : "text-good"}>{bypass.capacityWarning ? "Warning" : "No warning"}</strong></div>
                  </div>
                  <p className="fine-print">Route is computed against the configured demonstration graph. It is a decision-support output, not an automatic valve command.</p>
                </>
              ) : (
                <div className="route-empty route-empty-alert"><Icon name="warning" size={18} /><strong>No feasible bypass reported</strong><span>{bypass?.reason || "The backend did not return a feasible alternate path."}</span></div>
              )
            ) : (
              <div className="route-empty"><div className="route-icon"><Icon name="route" size={21} /></div><strong>No reroute requested</strong><span>A bypass is calculated when the detector flags an anomaly.</span></div>
            )}
          </article>

          <article className="surface events-panel">
            <div className="surface-header">
              <div>
                <div className="section-kicker">SESSION RECORD</div>
                <h2>Recent flagged events</h2>
              </div>
              <span className="trend-count">{events.length} recorded</span>
            </div>
            {events.length ? (
              <div className="event-list">
                {events.map((item) => (
                  <div className="event-row" key={item.id}>
                    <span className="event-marker"><Icon name="warning" size={14} /></span>
                    <div className="event-copy">
                      <strong>Node {item.node ?? "unresolved"} · {item.severity}</strong>
                      <span>{item.time}{item.path ? ` · ${item.path}` : " · No route reported"}</span>
                    </div>
                    <span className="event-score">{formatValue(item.score, 3)}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="events-empty">No anomaly transitions recorded in this browser session.</div>
            )}
            <div className="fine-print event-footnote">This event list is held in browser memory and clears when the page is reloaded.</div>
          </article>
        </section>

        <footer className="dashboard-footer">
          <span>HARP <span className="footer-separator">/</span> Pipeline anomaly research prototype</span>
          <span>Data source: local CSV stream <span className="footer-separator">·</span> Refresh: 1 s nominal tick</span>
        </footer>
      </main>
    </div>
  );
}
