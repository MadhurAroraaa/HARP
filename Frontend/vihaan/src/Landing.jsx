import { Link } from "react-router-dom";

function ArrowMark() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 12h15M13 5l7 7-7 7" />
    </svg>
  );
}

function NetworkIllustration() {
  const nodes = [
    { x: 55, y: 180, label: "SOURCE", kind: "source" },
    { x: 220, y: 92, label: "01" },
    { x: 405, y: 92, label: "02" },
    { x: 590, y: 92, label: "03" },
    { x: 220, y: 265, label: "04" },
    { x: 405, y: 265, label: "05" },
    { x: 590, y: 265, label: "06" },
    { x: 405, y: 365, label: "B1", kind: "bypass" },
    { x: 590, y: 365, label: "B2", kind: "bypass" },
    { x: 775, y: 180, label: "DELIVERY", kind: "destination" },
  ];
  const links = [
    [0, 1], [1, 2], [2, 3], [3, 9], [0, 4], [4, 5], [5, 6], [6, 9],
    [1, 4], [2, 5], [4, 7], [7, 8], [8, 9], [5, 8],
  ];

  return (
    <svg className="landing-network" viewBox="0 0 830 430" role="img" aria-label="Illustrative pipeline network diagram">
      <defs>
        <pattern id="landingGrid" width="26" height="26" patternUnits="userSpaceOnUse">
          <path d="M26 0H0V26" fill="none" stroke="#d9ded4" strokeWidth="0.8" />
        </pattern>
      </defs>
      <rect width="830" height="430" fill="url(#landingGrid)" opacity="0.58" />
      {links.map(([from, to], index) => {
        const a = nodes[from];
        const b = nodes[to];
        const highlighted = index === 10 || index === 11 || index === 12;
        return <line key={`${from}-${to}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} className={highlighted ? "landing-link landing-link-route" : "landing-link"} />;
      })}
      {nodes.map((node) => (
        <g key={node.label}>
          <circle cx={node.x} cy={node.y} r={node.kind ? 19 : 15} className={`landing-node ${node.kind ? `landing-node-${node.kind}` : ""}`} />
          <text x={node.x} y={node.y + 4} className="landing-node-text">{node.label}</text>
        </g>
      ))}
      <path d="M405 386h185" className="landing-route-dash" />
      <text x="497" y="414" textAnchor="middle" className="landing-route-label">ALTERNATE ROUTE</text>
    </svg>
  );
}

const capabilities = [
  {
    number: "01",
    title: "Read the network",
    body: "Four sensor channels are streamed at regular intervals and assembled into a rolling input window.",
  },
  {
    number: "02",
    title: "Score deviations",
    body: "An Isolation Forest model evaluates engineered features and compares the latest score with its learned threshold.",
  },
  {
    number: "03",
    title: "Explore a bypass",
    body: "When an anomaly is flagged, the backend evaluates an alternate path through the configured pipeline graph.",
  },
];

export default function Landing() {
  return (
    <div className="landing-page">
      <header className="landing-nav">
        <div className="landing-nav-inner">
          <Link to="/" className="brand-lockup">
            <span className="brand-mark">H</span>
            <span className="brand-word">HARP</span>
            <span className="brand-divider" />
            <span className="brand-descriptor">Pipeline intelligence</span>
          </Link>
          <div className="landing-nav-right">
            <span className="research-label">ENGINEERING RESEARCH PROTOTYPE</span>
            <Link className="nav-dashboard-link" to="/dashboard">Open workspace <ArrowMark /></Link>
          </div>
        </div>
      </header>

      <main>
        <section className="landing-hero">
          <div className="landing-hero-copy">
            <div className="eyebrow"><span className="eyebrow-line" /> PIPELINE MONITORING / 01</div>
            <h1>Know what is happening <em>inside</em> the network.</h1>
            <p className="landing-lede">
              HARP is a research prototype for pipeline sensor anomaly detection and graph-based rerouting. It brings incoming measurements, model output and network topology into one operator workspace.
            </p>
            <div className="landing-actions">
              <Link to="/dashboard" className="button-primary">Open live workspace <ArrowMark /></Link>
              <a href="#method" className="button-secondary">Explore the method</a>
            </div>
            <div className="landing-footnote"><span className="footnote-dot" /> Local demonstration environment · CSV-backed sensor stream</div>
          </div>

          <div className="landing-hero-art">
            <div className="art-topline"><span>NETWORK SCHEMATIC</span><span>FIG. 01 / HARP</span></div>
            <NetworkIllustration />
            <div className="art-bottomline"><span>Source → Junctions → Delivery</span><span>Illustrative topology</span></div>
          </div>
        </section>

        <section className="facts-strip" aria-label="Current implementation facts">
          <div className="fact-item"><strong>04</strong><span>Sensor channels</span></div>
          <div className="fact-item"><strong>09</strong><span>Engineered model features</span></div>
          <div className="fact-item"><strong>30</strong><span>Readings per window</span></div>
          <div className="fact-item"><strong>1s</strong><span>Nominal stream interval</span></div>
        </section>

        <section className="method-section" id="method">
          <div className="method-intro">
            <div className="eyebrow"><span className="eyebrow-line" /> THE WORKFLOW / 02</div>
            <h2>From a sensor reading to a route decision.</h2>
            <p>The current implementation connects a conventional anomaly-detection pipeline to a graph-routing service. Each stage is visible and inspectable.</p>
          </div>
          <div className="capability-list">
            {capabilities.map((item) => (
              <article className="capability-row" key={item.number}>
                <span className="capability-number">{item.number}</span>
                <div><h3>{item.title}</h3><p>{item.body}</p></div>
                <span className="capability-arrow"><ArrowMark /></span>
              </article>
            ))}
          </div>
        </section>

        <section className="architecture-section">
          <div className="architecture-heading">
            <div className="eyebrow"><span className="eyebrow-line" /> IMPLEMENTATION / 03</div>
            <h2>A deliberately simple stack.</h2>
            <p>Built as three local services so data flow and responsibilities remain easy to inspect.</p>
          </div>
          <div className="architecture-grid">
            <div className="architecture-cell"><span>01 / DETECTION</span><strong>Python · FastAPI</strong><small>Scikit-learn Isolation Forest</small></div>
            <div className="architecture-cell"><span>02 / ORCHESTRATION</span><strong>Node.js · Express</strong><small>CSV streaming · WebSocket</small></div>
            <div className="architecture-cell"><span>03 / ROUTING</span><strong>Dijkstra algorithm</strong><small>Shortest-path evaluation</small></div>
            <div className="architecture-cell"><span>04 / INTERFACE</span><strong>React · Vite</strong><small>Live monitoring workspace</small></div>
          </div>
          <p className="architecture-note">Model scope: the live API currently uses Isolation Forest. LSTM integration is not represented as an active detector in this prototype.</p>
        </section>
      </main>

      <footer className="landing-footer">
        <div className="footer-brand"><span className="brand-mark">H</span><span>HARP</span></div>
        <span>Pipeline anomaly research prototype</span>
        <Link to="/dashboard">Go to workspace <ArrowMark /></Link>
      </footer>
    </div>
  );
}
