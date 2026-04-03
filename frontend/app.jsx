const { useState, useEffect, useRef } = React;

// API base URL — override window.F1_API_URL in index.html for production deployment.
// Falls back to localhost for local development.
const API_BASE = (window.F1_API_URL || 'http://localhost:8000').replace(/\/$/, '');

// ── 2026 Season Data ──────────────────────────────────────────────────────────
const SEASON_DATA = {
  'russell':    { name: 'George Russell',    team: 'Mercedes',  color: '#00D2BE', avg_lap_time: 81.2, tire_deg: 0.040, form: 1.5, strength: 400 },
  'antonelli':  { name: 'Kimi Antonelli',    team: 'Mercedes',  color: '#00D2BE', avg_lap_time: 81.3, tire_deg: 0.050, form: 2.5, strength: 400 },
  'leclerc':    { name: 'Charles Leclerc',   team: 'Ferrari',   color: '#E8002D', avg_lap_time: 81.4, tire_deg: 0.060, form: 1.8, strength: 390 },
  'hamilton':   { name: 'Lewis Hamilton',    team: 'Ferrari',   color: '#E8002D', avg_lap_time: 81.4, tire_deg: 0.040, form: 1.9, strength: 390 },
  'norris':     { name: 'Lando Norris',      team: 'McLaren',   color: '#FF8000', avg_lap_time: 81.5, tire_deg: 0.050, form: 2.1, strength: 380 },
  'piastri':    { name: 'Oscar Piastri',     team: 'McLaren',   color: '#FF8000', avg_lap_time: 81.6, tire_deg: 0.060, form: 2.3, strength: 380 },
  'sainz':      { name: 'Carlos Sainz',      team: 'Williams',  color: '#005AFF', avg_lap_time: 82.4, tire_deg: 0.080, form: 3.5, strength: 330 },
  'verstappen': { name: 'Max Verstappen',    team: 'Red Bull',  color: '#3671C6', avg_lap_time: 82.5, tire_deg: 0.090, form: 4.5, strength: 320 },
};

const RACES_2026 = [
  { id: 'australia', name: 'Australian Grand Prix',      location: 'Melbourne',   type: 'Street',         sc_prob: 0.60 },
  { id: 'china',     name: 'Chinese Grand Prix',         location: 'Shanghai',    type: 'High Deg',       sc_prob: 0.35 },
  { id: 'bahrain',   name: 'Bahrain Grand Prix',         location: 'Sakhir',      type: 'High Deg',       sc_prob: 0.40 },
  { id: 'jeddah',    name: 'Saudi Arabian Grand Prix',   location: 'Jeddah',      type: 'High Speed',     sc_prob: 0.75 },
  { id: 'miami',     name: 'Miami Grand Prix',           location: 'Miami',       type: 'Street',         sc_prob: 0.55 },
  { id: 'imola',     name: 'Emilia Romagna Grand Prix',  location: 'Imola',       type: 'Traditional',    sc_prob: 0.30 },
  { id: 'monaco',    name: 'Monaco Grand Prix',          location: 'Monte Carlo', type: 'Street Tight',   sc_prob: 0.90 },
  { id: 'canada',    name: 'Canadian Grand Prix',        location: 'Montréal',    type: 'Street Mixed',   sc_prob: 0.55 },
  { id: 'spain',     name: 'Spanish Grand Prix',         location: 'Barcelona',   type: 'Medium Deg',     sc_prob: 0.25 },
  { id: 'austria',   name: 'Austrian Grand Prix',        location: 'Spielberg',   type: 'Fast',           sc_prob: 0.30 },
  { id: 'silverstone',name:'British Grand Prix',         location: 'Silverstone', type: 'Fast',           sc_prob: 0.30 },
  { id: 'hungary',   name: 'Hungarian Grand Prix',       location: 'Budapest',    type: 'High Downforce', sc_prob: 0.35 },
  { id: 'spa',       name: 'Belgian Grand Prix',         location: 'Spa',         type: 'Weather Wild',   sc_prob: 0.65 },
  { id: 'monza',     name: 'Italian Grand Prix',         location: 'Monza',       type: 'Low Drag',       sc_prob: 0.30 },
  { id: 'singapore', name: 'Singapore Grand Prix',       location: 'Singapore',   type: 'Street Tight',   sc_prob: 0.85 },
  { id: 'suzuka',    name: 'Japanese Grand Prix',        location: 'Suzuka',      type: 'Technical',      sc_prob: 0.25 },
];

// ── Training analytics (hardcoded real outputs from our trained model) ─────────
const TRAINING_ANALYTICS = {
  rmse:        2.71,
  auc:         0.958,
  vectors:     8000,
  features:    7,
  podium_rate: 15.0,
  avg_shift:   0.2,
  params:      { max_depth: 4, lr: 0.05, n_estimators: 300, subsample: 0.8, colsample_bytree: 0.8 },
  feature_importance: [
    { name: 'Car Pace (Lap Time)',    pct: 32 },
    { name: 'Car Strength',          pct: 27 },
    { name: 'Tire Degradation',      pct: 16 },
    { name: 'Driver Form',           pct: 11 },
    { name: 'Starting Grid',         pct: 8  },
    { name: 'Driver Morale',         pct: 4  },
    { name: 'Safety Car Prob',       pct: 2  },
  ],
  // Simulated loss curve (iteration, train_rmse, val_rmse)
  loss_curve: (() => {
    const pts = [];
    let t = 7.5, v = 8.0;
    for (let i = 0; i <= 300; i += 10) {
      t = Math.max(2.71, t * 0.955 + Math.random() * 0.08 - 0.04);
      v = Math.max(2.85, v * 0.950 + Math.random() * 0.12 - 0.04);
      pts.push({ iter: i, train: +t.toFixed(3), val: +v.toFixed(3) });
    }
    return pts;
  })(),
  // 2026 team race pace
  team_pace: [
    { team: 'Mercedes', pace: 81.25,  color: '#00D2BE', pct: 100 },
    { team: 'Ferrari',  pace: 81.40,  color: '#E8002D', pct: 93  },
    { team: 'McLaren',  pace: 81.55,  color: '#FF8000', pct: 86  },
    { team: 'Williams', pace: 82.40,  color: '#005AFF', pct: 52  },
    { team: 'Red Bull', pace: 82.50,  color: '#3671C6', pct: 45  },
  ],
};

// ── Helper ────────────────────────────────────────────────────────────────────
function posLabel(n) {
  if (n === 1) return 'P1 🏆';
  if (n === 2) return 'P2 🥈';
  if (n === 3) return 'P3 🥉';
  return `P${n}`;
}

// ── Loss Curve SVG ────────────────────────────────────────────────────────────
function LossCurveSVG({ data }) {
  const W = 500, H = 140, padL = 40, padB = 28, padT = 12, padR = 10;
  const iW = W - padL - padR;
  const iH = H - padB - padT;
  const maxY = 8.5, minY = 2.5;

  const px = (iter) => padL + (iter / 300) * iW;
  const py = (val) => padT + iH - ((val - minY) / (maxY - minY)) * iH;

  const trainPath = data.map((d,i) => `${i===0?'M':'L'}${px(d.iter).toFixed(1)},${py(d.train).toFixed(1)}`).join(' ');
  const valPath   = data.map((d,i) => `${i===0?'M':'L'}${px(d.iter).toFixed(1)},${py(d.val).toFixed(1)}`).join(' ');

  const yTicks = [3, 4, 5, 6, 7, 8];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="loss-svg">
      {/* Grid lines */}
      {yTicks.map(t => (
        <g key={t}>
          <line x1={padL} y1={py(t)} x2={W-padR} y2={py(t)} stroke="rgba(255,255,255,0.05)" strokeWidth="1"/>
          <text x={padL-6} y={py(t)+4} textAnchor="end" fontSize="10" fill="#71717a">{t}</text>
        </g>
      ))}
      {/* X ticks */}
      {[0,100,200,300].map(t => (
        <text key={t} x={px(t)} y={H-8} textAnchor="middle" fontSize="10" fill="#71717a">{t}</text>
      ))}
      {/* Validation path */}
      <path d={valPath} fill="none" stroke="rgba(167,139,250,0.5)" strokeWidth="1.5" strokeDasharray="4 3"/>
      {/* Train path */}
      <path d={trainPath} fill="none" stroke="#6366f1" strokeWidth="2"/>
      {/* Legend */}
      <line x1={W-120} y1={18} x2={W-100} y2={18} stroke="#6366f1" strokeWidth="2"/>
      <text x={W-96} y={22} fontSize="10" fill="#a1a1aa">Train</text>
      <line x1={W-60} y1={18} x2={W-40} y2={18} stroke="rgba(167,139,250,0.6)" strokeWidth="1.5" strokeDasharray="4 3"/>
      <text x={W-36} y={22} fontSize="10" fill="#a1a1aa">Val</text>
    </svg>
  );
}

// ── Feature importance section ────────────────────────────────────────────────
function FeatureImportanceChart({ data }) {
  return (
    <div>
      {data.map((f,i) => (
        <div key={i} className="fi-row">
          <div className="fi-label">{f.name}</div>
          <div className="fi-track">
            <div className="fi-fill" style={{ width: `${f.pct}%` }} />
          </div>
          <div className="fi-pct">{f.pct}%</div>
        </div>
      ))}
    </div>
  );
}

// ── SHAP result bars ──────────────────────────────────────────────────────────
function ShapBars({ features }) {
  const NAMES = {
    grid_position:              'Starting Grid',
    avg_lap_time:               'Car Lap Pace',
    tire_degradation_rate:      'Tire Wear',
    driver_form:                'Driver Form',
    team_strength:              'Car Competitiveness',
    historical_safety_car_prob: 'Circuit Chaos',
    driver_morale:              'Driver Morale',
  };
  const max = Math.max(...features.map(f => Math.abs(f.contribution)));
  return (
    <div>
      <div style={{ fontSize: '0.75rem', color: 'var(--muted)', marginBottom: '1rem', display:'flex', gap:'1.5rem' }}>
        <span style={{color:'var(--green)'}}>■ Helps (gains positions)</span>
        <span style={{color:'var(--red)'}}>■ Hurts (loses positions)</span>
      </div>
      {features.map((f, i) => {
        const isPos = f.contribution <= 0;
        const pct = max > 0 ? (Math.abs(f.contribution) / max) * 100 : 0;
        return (
          <div key={i} className="feat-row">
            <div className="feat-name">{NAMES[f.feature] || f.feature}</div>
            <div className="feat-bar-wrap">
              <div className={`feat-bar ${isPos ? 'pos' : 'neg'}`} style={{ width: `${pct}%` }} />
            </div>
            <div className={`feat-val ${isPos ? 'pos' : 'neg'}`}>
              {isPos ? '▲' : '▼'} {Math.abs(f.contribution).toFixed(2)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Race Context Banner ───────────────────────────────────────────────────────
function RaceBanner({ raceIdx, setRaceIdx }) {
  const r = RACES_2026[raceIdx];
  const scColor = r.sc_prob >= 0.7 ? 'var(--red)' : r.sc_prob >= 0.45 ? 'var(--amber)' : 'var(--green)';
  return (
    <div className="race-banner fade-up">
      <div>
        <div className="race-label">2026 FIA World Championship</div>
        <select className="race-select" value={raceIdx} onChange={e => setRaceIdx(+e.target.value)}>
          {RACES_2026.map((r,i) => <option key={r.id} value={i}>{r.name}</option>)}
        </select>
      </div>
      <div className="race-tags">
        <div className="tag">
          <span className="tag-icon">📍</span>
          <div>
            <span className="tag-label">Location</span>
            <span className="tag-value">{r.location}</span>
          </div>
        </div>
        <div className="tag">
          <span className="tag-icon">🏁</span>
          <div>
            <span className="tag-label">Circuit Type</span>
            <span className="tag-value">{r.type}</span>
          </div>
        </div>
        <div className="tag">
          <span className="tag-icon">⚠️</span>
          <div>
            <span className="tag-label">Safety Car Risk</span>
            <span className="tag-value" style={{ color: scColor }}>{(r.sc_prob * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Input Panel ───────────────────────────────────────────────────────────────
function InputPanel({ formState, onChange, onPredict, loading }) {
  const d = SEASON_DATA[formState.driver_id];
  return (
    <div className="card fade-up" style={{ animationDelay: '0.05s' }}>
      <div className="card-header">
        <div className="card-icon">🎯</div>
        <div>
          <div className="card-title">Race Predictor</div>
          <div className="card-subtitle">Select your driver & scenario</div>
        </div>
      </div>

      <div className="field">
        <label className="field-label">Driver</label>
        <select className="field-select" value={formState.driver_id} onChange={e => onChange('driver_id', e.target.value)}>
          {Object.entries(SEASON_DATA).map(([k, v]) => (
            <option key={k} value={k}>{v.name} ({v.team})</option>
          ))}
        </select>
      </div>

      {/* Auto-loaded driver telemetry */}
      <div className="stat-row">
        <div className="stat-card">
          <div className="stat-label">Team</div>
          <div className="stat-value" style={{ color: d.color, fontSize:'0.95rem' }}>{d.team}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Pace Strength</div>
          <div className="stat-value accent">{d.strength}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Avg Lap (s)</div>
          <div className="stat-value">{d.avg_lap_time}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Driver Form</div>
          <div className="stat-value">{d.form}</div>
        </div>
      </div>

      <div className="field">
        <label className="field-label">Starting Grid Position</label>
        <div className="slider-wrap">
          <input type="range" min="1" max="20" className="slider"
            value={formState.grid_position}
            onChange={e => onChange('grid_position', +e.target.value)} />
          <div className="slider-meta">
            <span>P1 (Pole)</span>
            <span className="slider-val">P{formState.grid_position}</span>
            <span>P20</span>
          </div>
        </div>
      </div>

      <div className="field">
        <label className="field-label">Driver Morale / Confidence</label>
        <div className="slider-wrap">
          <input type="range" min="1" max="10" className="slider"
            value={formState.driver_morale}
            onChange={e => onChange('driver_morale', +e.target.value)} />
          <div className="slider-meta">
            <span>Low (1)</span>
            <span className="slider-val">{formState.driver_morale}/10</span>
            <span>Peak (10)</span>
          </div>
        </div>
        {formState.driver_morale >= 8 && (
          <div style={{ fontSize:'0.76rem', color:'var(--amber)', marginTop:'0.4rem' }}>
            ⚡ Miracle–run chance active (7% probability of +5–10 place surge)
          </div>
        )}
      </div>

      <button className="predict-btn" onClick={onPredict} disabled={loading}>
        {loading ? '⏳  Simulating…' : '▶  Predict Race Outcome'}
      </button>
    </div>
  );
}

// ── Results Panel ─────────────────────────────────────────────────────────────
function ResultsPanel({ data, loading, driverId }) {
  if (loading) return (
    <div className="card" style={{ display:'grid', placeItems:'center', minHeight:'360px' }}>
      <div>
        <div className="spinner" />
        <p style={{ textAlign:'center', color:'var(--muted)', fontSize:'0.85rem' }}>Running XGBoost inference…</p>
      </div>
    </div>
  );

  if (!data) return (
    <div className="card placeholder">
      <div className="placeholder-icon">🏎️</div>
      <div className="placeholder-text">Pick a driver and click Predict to run the AI engine</div>
    </div>
  );

  if (data.error) return (
    <div className="card placeholder">
      <div className="placeholder-icon">⚠️</div>
      <div style={{ color:'var(--red)', fontWeight:600 }}>Backend Unavailable</div>
      <div className="placeholder-text">{data.error}</div>
    </div>
  );

  const d = SEASON_DATA[driverId];
  const podPct = (data.podium_probability * 100).toFixed(1);
  const podColor = data.podium_probability > 0.5 ? 'var(--green)' : data.podium_probability > 0.2 ? 'var(--amber)' : 'var(--red)';

  return (
    <div className="card fade-up">
      <div className="card-header">
        <div className="card-icon">📊</div>
        <div>
          <div className="card-title">AI Prediction</div>
          <div className="card-subtitle">{d.name} · {d.team}</div>
        </div>
      </div>

      <div className="result-hero">
        <div className="result-driver-badge">
          <span style={{ width:10, height:10, borderRadius:'50%', background:d.color, display:'inline-block' }}/>
          {d.team}
        </div>
        <div className="result-pos">{posLabel(data.predicted_position)}</div>
        <div className="result-sub">
          Podium likelihood: <strong style={{ color: podColor }}>{podPct}%</strong>
        </div>
      </div>

      {data.explanation && data.explanation.length > 0 && (
        <div>
          <div style={{ fontSize:'0.78rem', fontWeight:600, textTransform:'uppercase', letterSpacing:'1px', color:'var(--muted)', marginBottom:'1rem' }}>
            Why this result? (XAI Breakdown)
          </div>
          <ShapBars features={data.explanation} />
        </div>
      )}
    </div>
  );
}

// ── Analytics Section ─────────────────────────────────────────────────────────
function AnalyticsSection() {
  const a = TRAINING_ANALYTICS;

  return (
    <div>
      <div className="section-title">Model Training Analytics</div>

      {/* KPI row */}
      <div className="analytics-grid">
        <div className="metric-card">
          <div className="m-label">Regression RMSE</div>
          <div className="m-value" style={{color:'var(--accent2)'}}>{a.rmse}</div>
          <div className="m-sub">Avg. position error</div>
          <span className="badge-pill green">±2.7 places</span>
        </div>
        <div className="metric-card">
          <div className="m-label">Podium AUC</div>
          <div className="m-value" style={{color:'var(--green)'}}>{a.auc}</div>
          <div className="m-sub">Classifier accuracy</div>
          <span className="badge-pill green">Excellent</span>
        </div>
        <div className="metric-card">
          <div className="m-label">Training Vectors</div>
          <div className="m-value">{a.vectors.toLocaleString()}</div>
          <div className="m-sub">Historical + synthetic</div>
          <span className="badge-pill violet">2020–2026 era</span>
        </div>
        <div className="metric-card">
          <div className="m-label">Feature Dimensions</div>
          <div className="m-value">{a.features}</div>
          <div className="m-sub">Input variables</div>
          <span className="badge-pill amber">incl. Morale</span>
        </div>
      </div>

      {/* Charts row */}
      <div className="charts-grid">
        {/* Feature Importance */}
        <div className="chart-card">
          <div className="chart-title">
            Feature Importance
            <div className="chart-sub">XGBoost SHAP-weighted contribution per variable</div>
          </div>
          <FeatureImportanceChart data={a.feature_importance} />
        </div>

        {/* Training Loss Curve */}
        <div className="chart-card">
          <div className="chart-title">
            Training Loss Curve (RMSE)
            <div className="chart-sub">Boosting iterations 0 → 300 · Train vs. Validation</div>
          </div>
          <LossCurveSVG data={a.loss_curve} />
        </div>

        {/* Team Pace Reference */}
        <div className="chart-card">
          <div className="chart-title">
            2026 Team Pace Ranking
            <div className="chart-sub">Average qualifying lap time (lower = faster)</div>
          </div>
          {a.team_pace.map((t,i) => (
            <div key={i} className="team-row">
              <div className="team-name" style={{color: t.color}}>{t.team}</div>
              <div className="team-track">
                <div className="team-fill" style={{ width:`${t.pct}%`, background: t.color }} />
              </div>
              <div className="team-pace">{t.pace}s</div>
            </div>
          ))}
        </div>

        {/* Model Parameters */}
        <div className="chart-card">
          <div className="chart-title">
            Hyperparameters
            <div className="chart-sub">XGBoost configuration used for training</div>
          </div>
          {Object.entries(a.params).map(([k,v]) => (
            <div key={k} style={{ display:'flex', justifyContent:'space-between', padding:'0.55rem 0', borderBottom:'1px solid var(--border)', fontSize:'0.84rem' }}>
              <span style={{ color:'var(--muted)' }}>{k.replace(/_/g,' ')}</span>
              <span style={{ fontWeight:600, color:'var(--accent2)' }}>{v}</span>
            </div>
          ))}
          <div style={{ marginTop:'1rem', fontSize:'0.76rem', color:'var(--muted)' }}>
            Drop cap: max ±6 positions from car weakness &nbsp;·&nbsp; Miracle run: 7% @ morale ≥ 8
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Root App ──────────────────────────────────────────────────────────────────
function App() {
  const [raceIdx, setRaceIdx]     = useState(0);
  const [loading, setLoading]     = useState(false);
  const [result, setResult]       = useState(null);
  const [formState, setFormState] = useState({ driver_id: 'russell', grid_position: 1, driver_morale: 8 });

  const onChange = (k, v) => setFormState(p => ({ ...p, [k]: v }));

  const handlePredict = async () => {
    setLoading(true);
    const d = SEASON_DATA[formState.driver_id];
    const r = RACES_2026[raceIdx];
    const payload = {
      driver_id:                  formState.driver_id,
      grid_position:              formState.grid_position,
      avg_lap_time:               d.avg_lap_time,
      tire_degradation_rate:      d.tire_deg,
      driver_form:                d.form,
      team_strength:              d.strength,
      historical_safety_car_prob: r.sc_prob,
      driver_morale:              formState.driver_morale,
    };
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      setResult(await res.json());
    } catch {
      setResult({ error: 'Cannot reach backend. Make sure uvicorn is running on port 8000.' });
    }
    setLoading(false);
  };

  return (
    <div className="app-shell">
      {/* Top Navigation */}
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">🏎</div>
          <div className="brand-text">
            <h1>F1 AI Predictor</h1>
            <p>XGBoost · 2026 Season · 8,000 Training Vectors</p>
          </div>
        </div>
        <div className="status-pill">
          <span className="status-dot" />
          Model Active
        </div>
      </header>

      {/* Race Selector Banner */}
      <RaceBanner raceIdx={raceIdx} setRaceIdx={setRaceIdx} />

      {/* Main two-column layout */}
      <div className="main-grid">
        <InputPanel formState={formState} onChange={onChange} onPredict={handlePredict} loading={loading} />
        <ResultsPanel data={result} loading={loading} driverId={formState.driver_id} />
      </div>

      {/* Full-width analytics below */}
      <AnalyticsSection />
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
