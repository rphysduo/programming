"use client";

import { useMemo, useState } from "react";
import SimulationScene from "../components/SimulationScene";

type LayerKey = "cad" | "water" | "rays" | "board";

const layerLabels: Record<LayerKey, string> = {
  cad: "CAD shell",
  water: "Water surface",
  rays: "Stereo rays",
  board: "Reference grid",
};

export default function Home() {
  const [playing, setPlaying] = useState(true);
  const [amplitude, setAmplitude] = useState(24);
  const [frequency, setFrequency] = useState(0.78);
  const [speed, setSpeed] = useState(0.34);
  const [rayDensity, setRayDensity] = useState(7);
  const [layers, setLayers] = useState<Record<LayerKey, boolean>>({
    cad: true,
    water: true,
    rays: true,
    board: true,
  });
  const [viewRevision, setViewRevision] = useState(0);

  const wavelength = useMemo(() => Math.max(0.28, speed / frequency), [speed, frequency]);
  const steepness = useMemo(
    () => ((2 * Math.PI * (amplitude / 1000)) / wavelength).toFixed(3),
    [amplitude, wavelength],
  );

  function toggleLayer(key: LayerKey) {
    setLayers((current) => ({ ...current, [key]: !current[key] }));
  }

  return (
    <main className="sim-shell">
      <SimulationScene
        playing={playing}
        amplitudeMm={amplitude}
        frequencyHz={frequency}
        phaseSpeed={speed}
        rayDensity={rayDensity}
        layers={layers}
        viewRevision={viewRevision}
      />

      <header className="topbar glass-panel">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <div>
            <p className="eyebrow">RPHYS / DIGITAL TWIN</p>
            <h1>Wave Tank <span>Field Lab</span></h1>
          </div>
        </div>
        <div className="topbar-center">
          <span className="live-dot" />
          <span>SIMULATION ACTIVE</span>
          <i />
          <span>FULL ASSEMBLY.STL</span>
        </div>
        <button className="icon-button" onClick={() => setViewRevision((value) => value + 1)} aria-label="Reset camera view" title="Reset camera view">
          ↗
        </button>
      </header>

      <aside className="left-rail">
        <section className="glass-panel panel-block run-card">
          <div className="section-heading">
            <div>
              <p className="eyebrow">EXPERIMENT STATE</p>
              <h2>Pre-breaking run</h2>
            </div>
            <span className="status-pill">LIVE</span>
          </div>
          <div className="run-id">
            <span>RUN</span>
            <strong>RPH-27A</strong>
            <small>synthetic</small>
          </div>
          <div className="mini-stats">
            <div><span>Tank model</span><strong>1.74 m</strong></div>
            <div><span>Mesh</span><strong>453.9k Δ</strong></div>
          </div>
        </section>

        <section className="glass-panel panel-block stereo-card">
          <div className="section-heading compact">
            <div>
              <p className="eyebrow">STEREO RIG</p>
              <h2>Camera pair</h2>
            </div>
            <span className="signal-bars" aria-hidden="true"><i /><i /><i /></span>
          </div>
          <div className="camera-row camera-a">
            <span className="camera-chip">A</span>
            <div><strong>LEFT / MASTER</strong><small>120 fps · synced</small></div>
            <b>●</b>
          </div>
          <div className="camera-row camera-b">
            <span className="camera-chip">B</span>
            <div><strong>RIGHT / SLAVE</strong><small>120 fps · +0.3 ms</small></div>
            <b>●</b>
          </div>
          <div className="baseline-readout">
            <span>Baseline</span><strong>36.0 <small>mm</small></strong>
          </div>
        </section>
      </aside>

      <aside className="right-rail glass-panel panel-block">
        <div className="section-heading compact">
          <div>
            <p className="eyebrow">FIELD CONTROLS</p>
            <h2>Wave generator</h2>
          </div>
          <button className="tiny-button" onClick={() => {
            setAmplitude(24);
            setFrequency(0.78);
            setSpeed(0.34);
          }}>RESET</button>
        </div>

        <Control label="Amplitude" value={`${amplitude} mm`} min={6} max={48} step={1} current={amplitude} onChange={setAmplitude} />
        <Control label="Frequency" value={`${frequency.toFixed(2)} Hz`} min={0.25} max={1.45} step={0.01} current={frequency} onChange={setFrequency} />
        <Control label="Phase speed" value={`${speed.toFixed(2)} m/s`} min={0.12} max={0.72} step={0.01} current={speed} onChange={setSpeed} />
        <Control label="Ray samples" value={`${rayDensity * 2}`} min={3} max={12} step={1} current={rayDensity} onChange={setRayDensity} />

        <div className="metrics-grid">
          <div><span>λ</span><strong>{wavelength.toFixed(2)} m</strong><small>wavelength</small></div>
          <div><span>ak</span><strong>{steepness}</strong><small>steepness</small></div>
        </div>

        <div className="layer-list">
          <p className="eyebrow">SCENE LAYERS</p>
          {(Object.keys(layerLabels) as LayerKey[]).map((key) => (
            <button key={key} className={layers[key] ? "layer-row active" : "layer-row"} onClick={() => toggleLayer(key)}>
              <span className={`layer-swatch ${key}`} />
              <span>{layerLabels[key]}</span>
              <i />
            </button>
          ))}
        </div>
      </aside>

      <section className="bottom-console glass-panel">
        <button className="play-button" onClick={() => setPlaying((value) => !value)} aria-label={playing ? "Pause simulation" : "Play simulation"}>
          {playing ? "Ⅱ" : "▶"}
        </button>
        <div className="timeline-copy">
          <p className="eyebrow">SIMULATION CLOCK</p>
          <strong>{playing ? "EVOLVING" : "PAUSED"}</strong>
        </div>
        <div className="timeline-track"><span /><i style={{ left: "58%" }} /></div>
        <div className="clock-readout"><small>PHASE VELOCITY</small><strong>{speed.toFixed(2)} m/s</strong></div>
      </section>

      <div className="scientific-note">
        <span>2.5D</span>
        Synthetic surface visualization — not CFD; no overturning, splashing, turbulence, or air entrainment.
      </div>
    </main>
  );
}

type ControlProps = {
  label: string;
  value: string;
  min: number;
  max: number;
  step: number;
  current: number;
  onChange: (value: number) => void;
};

function Control({ label, value, min, max, step, current, onChange }: ControlProps) {
  const progress = ((current - min) / (max - min)) * 100;
  const id = `control-${label.toLowerCase().replaceAll(" ", "-")}`;
  return (
    <label className="control-row" htmlFor={id}>
      <span><b>{label}</b><strong>{value}</strong></span>
      <input
        id={id}
        aria-label={label}
        type="range"
        min={min}
        max={max}
        step={step}
        value={current}
        onChange={(event) => onChange(Number(event.target.value))}
        style={{ "--range-progress": `${progress}%` } as React.CSSProperties}
      />
    </label>
  );
}
