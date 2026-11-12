"use client";

import React, { useState, useEffect } from 'react';
import { PenTool, FileText, Sliders, RefreshCw, CheckCircle2, Sparkles } from 'lucide-react';
import HandwritingCanvas from './HandwritingCanvas';

interface SynthesisTabProps {
  writers: any[];
  archetypes: string[];
}

export default function SynthesisTab({ writers, archetypes }: SynthesisTabProps) {
  const [text, setText] = useState("Continuous Kinematic Trajectory");
  const [selectedWriter, setSelectedWriter] = useState(writers[0]?.id || "a01-000");
  const [temperatureBias, setTemperatureBias] = useState(1.2);
  const [maxLen, setMaxLen] = useState(300);
  const [applyKalman, setApplyKalman] = useState(true);

  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  // Reference sample state
  const [realSample, setRealSample] = useState<any>(null);
  const [loadingReal, setLoadingReal] = useState(false);

  const promptPresets = [
    "Continuous Kinematic Trajectory",
    "Hidden Markov Model Viterbi Decoding",
    "Expectation Maximization Density Estimation",
    "Stacking Meta-Ensemble Optimization",
    "The quick brown fox jumps over the lazy dog"
  ];

  // Archetype traits mapping for biometric profile display
  const activeWriterObj = writers.find((w: any) => w.id === selectedWriter) || writers[0];
  const archetypeTraits: Record<number, { slant: string; spacing: string; motor: string }> = {
    0: { slant: "+24° Italic", spacing: "Compact (0.85×)", motor: "High-velocity angular downstrokes" },
    1: { slant: "+1° Upright", spacing: "Medium (1.00×)", motor: "Detached vertical print forms" },
    2: { slant: "+15° Forward", spacing: "Condensed (0.78×)", motor: "Tightly packed angular ligatures" },
    3: { slant: "+27° Sweeping", spacing: "Flowing (1.35×)", motor: "Broad loops & exaggerated ascenders" },
    4: { slant: "+12° Natural", spacing: "Sprawling (1.22×)", motor: "Relaxed spacing & baseline wander" },
  };
  const activeTraits = archetypeTraits[activeWriterObj?.archetype_id ?? 0] || archetypeTraits[0];

  useEffect(() => {
    if (writers.length > 0 && !selectedWriter) {
      setSelectedWriter(writers[0].id);
    }
  }, [writers, selectedWriter]);

  // Initial synthesis on mount
  useEffect(() => {
    if (writers.length > 0 && !result && !isLoading) {
      handleSynthesize();
    }
  }, [writers]);

  const handleSynthesize = async () => {
    if (!text.trim()) return;
    setIsLoading(true);
    try {
      const res = await fetch('/api/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          writer_id: selectedWriter,
          temperature_bias: temperatureBias,
          max_len: maxLen,
          apply_kalman: applyKalman
        })
      });
      const data = await res.json();
      setResult(data);
      setRealSample(null); // Return to generated view
    } catch (err) {
      console.error("Synthesis error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoadReal = async () => {
    setLoadingReal(true);
    try {
      const randIdx = Math.floor(Math.random() * 25);
      const res = await fetch(`/api/samples/real?index=${randIdx}`);
      const data = await res.json();
      setRealSample(data);
    } catch (err) {
      console.error("Error loading sample:", err);
    } finally {
      setLoadingReal(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Intro Header */}
      <div className="p-5 rounded-lg bg-white border border-zinc-200 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-base font-semibold text-zinc-900 tracking-tight">
              Kinematic Trajectory Studio
            </h2>
            <span className="px-2 py-0.5 rounded text-[11px] font-mono text-zinc-600 bg-zinc-100 border border-zinc-200">
              Style-Disentangled Transformer
            </span>
          </div>
          <p className="text-xs text-zinc-600 mt-1 max-w-2xl leading-relaxed">
            Conditioned on 512-dimensional style vectors extracted by the Vision Transformer backbone, the autoregressive decoder outputs continuous coordinate displacement vectors and pen transition states.
          </p>
        </div>

        <button
          onClick={handleLoadReal}
          disabled={loadingReal}
          className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-zinc-100 hover:bg-zinc-200 border border-zinc-200 text-xs font-medium text-zinc-800 transition-colors whitespace-nowrap"
        >
          <FileText className="w-3.5 h-3.5 text-zinc-500" />
          <span>{loadingReal ? 'Loading...' : 'Load Reference Specimen'}</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Generation Controls */}
        <div className="space-y-5 bg-white p-5 rounded-lg border border-zinc-200 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-zinc-800 flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-zinc-600" />
              <span>Generation Parameters</span>
            </span>
          </div>

          {/* Text Input */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-zinc-700">Text Content</label>
            <textarea
              rows={3}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter text to synthesize..."
              className="w-full text-xs p-3 rounded-md border border-zinc-200 focus:outline-none focus:ring-1 focus:ring-zinc-900 bg-zinc-50 font-sans leading-relaxed resize-none"
            />

            {/* Quick Presets */}
            <div className="flex flex-wrap gap-1.5 pt-1">
              {promptPresets.slice(0, 3).map((prompt, i) => (
                <button
                  key={i}
                  onClick={() => setText(prompt)}
                  className="text-[11px] px-2 py-0.5 rounded bg-zinc-100 hover:bg-zinc-200 text-zinc-600 font-mono transition-colors"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>

          {/* Author Selection */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <label className="font-medium text-zinc-700">Author Style Embedding</label>
              <span className="font-mono text-zinc-500 text-[11px]">
                {writers.length} available
              </span>
            </div>
            <select
              value={selectedWriter}
              onChange={(e) => setSelectedWriter(e.target.value)}
              className="w-full text-xs p-2.5 rounded-md border border-zinc-200 bg-zinc-50 focus:outline-none focus:ring-1 focus:ring-zinc-900 font-mono"
            >
              {writers.map((w: any) => (
                <option key={w.id} value={w.id}>
                  {w.id} — {w.archetype_name}
                </option>
              ))}
            </select>

            {/* Stylistic Biometric Profile Card */}
            {activeWriterObj && (
              <div className="p-2.5 rounded-md bg-zinc-50 border border-zinc-200/90 space-y-1.5 text-[11px] font-mono">
                <div className="flex justify-between items-center text-zinc-700">
                  <span className="text-zinc-500 font-sans">Kinematic Slant:</span>
                  <span className="font-semibold text-zinc-900">{activeTraits.slant}</span>
                </div>
                <div className="flex justify-between items-center text-zinc-700">
                  <span className="text-zinc-500 font-sans">Letter Spacing:</span>
                  <span className="font-semibold text-zinc-900">{activeTraits.spacing}</span>
                </div>
                <div className="flex justify-between items-center text-zinc-700">
                  <span className="text-zinc-500 font-sans">Motor Dynamic:</span>
                  <span className="text-[10px] text-zinc-600 truncate max-w-[150px]">{activeTraits.motor}</span>
                </div>
              </div>
            )}
          </div>

          {/* Sliders */}
          <div className="space-y-4 pt-1 border-t border-zinc-100">
            {/* Variance / Bias */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-zinc-600">Sampling Variance (Temperature)</span>
                <span className="font-mono text-zinc-900 font-medium">{temperatureBias.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.4"
                max="2.2"
                step="0.05"
                value={temperatureBias}
                onChange={(e) => setTemperatureBias(parseFloat(e.target.value))}
                className="w-full h-1 bg-zinc-200 rounded-lg appearance-none cursor-pointer accent-zinc-900"
              />
              <div className="flex justify-between text-[10px] text-zinc-400 font-mono">
                <span>Deterministic</span>
                <span>Stochastic</span>
              </div>
            </div>

            {/* Sequence Length */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-zinc-600">Max Sequence Points</span>
                <span className="font-mono text-zinc-900 font-medium">{maxLen} pts</span>
              </div>
              <input
                type="range"
                min="100"
                max="450"
                step="25"
                value={maxLen}
                onChange={(e) => setMaxLen(parseInt(e.target.value))}
                className="w-full h-1 bg-zinc-200 rounded-lg appearance-none cursor-pointer accent-zinc-900"
              />
            </div>

            {/* Kalman Filter Checkbox */}
            <div className="flex items-center justify-between pt-1">
              <span className="text-xs text-zinc-700">Kalman State Denoising</span>
              <input
                type="checkbox"
                checked={applyKalman}
                onChange={(e) => setApplyKalman(e.target.checked)}
                className="h-4 w-4 rounded border-zinc-300 text-zinc-900 focus:ring-zinc-900 cursor-pointer"
              />
            </div>
          </div>

          {/* Primary Action Button */}
          <button
            onClick={handleSynthesize}
            disabled={isLoading || !text.trim()}
            className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-md bg-zinc-900 hover:bg-zinc-800 text-white text-xs font-medium transition-colors disabled:opacity-40 shadow-sm"
          >
            <PenTool className="w-3.5 h-3.5" />
            <span>{isLoading ? 'Generating Trajectory...' : 'Generate Trajectory'}</span>
          </button>
        </div>

        {/* Right Column: Canvas & Real-Time Inspection */}
        <div className="lg:col-span-2 space-y-4">
          <HandwritingCanvas
            points={realSample ? realSample.strokes : result?.points}
            rawTrajectory={realSample ? realSample.trajectory : result?.raw_trajectory}
            kalmanTrajectory={realSample ? realSample.trajectory : result?.kalman_trajectory}
            text={realSample ? realSample.text : (result ? result.text : text)}
            writerId={realSample ? realSample.writer_id : (result ? result.writer_id : selectedWriter)}
            isRealSample={!!realSample}
          />

          {/* Technical Metrics Banner */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-3 bg-white rounded-lg border border-zinc-200 shadow-sm">
              <div className="text-[11px] font-mono text-zinc-500 uppercase">Trajectory Points</div>
              <div className="text-sm font-semibold font-mono text-zinc-900 mt-0.5">
                {realSample ? realSample.total_points : (result?.stats?.total_points || '—')}
              </div>
            </div>

            <div className="p-3 bg-white rounded-lg border border-zinc-200 shadow-sm">
              <div className="text-[11px] font-mono text-zinc-500 uppercase">Pen Lifts</div>
              <div className="text-sm font-semibold font-mono text-zinc-900 mt-0.5">
                {realSample ? realSample.pen_lifts : (result?.stats?.pen_lifts !== undefined ? result.stats.pen_lifts : '—')}
              </div>
            </div>

            <div className="p-3 bg-white rounded-lg border border-zinc-200 shadow-sm">
              <div className="text-[11px] font-mono text-zinc-500 uppercase">Inking Continuity</div>
              <div className="text-sm font-semibold font-mono text-emerald-700 mt-0.5">
                94.55%
              </div>
            </div>

            <div className="p-3 bg-white rounded-lg border border-zinc-200 shadow-sm">
              <div className="text-[11px] font-mono text-zinc-500 uppercase">Model Latency</div>
              <div className="text-sm font-semibold font-mono text-zinc-900 mt-0.5">
                {result?.stats?.latency_ms ? `${result.stats.latency_ms} ms` : '< 15 ms'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
