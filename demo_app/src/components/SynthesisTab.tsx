"use client";

import React, { useState } from 'react';
import { Sparkles, Sliders, Wand2, ShieldCheck, RefreshCw, Cpu } from 'lucide-react';
import HandwritingCanvas from './HandwritingCanvas';

interface SynthesisTabProps {
  writers: any[];
  archetypes: string[];
}

export default function SynthesisTab({ writers, archetypes }: SynthesisTabProps) {
  const [text, setText] = useState("Machine Learning Unit 3 4 5 Integration");
  const [selectedWriter, setSelectedWriter] = useState(writers[0]?.id || "a01-000");
  const [temperatureBias, setTemperatureBias] = useState(1.4);
  const [maxLen, setMaxLen] = useState(300);
  const [applyKalman, setApplyKalman] = useState(true);

  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  // Real sample comparison state
  const [realSample, setRealSample] = useState<any>(null);
  const [loadingReal, setLoadingReal] = useState(false);

  const quickPrompts = [
    "Machine Learning Sem 5 Master Project",
    "Expectation Maximization & Gaussian Mixtures",
    "Hidden Markov Models and Viterbi Decoding",
    "Ensemble Stacking with Random Forests and SVM",
    "The quick brown fox jumps over the lazy dog"
  ];

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
    } catch (err) {
      console.error("Synthesis error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoadReal = async () => {
    setLoadingReal(true);
    try {
      const randIdx = Math.floor(Math.random() * 200);
      const res = await fetch(`/api/samples/real?index=${randIdx}`);
      const data = await res.json();
      setRealSample(data);
    } catch (err) {
      console.error("Error loading real sample:", err);
    } finally {
      setLoadingReal(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Intro Banner */}
      <div className="p-4 rounded-xl bg-white border border-slate-200 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 shadow-sm">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-base font-bold text-slate-900 tracking-tight">
              Biometric Vector Handwriting Synthesis Studio
            </h2>
            <span className="px-2 py-0.5 rounded text-[11px] font-mono font-medium bg-blue-50 text-blue-700 border border-blue-200">
              Style-Disentangled Transformer (SDT)
            </span>
          </div>
          <p className="text-xs text-slate-600 mt-1 max-w-2xl leading-relaxed">
            Conditioned on Vision Transformer (ViT) 512-D style vectors from IAM authors, our 20-component Gaussian Mixture Density Network (MDN) synthesizes continuous (dx, dy, p1, p2, p3) stroke trajectories.
          </p>
        </div>

        <button
          onClick={handleLoadReal}
          disabled={loadingReal}
          className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-slate-100 hover:bg-slate-200 border border-slate-200 text-xs font-medium text-slate-800 transition-colors whitespace-nowrap"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loadingReal ? 'animate-spin' : ''}`} />
          <span>Load Real Ground Truth Sample</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Generation Controls */}
        <div className="lg:col-span-1 space-y-5">
          {/* Text Input */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-800 mb-1.5 uppercase tracking-wider">
                Digital Text Input
              </label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={3}
                className="w-full text-xs font-mono p-3 rounded-lg border border-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all text-slate-900 resize-none bg-slate-50/50"
                placeholder="Enter text to synthesize into biometric handwriting..."
              />
            </div>

            {/* Quick Prompts */}
            <div>
              <span className="block text-[11px] font-medium text-slate-500 mb-1.5">
                Suggested Review Prompts:
              </span>
              <div className="flex flex-wrap gap-1.5">
                {quickPrompts.map((p, i) => (
                  <button
                    key={i}
                    onClick={() => setText(p)}
                    className="px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 text-[10px] text-slate-600 font-medium transition-colors text-left truncate max-w-full"
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>

            {/* Writer Style Selector */}
            <div>
              <label className="block text-xs font-semibold text-slate-800 mb-1.5 uppercase tracking-wider">
                Writer Identity (512-D Style Manifold)
              </label>
              <select
                value={selectedWriter}
                onChange={(e) => setSelectedWriter(e.target.value)}
                className="w-full text-xs font-mono p-2.5 rounded-lg border border-slate-200 bg-white text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500"
              >
                {writers.slice(0, 30).map((w) => (
                  <option key={w.id} value={w.id}>
                    Writer {w.id} — {w.archetype_name}
                  </option>
                ))}
              </select>
            </div>

            {/* Sliders */}
            <div className="space-y-3 pt-2 border-t border-slate-100">
              <div>
                <div className="flex justify-between items-center text-xs mb-1">
                  <span className="font-medium text-slate-700">Neatness Bias (Temperature)</span>
                  <span className="font-mono text-blue-600 font-bold">{temperatureBias.toFixed(1)}x</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="2.5"
                  step="0.1"
                  value={temperatureBias}
                  onChange={(e) => setTemperatureBias(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
                <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                  <span>Natural Jitter (0.5x)</span>
                  <span>Ultra-Neat Low Entropy (2.5x)</span>
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center text-xs mb-1">
                  <span className="font-medium text-slate-700">Max Sequence Coordinate Length</span>
                  <span className="font-mono text-blue-600 font-bold">{maxLen} pts</span>
                </div>
                <input
                  type="range"
                  min="100"
                  max="450"
                  step="25"
                  value={maxLen}
                  onChange={(e) => setMaxLen(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
              </div>

              <div className="flex items-center justify-between pt-1">
                <span className="text-xs font-medium text-slate-700">Apply Kalman Filter (LDS Denoising)</span>
                <input
                  type="checkbox"
                  checked={applyKalman}
                  onChange={(e) => setApplyKalman(e.target.checked)}
                  className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 accent-blue-600 cursor-pointer"
                />
              </div>
            </div>

            {/* Synthesize Button */}
            <button
              onClick={handleSynthesize}
              disabled={isLoading || !text.trim()}
              className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium text-xs shadow-sm transition-all disabled:opacity-50"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Generating Coordinates via SDT...</span>
                </>
              ) : (
                <>
                  <Wand2 className="w-4 h-4" />
                  <span>Synthesize Biometric Handwriting</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Columns: Canvas & Results */}
        <div className="lg:col-span-2 space-y-6">
          {/* Synthesized Output Canvas */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-blue-600" />
                <span>Generated Output Canvas</span>
              </span>
              {result && (
                <div className="flex items-center gap-3 text-xs font-mono text-slate-500">
                  <span>Latency: {result.stats.inference_time_sec}s</span>
                  <span>·</span>
                  <span>Pen Lifts: {result.stats.pen_lifts}</span>
                </div>
              )}
            </div>

            <HandwritingCanvas
              points={result?.points || []}
              rawTrajectory={result?.raw_trajectory || []}
              kalmanTrajectory={result?.kalman_trajectory || []}
              text={text}
              writerId={selectedWriter}
            />
          </div>

          {/* Side-by-side Visual Turing Test Ground Truth */}
          {realSample && (
            <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-sm space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
                  <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
                  <span>Visual Turing Benchmark (Ground Truth IAM Online)</span>
                </span>
                <span className="font-mono text-xs text-slate-500">
                  Target: &ldquo;{realSample.text}&rdquo; (Author: {realSample.writer_id})
                </span>
              </div>
              <HandwritingCanvas
                rawTrajectory={realSample.trajectory}
                kalmanTrajectory={realSample.trajectory}
                text={realSample.text}
                writerId={realSample.writer_id}
                isRealSample={true}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

