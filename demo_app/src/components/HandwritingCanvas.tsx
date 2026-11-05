"use client";

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Play, Pause, RotateCcw, Eye, Download, Sparkles, Filter } from 'lucide-react';

interface HandwritingCanvasProps {
  points?: number[][]; // [dx, dy, pen]
  rawTrajectory?: number[][]; // [x, y]
  kalmanTrajectory?: number[][]; // [x, y]
  viterbiStates?: number[];
  text?: string;
  writerId?: string;
  isRealSample?: boolean;
}

export default function HandwritingCanvas({
  points = [],
  rawTrajectory = [],
  kalmanTrajectory = [],
  viterbiStates = [],
  text = "",
  writerId = "",
  isRealSample = false
}: HandwritingCanvasProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(1.0); // 0 to 1
  const [showPenLifts, setShowPenLifts] = useState(false);
  const [useKalman, setUseKalman] = useState(true);
  const [colorByHMM, setColorByHMM] = useState(false);

  const animRef = useRef<number | null>(null);

  // Active trajectory
  const trajectory = useMemo(() => {
    if (useKalman && kalmanTrajectory.length > 0) {
      return kalmanTrajectory;
    }
    return rawTrajectory;
  }, [useKalman, kalmanTrajectory, rawTrajectory]);

  // Compute bounding box for auto-scaling
  const bounds = useMemo(() => {
    if (trajectory.length === 0) return { minX: 0, maxX: 400, minY: -50, maxY: 50, width: 400, height: 100 };
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    trajectory.forEach(([x, y]) => {
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    });
    const paddingX = Math.max(20, (maxX - minX) * 0.05);
    const paddingY = Math.max(20, (maxY - minY) * 0.15);
    return {
      minX: minX - paddingX,
      maxX: maxX + paddingX,
      minY: minY - paddingY,
      maxY: maxY + paddingY,
      width: Math.max(100, maxX - minX + 2 * paddingX),
      height: Math.max(80, maxY - minY + 2 * paddingY),
    };
  }, [trajectory]);

  // Playback animation loop
  useEffect(() => {
    if (!isPlaying) {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      return;
    }

    let start: number | null = null;
    const duration = Math.max(1500, points.length * 14); // Dynamic duration based on stroke count

    const step = (timestamp: number) => {
      if (!start) start = timestamp;
      const elapsed = timestamp - start;
      const nextProgress = Math.min(1.0, elapsed / duration);
      setProgress(nextProgress);

      if (nextProgress < 1.0) {
        animRef.current = requestAnimationFrame(step);
      } else {
        setIsPlaying(false);
      }
    };

    animRef.current = requestAnimationFrame(step);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [isPlaying, points.length]);

  const visibleCount = Math.max(1, Math.floor(trajectory.length * progress));

  // Build SVG path segments
  const segments = useMemo(() => {
    const list: Array<{ path: string; isLift: boolean; color: string }> = [];
    if (trajectory.length < 2) return list;

    const HMM_COLORS = ['#dc2626', '#2563eb', '#16a34a', '#f59e0b']; // Downstroke, Upward, Ligature, Airborne
    let currentD = `M ${trajectory[0][0]} ${-trajectory[0][1]}`;
    let isCurrentLift = false;
    let currentColor = '#0f172a';

    for (let i = 1; i < visibleCount; i++) {
      const prevPt = trajectory[i - 1];
      const pt = trajectory[i];
      const penState = points[i - 1] ? points[i - 1][2] : 0;
      const isLift = penState === 1;

      let segColor = '#0f172a';
      if (colorByHMM && viterbiStates[i] !== undefined) {
        segColor = HMM_COLORS[viterbiStates[i] % HMM_COLORS.length];
      }

      if (isLift !== isCurrentLift || (colorByHMM && segColor !== currentColor)) {
        if (currentD) {
          list.push({ path: currentD, isLift: isCurrentLift, color: currentColor });
        }
        currentD = `M ${prevPt[0]} ${-prevPt[1]} L ${pt[0]} ${-pt[1]}`;
        isCurrentLift = isLift;
        currentColor = segColor;
      } else {
        currentD += ` L ${pt[0]} ${-pt[1]}`;
      }
    }

    if (currentD) {
      list.push({ path: currentD, isLift: isCurrentLift, color: currentColor });
    }

    return list;
  }, [trajectory, visibleCount, points, colorByHMM, viterbiStates]);

  const handleExportSVG = () => {
    const svgEl = document.getElementById("handwriting-svg");
    if (!svgEl) return;
    const serializer = new XMLSerializer();
    const source = serializer.serializeToString(svgEl);
    const blob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `handwriting_${writerId || "sample"}.svg`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
      {/* Top Bar with metadata & controls */}
      <div className="px-4 py-3 bg-slate-50/80 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-2">
          <span className="font-mono font-semibold text-slate-800">
            {isRealSample ? 'IAM Online Ground Truth' : 'Generative SDT Vector Canvas'}
          </span>
          {writerId && (
            <span className="px-2 py-0.5 rounded bg-slate-200 text-slate-700 font-mono text-[11px]">
              Writer: {writerId}
            </span>
          )}
          <span className="text-slate-400">·</span>
          <span className="font-mono text-slate-500">
            {trajectory.length} coordinates
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Kalman Filter Toggle */}
          <button
            onClick={() => setUseKalman(!useKalman)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium border transition-colors ${
              useKalman
                ? 'bg-blue-50 border-blue-200 text-blue-700'
                : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-100'
            }`}
            title="Toggle Kalman Filter LDS Trajectory Smoothing"
          >
            <Filter className="w-3.5 h-3.5" />
            <span>Kalman LDS: {useKalman ? 'ON' : 'OFF'}</span>
          </button>

          {/* Pen Lifts Toggle */}
          <button
            onClick={() => setShowPenLifts(!showPenLifts)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium border transition-colors ${
              showPenLifts
                ? 'bg-amber-50 border-amber-200 text-amber-700'
                : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-100'
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            <span>Air-Strokes</span>
          </button>

          {/* HMM State Colors Toggle */}
          <button
            onClick={() => setColorByHMM(!colorByHMM)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium border transition-colors ${
              colorByHMM
                ? 'bg-purple-50 border-purple-200 text-purple-700'
                : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-100'
            }`}
            title="Color code by Unit 4 HMM Viterbi Motor States"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>HMM States</span>
          </button>

          {/* Export SVG */}
          <button
            onClick={handleExportSVG}
            className="p-1.5 rounded text-slate-600 hover:text-slate-900 hover:bg-slate-200/60 border border-slate-200"
            title="Export Vector SVG"
          >
            <Download className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* SVG Canvas Area */}
      <div className="relative p-6 lined-paper min-h-[220px] flex items-center justify-center overflow-x-auto">
        {trajectory.length === 0 ? (
          <div className="text-center text-slate-400 py-12 font-mono text-sm">
            No handwriting generated yet. Enter text and click Synthesize.
          </div>
        ) : (
          <svg
            id="handwriting-svg"
            viewBox={`${bounds.minX} ${-bounds.maxY} ${bounds.width} ${bounds.height}`}
            className="w-full h-auto max-h-[300px]"
            style={{ filter: 'drop-shadow(0px 1px 1.5px rgba(15, 23, 42, 0.12))' }}
          >
            {/* Draw faint baseline */}
            <line
              x1={bounds.minX}
              y1={0}
              x2={bounds.maxX}
              y2={0}
              stroke="#e2e8f0"
              strokeDasharray="4,4"
              strokeWidth="0.8"
            />

            {/* Stroke segments */}
            {segments.map((seg, idx) => {
              if (seg.isLift && !showPenLifts) return null;
              return (
                <path
                  key={idx}
                  d={seg.path}
                  fill="none"
                  stroke={seg.isLift ? '#94a3b8' : seg.color}
                  strokeWidth={seg.isLift ? '0.8' : '2.2'}
                  strokeDasharray={seg.isLift ? '3,3' : 'none'}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  opacity={seg.isLift ? 0.6 : 0.95}
                />
              );
            })}

            {/* Current pen cursor during playback */}
            {isPlaying && visibleCount > 0 && (
              <circle
                cx={trajectory[visibleCount - 1][0]}
                cy={-trajectory[visibleCount - 1][1]}
                r="3.5"
                fill="#2563eb"
                stroke="#ffffff"
                strokeWidth="1.5"
                className="animate-ping"
              />
            )}
          </svg>
        )}
      </div>

      {/* Bottom Playback Bar */}
      <div className="px-4 py-2.5 bg-slate-50 border-t border-slate-200 flex items-center gap-3">
        <button
          onClick={() => {
            if (progress >= 1.0) setProgress(0);
            setIsPlaying(!isPlaying);
          }}
          disabled={trajectory.length === 0}
          className="p-1.5 rounded-lg bg-slate-900 text-white hover:bg-slate-800 disabled:opacity-40 transition-colors"
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
        </button>

        <button
          onClick={() => {
            setIsPlaying(false);
            setProgress(0);
          }}
          disabled={trajectory.length === 0}
          className="p-1.5 rounded-lg text-slate-600 hover:bg-slate-200 border border-slate-200 disabled:opacity-40 transition-colors"
          title="Reset Animation"
        >
          <RotateCcw className="w-4 h-4" />
        </button>

        {/* Scrubbing slider */}
        <div className="flex-1 flex items-center gap-2">
          <input
            type="range"
            min="0"
            max="1"
            step="0.005"
            value={progress}
            onChange={(e) => {
              setIsPlaying(false);
              setProgress(parseFloat(e.target.value));
            }}
            disabled={trajectory.length === 0}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
          <span className="font-mono text-xs text-slate-500 w-12 text-right">
            {Math.round(progress * 100)}%
          </span>
        </div>
      </div>
    </div>
  );
}

