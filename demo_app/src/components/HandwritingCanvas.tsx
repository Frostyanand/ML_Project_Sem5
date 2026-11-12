"use client";

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Play, Pause, RotateCcw, Download, Eye, EyeOff, Layers, FileCode } from 'lucide-react';

interface HandwritingCanvasProps {
  points?: number[][]; // [dx, dy, pen]
  rawTrajectory?: number[][]; // [x, y]
  kalmanTrajectory?: number[][]; // [x, y]
  viterbiStates?: number[];
  text?: string;
  writerId?: string;
  isRealSample?: boolean;
  compact?: boolean;
  titleBadge?: string;
}

export default function HandwritingCanvas({
  points = [],
  rawTrajectory = [],
  kalmanTrajectory = [],
  viterbiStates = [],
  text = "",
  writerId = "",
  isRealSample = false,
  compact = false,
  titleBadge = ""
}: HandwritingCanvasProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(1.0); // 0 to 1
  const [showPenLifts, setShowPenLifts] = useState(false);
  const [useKalman, setUseKalman] = useState(true);
  const [colorByHMM, setColorByHMM] = useState(false);
  const [paperBackground, setPaperBackground] = useState(true);

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
    if (trajectory.length === 0) return { minX: 0, maxX: 500, minY: -50, maxY: 50, width: 500, height: 110 };
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    trajectory.forEach(([x, y]) => {
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    });
    const paddingX = Math.max(25, (maxX - minX) * 0.05);
    const paddingY = Math.max(25, (maxY - minY) * 0.20);
    return {
      minX: minX - paddingX,
      maxX: maxX + paddingX,
      minY: minY - paddingY,
      maxY: maxY + paddingY,
      width: Math.max(120, maxX - minX + 2 * paddingX),
      height: Math.max(90, maxY - minY + 2 * paddingY),
    };
  }, [trajectory]);

  // Playback animation loop
  useEffect(() => {
    if (!isPlaying) {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      return;
    }

    let start: number | null = null;
    const duration = Math.max(1600, points.length * 15);

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

    const HMM_COLORS = ['#b91c1c', '#1d4ed8', '#15803d', '#b45309']; // Downstroke, Loop, Ligature, Air
    let currentD = `M ${trajectory[0][0]} ${-trajectory[0][1]}`;
    let isCurrentLift = false;
    let currentColor = '#18181b';

    for (let i = 1; i < visibleCount; i++) {
      const prevPt = trajectory[i - 1];
      const pt = trajectory[i];
      const penState = points[i - 1] ? points[i - 1][2] : 0;
      const isLift = penState === 1;

      let segColor = '#18181b';
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
    <div className="bg-white rounded-lg border border-zinc-200 overflow-hidden shadow-sm">
      {/* Top Bar with metadata & controls */}
      <div className="px-4 py-2.5 bg-zinc-50 border-b border-zinc-200 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-2 font-mono">
          <span className="font-medium text-zinc-900">
            {isRealSample ? 'Reference Specimen' : 'Generated Trajectory'}
          </span>
          {titleBadge && (
            <span className="px-1.5 py-0.5 rounded bg-zinc-200/60 text-zinc-700 text-[10px] font-sans">
              {titleBadge}
            </span>
          )}
          {writerId && (
            <span className="px-1.5 py-0.5 rounded bg-zinc-200/80 text-zinc-700 text-[11px]">
              Author: {writerId}
            </span>
          )}
          <span className="text-zinc-300">·</span>
          <span className="text-zinc-500">
            {trajectory.length} points
          </span>
        </div>

        {/* View Controls */}
        <div className="flex items-center gap-2">
          {/* Kalman Filter Toggle */}
          <button
            onClick={() => setUseKalman(!useKalman)}
            className={`px-2 py-1 rounded text-xs font-mono font-medium border transition-colors ${
              useKalman
                ? 'bg-zinc-900 border-zinc-900 text-white'
                : 'bg-white border-zinc-200 text-zinc-600 hover:bg-zinc-100'
            }`}
          >
            Kalman: {useKalman ? 'ON' : 'OFF'}
          </button>

          {/* Pen Lifts Toggle */}
          <button
            onClick={() => setShowPenLifts(!showPenLifts)}
            className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-mono font-medium border transition-colors ${
              showPenLifts
                ? 'bg-amber-100/70 border-amber-300 text-amber-800'
                : 'bg-white border-zinc-200 text-zinc-600 hover:bg-zinc-100'
            }`}
          >
            {showPenLifts ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3 text-zinc-400" />}
            <span>Air Strokes</span>
          </button>

          {/* HMM State Colors Toggle */}
          <button
            onClick={() => setColorByHMM(!colorByHMM)}
            className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-mono font-medium border transition-colors ${
              colorByHMM
                ? 'bg-purple-100/70 border-purple-300 text-purple-800'
                : 'bg-white border-zinc-200 text-zinc-600 hover:bg-zinc-100'
            }`}
          >
            <Layers className="w-3 h-3" />
            <span>HMM States</span>
          </button>

          {/* Canvas Texture Toggle */}
          <button
            onClick={() => setPaperBackground(!paperBackground)}
            className="px-2 py-1 rounded text-xs font-mono border border-zinc-200 bg-white text-zinc-600 hover:bg-zinc-100"
            title="Toggle Ruled Paper Lines"
          >
            {paperBackground ? 'Ruled Paper' : 'Plain'}
          </button>

          {/* Export SVG */}
          <button
            onClick={handleExportSVG}
            disabled={trajectory.length === 0}
            className="flex items-center gap-1 px-2 py-1 rounded text-xs font-mono font-medium border border-zinc-200 bg-white text-zinc-700 hover:bg-zinc-100 disabled:opacity-40"
          >
            <Download className="w-3 h-3" />
            <span>Export SVG</span>
          </button>
        </div>
      </div>

      {/* SVG Vector Canvas */}
      <div className={`relative ${compact ? 'min-h-[170px] max-h-[220px] p-3' : 'min-h-[260px] max-h-[380px] p-6'} w-full flex items-center justify-center overflow-hidden ${
        paperBackground ? 'bg-[#fcfdfd]' : 'bg-white'
      }`}>
        {/* Subtle Ruled Lines */}
        {paperBackground && (
          <div className="absolute inset-0 pointer-events-none opacity-40">
            <div className="w-full h-full bg-[linear-gradient(to_bottom,transparent_31px,#e4e4e7_31px,#e4e4e7_32px)] bg-[size:100%_32px]"></div>
            <div className="absolute left-8 top-0 bottom-0 w-[1px] bg-red-200"></div>
          </div>
        )}

        {trajectory.length === 0 ? (
          <div className="flex flex-col items-center justify-center text-zinc-400 py-12 font-mono text-xs">
            <FileCode className="w-6 h-6 mb-2 text-zinc-300" />
            <span>No trajectory data. Enter text and click Generate.</span>
          </div>
        ) : (
          <svg
            id={`handwriting-svg-${writerId || 'default'}-${isRealSample ? 'real' : 'synth'}`}
            viewBox={`${bounds.minX} ${-bounds.maxY} ${bounds.width} ${bounds.height}`}
            className={`w-full h-full ${compact ? 'max-h-[190px]' : 'max-h-[300px]'} transition-all duration-300`}
            style={{ filter: 'drop-shadow(0px 1px 1px rgba(0,0,0,0.06))' }}
          >
            {/* Draw Path Segments */}
            {segments.map((seg, i) => {
              if (seg.isLift) {
                if (!showPenLifts) return null;
                return (
                  <path
                    key={i}
                    d={seg.path}
                    fill="none"
                    stroke="#d97706"
                    strokeWidth="0.8"
                    strokeDasharray="2,2"
                    strokeOpacity="0.6"
                  />
                );
              }
              return (
                <path
                  key={i}
                  d={seg.path}
                  fill="none"
                  stroke={seg.color}
                  strokeWidth="1.9"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              );
            })}

            {/* Current pen tip head */}
            {progress < 1.0 && trajectory[visibleCount - 1] && (
              <circle
                cx={trajectory[visibleCount - 1][0]}
                cy={-trajectory[visibleCount - 1][1]}
                r="2.5"
                fill="#2563eb"
              />
            )}
          </svg>
        )}
      </div>

      {/* Bottom Timeline & Metrics Toolbar */}
      <div className="px-4 py-2.5 bg-zinc-50/80 border-t border-zinc-200 flex flex-wrap items-center justify-between gap-4 text-xs">
        {/* Playback Controls */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => {
              if (progress >= 1.0) setProgress(0);
              setIsPlaying(!isPlaying);
            }}
            disabled={trajectory.length === 0}
            className="flex items-center gap-1.5 px-3 py-1 rounded-md bg-zinc-900 text-white text-xs font-medium hover:bg-zinc-800 disabled:opacity-40 transition-colors"
          >
            {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span>{isPlaying ? 'Pause' : 'Animate'}</span>
          </button>

          <button
            onClick={() => {
              setIsPlaying(false);
              setProgress(1.0);
            }}
            disabled={trajectory.length === 0}
            className="p-1 rounded text-zinc-500 hover:text-zinc-900 hover:bg-zinc-200/60"
            title="Reset to Full Stroke"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>

          {/* Progress scrubber */}
          <div className="flex items-center gap-2">
            <input
              type="range"
              min="0.05"
              max="1.0"
              step="0.01"
              value={progress}
              onChange={(e) => {
                setIsPlaying(false);
                setProgress(parseFloat(e.target.value));
              }}
              className="w-24 sm:w-36 h-1 bg-zinc-200 rounded-lg appearance-none cursor-pointer accent-zinc-900"
            />
            <span className="font-mono text-[11px] text-zinc-500 w-9">
              {Math.round(progress * 100)}%
            </span>
          </div>
        </div>

        {/* Technical Sequence Metrics */}
        {text && (
          <div className="text-zinc-500 font-mono text-[11px] truncate max-w-xs sm:max-w-md">
            Text: &quot;<span className="text-zinc-800">{text}</span>&quot;
          </div>
        )}
      </div>
    </div>
  );
}
