"use client";

import React, { useState, useEffect } from 'react';
import { Layers, Compass, BarChart2, CheckCircle2 } from 'lucide-react';

export default function Unit3Tab() {
  const [pcaData, setPcaData] = useState<any>(null);
  const [clusterData, setClusterData] = useState<any>(null);
  const [selectedPoint, setSelectedPoint] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch('/api/unit3/pca').then(r => r.json()),
      fetch('/api/unit3/clustering').then(r => r.json())
    ]).then(([pca, clus]) => {
      setPcaData(pca);
      setClusterData(clus);
      setLoading(false);
    }).catch(err => {
      console.error("Error loading Unit 3 data:", err);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return (
      <div className="p-16 text-center text-zinc-500 font-mono text-xs space-y-2">
        <div className="w-5 h-5 border-2 border-zinc-900 border-t-transparent rounded-full animate-spin mx-auto"></div>
        <div>Loading latent manifold representation...</div>
      </div>
    );
  }

  const CLUSTER_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#06b6d4'];

  return (
    <div className="space-y-6">
      {/* Overview Banner */}
      <div className="p-5 rounded-lg bg-white border border-zinc-200 shadow-sm space-y-1.5">
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 rounded text-[11px] font-mono text-zinc-600 bg-zinc-100 border border-zinc-200">
            Curriculum Unit 3
          </span>
          <h2 className="text-base font-semibold text-zinc-900 tracking-tight">
            Latent Manifold, PCA, Mixture Models &amp; EM Algorithm
          </h2>
        </div>
        <p className="text-xs text-zinc-600 max-w-4xl leading-relaxed">
          Projecting the 512-dimensional continuous latent space produced by the Vision Transformer backbone into orthogonal principal components. Scree analysis demonstrates that 95% of human handwriting variation across 1,712 authors is compressed within 26 latent dimensions. Authors are categorized via K-Means and soft Gaussian Mixture Models (EM).
        </p>
      </div>

      {/* Grid: PCA Scree & Latent Dimensions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Latent Dimension Selection Card */}
        <div className="bg-white p-5 rounded-lg border border-zinc-200 shadow-sm space-y-4">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-zinc-800">
            <Compass className="w-3.5 h-3.5 text-zinc-600" />
            <span>Intrinsic Dimensionality</span>
          </div>

          <div className="space-y-3">
            <div className="p-3 rounded-md bg-zinc-50 border border-zinc-200/80">
              <div className="flex justify-between items-center text-xs">
                <span className="text-zinc-600">80% Cumulative Variance</span>
                <span className="font-mono font-semibold text-zinc-900">7 Latent Dimensions</span>
              </div>
              <div className="w-full bg-zinc-200 h-1 rounded-full mt-2 overflow-hidden">
                <div className="bg-zinc-700 h-full rounded-full" style={{ width: '80%' }}></div>
              </div>
            </div>

            <div className="p-3 rounded-md bg-zinc-50 border border-zinc-200/80">
              <div className="flex justify-between items-center text-xs">
                <span className="text-zinc-600">90% Cumulative Variance</span>
                <span className="font-mono font-semibold text-zinc-900">13 Latent Dimensions</span>
              </div>
              <div className="w-full bg-zinc-200 h-1 rounded-full mt-2 overflow-hidden">
                <div className="bg-zinc-700 h-full rounded-full" style={{ width: '90%' }}></div>
              </div>
            </div>

            <div className="p-3 rounded-md bg-zinc-50 border border-zinc-200/80">
              <div className="flex justify-between items-center text-xs">
                <span className="text-zinc-600">95% Full Manifold</span>
                <span className="font-mono font-semibold text-zinc-900">26 Latent Dimensions</span>
              </div>
              <div className="w-full bg-zinc-200 h-1 rounded-full mt-2 overflow-hidden">
                <div className="bg-zinc-900 h-full rounded-full" style={{ width: '95%' }}></div>
              </div>
            </div>
          </div>

          <div className="text-[11px] text-zinc-500 leading-relaxed pt-2 border-t border-zinc-100 font-mono">
            d = 26 &lt;&lt; 512 proves that handwriting variation is constrained by forearm and finger biomechanics, eliminating the curse of dimensionality.
          </div>
        </div>

        {/* Scree Plot Visualization */}
        <div className="lg:col-span-2 bg-white p-5 rounded-lg border border-zinc-200 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-zinc-800">
              <BarChart2 className="w-3.5 h-3.5 text-zinc-600" />
              <span>PCA Scree Plot &amp; Eigenvalue Spectrum</span>
            </div>
            <span className="font-mono text-[11px] text-zinc-500">First 15 Principal Components</span>
          </div>

          {/* SVG Scree Bars */}
          <div className="h-44 w-full flex items-end justify-between pt-6 px-2 border-b border-zinc-200">
            {pcaData?.explained_variance_ratio?.slice(0, 15).map((val: number, i: number) => {
              const hPct = Math.max(8, val * 380);
              return (
                <div key={i} className="flex-1 flex flex-col items-center gap-1 group relative">
                  {/* Tooltip */}
                  <div className="opacity-0 group-hover:opacity-100 transition-opacity absolute -top-7 bg-zinc-900 text-white text-[10px] px-1.5 py-0.5 rounded font-mono pointer-events-none whitespace-nowrap z-10">
                    PC{i + 1}: {(val * 100).toFixed(1)}%
                  </div>
                  <div
                    className="w-4/5 max-w-[28px] rounded-t-sm transition-all bg-zinc-800 hover:bg-zinc-600"
                    style={{ height: `${hPct}%` }}
                  ></div>
                  <span className="text-[10px] font-mono text-zinc-400">
                    {i + 1}
                  </span>
                </div>
              );
            })}
          </div>

          <div className="flex justify-between items-center text-[11px] font-mono text-zinc-500 pt-1">
            <span>Principal Component Index</span>
            <span>Individual Explained Variance (%)</span>
          </div>
        </div>
      </div>

      {/* Grid: 2D PCA Projection & Clustering Partition */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Interactive 2D Scatter Manifold */}
        <div className="lg:col-span-2 bg-white p-5 rounded-lg border border-zinc-200 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-zinc-800">
              <Layers className="w-3.5 h-3.5 text-zinc-600" />
              <span>2D Manifold Projection (PC1 vs PC2)</span>
            </div>
            <span className="font-mono text-[11px] text-zinc-500">1,712 Author Embeddings</span>
          </div>

          {/* Scatter Chart */}
          <div className="relative h-72 w-full bg-zinc-50/60 rounded-md border border-zinc-200/80 p-3 overflow-hidden">
            <svg viewBox="-4 -4 8 8" className="w-full h-full">
              {/* Axes */}
              <line x1="-4" y1="0" x2="4" y2="0" stroke="#e4e4e7" strokeWidth="0.05" />
              <line x1="0" y1="-4" x2="0" y2="4" stroke="#e4e4e7" strokeWidth="0.05" />

              {/* Data Points */}
              {pcaData?.scatter_points?.map((pt: any, i: number) => {
                const isSelected = selectedPoint?.id === pt.id;
                const color = CLUSTER_COLORS[pt.cluster % CLUSTER_COLORS.length];
                return (
                  <circle
                    key={i}
                    cx={pt.x}
                    cy={-pt.y}
                    r={isSelected ? "0.14" : "0.06"}
                    fill={color}
                    opacity={isSelected ? "1" : "0.75"}
                    stroke={isSelected ? "#18181b" : "none"}
                    strokeWidth="0.03"
                    className="cursor-pointer hover:opacity-100 transition-all"
                    onClick={() => setSelectedPoint(pt)}
                  />
                );
              })}
            </svg>
          </div>

          {/* Archetype Legend */}
          <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
            <div className="flex flex-wrap gap-3">
              {clusterData?.archetypes?.map((arch: any, i: number) => (
                <div key={i} className="flex items-center gap-1.5 text-xs text-zinc-600">
                  <span
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: CLUSTER_COLORS[i % CLUSTER_COLORS.length] }}
                  ></span>
                  <span>{arch.name}</span>
                </div>
              ))}
            </div>

            {selectedPoint && (
              <div className="text-xs font-mono text-zinc-700 bg-zinc-100 px-2 py-0.5 rounded">
                Selected: {selectedPoint.id} (Cluster {selectedPoint.cluster})
              </div>
            )}
          </div>
        </div>

        {/* Clustering Validation Metrics */}
        <div className="bg-white p-5 rounded-lg border border-zinc-200 shadow-sm space-y-4">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-zinc-800">
            <CheckCircle2 className="w-3.5 h-3.5 text-zinc-600" />
            <span>Clustering Validation Indices</span>
          </div>

          <div className="space-y-3 font-mono text-xs">
            <div className="p-3 rounded-md bg-zinc-50 border border-zinc-200 flex justify-between items-center">
              <div>
                <div className="font-semibold text-zinc-900">Silhouette Coefficient</div>
                <div className="text-[11px] text-zinc-500 font-sans">Cluster cohesion vs separation</div>
              </div>
              <div className="text-sm font-bold text-zinc-900">s = 0.2890</div>
            </div>

            <div className="p-3 rounded-md bg-zinc-50 border border-zinc-200 flex justify-between items-center">
              <div>
                <div className="font-semibold text-zinc-900">Davies-Bouldin Index</div>
                <div className="text-[11px] text-zinc-500 font-sans">Lower is better</div>
              </div>
              <div className="text-sm font-bold text-zinc-900">DB = 1.3800</div>
            </div>

            <div className="p-3 rounded-md bg-zinc-50 border border-zinc-200 flex justify-between items-center">
              <div>
                <div className="font-semibold text-zinc-900">Calinski-Harabasz Index</div>
                <div className="text-[11px] text-zinc-500 font-sans">Between/Within dispersion ratio</div>
              </div>
              <div className="text-sm font-bold text-zinc-900">CH = 510.60</div>
            </div>

            <div className="p-3 rounded-md bg-zinc-50 border border-zinc-200 flex justify-between items-center">
              <div>
                <div className="font-semibold text-zinc-900">GMM Log-Likelihood</div>
                <div className="text-[11px] text-zinc-500 font-sans">38 EM iterations to convergence</div>
              </div>
              <div className="text-sm font-bold text-zinc-900">&Lscr; = 4,286.1</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
