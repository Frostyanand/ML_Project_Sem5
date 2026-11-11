"use client";

import React from 'react';
import { PenTool, Layers, Activity, BarChart2, CheckCircle2 } from 'lucide-react';

interface HeaderProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  status: any;
}

export default function Header({ activeTab, setActiveTab, status }: HeaderProps) {
  const tabs = [
    { id: 'synthesis', label: 'Synthesis Studio', icon: PenTool },
    { id: 'unit3', label: 'Latent Manifold & Clustering', icon: Layers },
    { id: 'unit4', label: 'Sequential Modeling (HMM & Kalman)', icon: Activity },
    { id: 'unit5', label: 'Model Ensembles & Benchmarks', icon: BarChart2 },
  ];

  const totalWriters = status?.total_writers || 1712;

  return (
    <header className="border-b border-zinc-200 bg-white/95 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Meta */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-md bg-zinc-900 flex items-center justify-center text-white font-medium shadow-sm">
              <PenTool className="w-4 h-4 text-zinc-100" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-semibold text-zinc-900 tracking-tight">
                  Handwriting Synthesis Platform
                </h1>
                <span className="px-1.5 py-0.5 rounded text-[11px] font-mono text-zinc-600 bg-zinc-100 border border-zinc-200">
                  Research Prototype
                </span>
              </div>
              <p className="text-xs text-zinc-500 hidden sm:block">
                Metric Learning Style Manifold · Kinematic Trajectory Synthesis
              </p>
            </div>
          </div>

          {/* Clean Status Indicator */}
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-zinc-50 border border-zinc-200 text-xs font-mono text-zinc-700">
              <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
              <span>Backend Ready</span>
            </div>
            <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-zinc-50 border border-zinc-200 text-xs font-mono text-zinc-600">
              <CheckCircle2 className="w-3.5 h-3.5 text-zinc-400" />
              <span>{totalWriters} Authors</span>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-1 overflow-x-auto no-scrollbar py-2 border-t border-zinc-100">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
                  isActive
                    ? 'bg-zinc-900 text-white shadow-sm'
                    : 'text-zinc-600 hover:text-zinc-900 hover:bg-zinc-100'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-zinc-100' : 'text-zinc-500'}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
}
