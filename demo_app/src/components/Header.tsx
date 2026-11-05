"use client";

import React from 'react';
import { Layers, Activity, GitCommit, Sparkles, BookOpen, CheckCircle2 } from 'lucide-react';

interface HeaderProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  status: any;
}

export default function Header({ activeTab, setActiveTab, status }: HeaderProps) {
  const tabs = [
    { id: 'synthesis', label: 'Handwriting Studio', icon: Sparkles, badge: 'Phase 1 & 2' },
    { id: 'unit3', label: 'Unit 3: Mixture & EM', icon: Layers, badge: 'PCA · GMM' },
    { id: 'unit4', label: 'Unit 4: HMM & Markov', icon: Activity, badge: 'Viterbi · Kalman' },
    { id: 'unit5', label: 'Unit 5: Combining Models', icon: GitCommit, badge: 'Ensembles' },
    { id: 'defense', label: 'Academic Defense', icon: BookOpen, badge: 'IEEE Theory' },
  ];

  return (
    <header className="border-b border-slate-200 bg-white/90 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Meta */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center text-white font-bold shadow-sm">
              <span className="font-mono text-sm tracking-tight">HRP</span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-semibold text-slate-900 tracking-tight">
                  Hyper-Realistic Handwriting Synthesis
                </h1>
                <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-blue-50 text-blue-700 border border-blue-200/60 font-mono">
                  v2.4 Final
                </span>
              </div>
              <p className="text-xs text-slate-500 font-mono hidden sm:block">
                ViT Metric Learning · SDT Transformer · ML Curriculum Units 3-5
              </p>
            </div>
          </div>

          {/* Model Status Indicator */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-xs font-mono text-emerald-700">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span>SDT Core: Epoch 99</span>
            </div>
            <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-100 border border-slate-200 text-xs font-mono text-slate-700">
              <CheckCircle2 className="w-3.5 h-3.5 text-blue-600" />
              <span>1,712 Style Manifold</span>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-1 overflow-x-auto no-scrollbar py-2 border-t border-slate-100">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all whitespace-nowrap ${
                  isActive
                    ? 'bg-slate-900 text-white shadow-sm'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-white' : 'text-slate-500'}`} />
                <span>{tab.label}</span>
                <span
                  className={`text-[10px] px-1.5 py-0.2 rounded font-mono ${
                    isActive ? 'bg-slate-800 text-blue-300' : 'bg-slate-100 text-slate-500'
                  }`}
                >
                  {tab.badge}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
}

