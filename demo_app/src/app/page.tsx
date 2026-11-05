"use client";

import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import SynthesisTab from '../components/SynthesisTab';
import Unit3Tab from '../components/Unit3Tab';
import Unit4Tab from '../components/Unit4Tab';
import Unit5Tab from '../components/Unit5Tab';
import TheoryTab from '../components/TheoryTab';

export default function Home() {
  const [activeTab, setActiveTab] = useState<string>('synthesis');
  const [status, setStatus] = useState<any>(null);
  const [writers, setWriters] = useState<any[]>([]);
  const [archetypes, setArchetypes] = useState<string[]>([]);
  const [isInitializing, setIsInitializing] = useState<boolean>(true);

  useEffect(() => {
    Promise.all([
      fetch('/api/status').then(r => r.json()).catch(() => null),
      fetch('/api/writers').then(r => r.json()).catch(() => null)
    ]).then(([st, wr]) => {
      if (st) setStatus(st);
      if (wr) {
        setWriters(wr.writers || []);
        setArchetypes(wr.archetypes || []);
      }
      setIsInitializing(false);
    }).catch(err => {
      console.error("Initialization error:", err);
      setIsInitializing(false);
    });
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      {/* Top Header */}
      <Header activeTab={activeTab} setActiveTab={setActiveTab} status={status} />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {isInitializing ? (
          <div className="flex flex-col items-center justify-center min-h-[400px] space-y-3">
            <div className="w-8 h-8 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
            <p className="font-mono text-xs text-slate-500">
              Connecting to FastAPI Model Server (PyTorch SDT &amp; ViT Style Manifold)...
            </p>
          </div>
        ) : (
          <div>
            {activeTab === 'synthesis' && (
              <SynthesisTab writers={writers} archetypes={archetypes} />
            )}
            {activeTab === 'unit3' && <Unit3Tab />}
            {activeTab === 'unit4' && <Unit4Tab />}
            {activeTab === 'unit5' && <Unit5Tab />}
            {activeTab === 'defense' && <TheoryTab />}
          </div>
        )}
      </main>

      {/* Minimal Engineering Footer */}
      <footer className="border-t border-slate-200 bg-white py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500 font-mono">
          <div className="flex items-center gap-2">
            <span className="font-bold text-slate-700">Project Nano Banana Pro</span>
            <span>·</span>
            <span>Machine Learning &amp; Deep Generative Handwriting Systems</span>
          </div>
          <div className="flex items-center gap-4">
            <span>SDT MDN-20 Checkpoint</span>
            <span>·</span>
            <span>IAM Offline &amp; Online Dataset</span>
            <span>·</span>
            <span className="text-emerald-600 font-bold">IEEE Standard Compliant</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

