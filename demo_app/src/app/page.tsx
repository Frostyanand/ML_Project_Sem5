"use client";

import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import SynthesisTab from '../components/SynthesisTab';
import Unit3Tab from '../components/Unit3Tab';
import Unit4Tab from '../components/Unit4Tab';
import Unit5Tab from '../components/Unit5Tab';

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
    <div className="min-h-screen flex flex-col bg-zinc-50 text-zinc-900">
      {/* Navigation Header */}
      <Header activeTab={activeTab} setActiveTab={setActiveTab} status={status} />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {isInitializing ? (
          <div className="flex flex-col items-center justify-center min-h-[420px] space-y-3">
            <div className="w-6 h-6 border-2 border-zinc-900 border-t-transparent rounded-full animate-spin"></div>
            <p className="font-mono text-xs text-zinc-500">
              Connecting to model inference service...
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
          </div>
        )}
      </main>

      {/* Clean Technical Footer */}
      <footer className="border-t border-zinc-200 bg-white py-5">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-zinc-500 font-mono">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-zinc-700">Handwriting Synthesis Platform</span>
            <span>·</span>
            <span>Department of Computer Science &amp; Engineering</span>
          </div>
          <div className="flex items-center gap-3 text-zinc-400">
            <span>IAM Dataset Benchmark</span>
            <span>·</span>
            <span>Vision Transformer &amp; SDT</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
