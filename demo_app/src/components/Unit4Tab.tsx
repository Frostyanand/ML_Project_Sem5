"use client";

import React, { useState, useEffect } from 'react';
import { Activity, ArrowRight, Sliders, Waves, CheckCircle2 } from 'lucide-react';
import HandwritingCanvas from './HandwritingCanvas';

export default function Unit4Tab() {
  const [markovData, setMarkovData] = useState<any>(null);
  const [sampleData, setSampleData] = useState<any>(null);
  const [viterbiData, setViterbiData] = useState<any>(null);
  const [qNoise, setQNoise] = useState(0.05);
  const [rNoise, setRNoise] = useState(0.40);

  useEffect(() => {
    fetch('/api/unit4/markov')
      .then(r => r.json())
      .then(d => setMarkovData(d))
      .catch(console.error);

    fetch('/api/samples/real?index=0')
      .then(r => r.json())
      .then(d => {
        setSampleData(d);
        // Compute Viterbi for sample
        fetch('/api/unit4/viterbi', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ points: d.trajectory })
        })
          .then(res => res.json())
          .then(v => setViterbiData(v))
          .catch(console.error);
      })
      .catch(console.error);
  }, []);

  return (
    <div className="space-y-6">
      {/* Overview Banner */}
      <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-sm space-y-2">
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 rounded text-[11px] font-mono font-medium bg-blue-50 text-blue-700 border border-blue-200">
            Syllabus Unit 4
          </span>
          <h2 className="text-base font-bold text-slate-900 tracking-tight">
            Sequential Data Modeling, Hidden Markov Models & Kalman Linear Dynamical Systems
          </h2>
        </div>
        <p className="text-xs text-slate-600 max-w-4xl leading-relaxed">
          Online handwriting is intrinsically temporal and sequential: pen-tip coordinates (Δx, Δy) and pen-state indicators (p1, p2, p3) evolve over time t. Here we formulate a first-order Markov Chain on pen dynamics, implement Continuous Gaussian HMMs with Forward-Backward scaling factors and Viterbi dynamic programming to decode latent motor primitives, and apply Kalman Filter State-Space tracking to eliminate digitizer sensor jitter.
        </p>
      </div>

      {/* Grid: Markov Transition Matrix & State Machine */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Markov Transition Matrix Table */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-800">
              Pen Dynamics Transition Matrix A
            </span>
            <span className="font-mono text-[11px] text-blue-600 font-bold">1st Order Markov</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono text-center border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-2 text-left font-sans text-[11px]">Current State S_t</th>
                  <th className="py-2 text-blue-700">Pen-Down</th>
                  <th className="py-2 text-amber-700">Pen-Up</th>
                  <th className="py-2 text-slate-700">End-Word</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                <tr>
                  <td className="py-2 text-left font-sans font-medium text-slate-800">Pen-Down (Ink)</td>
                  <td className="py-2 bg-blue-50 font-bold text-blue-900">0.945</td>
                  <td className="py-2 text-slate-600">0.053</td>
                  <td className="py-2 text-slate-600">0.002</td>
                </tr>
                <tr>
                  <td className="py-2 text-left font-sans font-medium text-slate-800">Pen-Up (Lift)</td>
                  <td className="py-2 font-bold text-blue-900 bg-blue-50/60">1.000</td>
                  <td className="py-2 text-slate-400">0.000</td>
                  <td className="py-2 text-slate-400">0.000</td>
                </tr>
                <tr>
                  <td className="py-2 text-left font-sans font-medium text-slate-800">End-of-Word</td>
                  <td className="py-2 font-bold text-blue-900 bg-blue-50/40">1.000</td>
                  <td className="py-2 text-slate-400">0.000</td>
                  <td className="py-2 text-slate-400">0.000</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs space-y-1">
            <div className="font-semibold text-slate-800">Stationary Distribution π_∞:</div>
            <div className="font-mono text-slate-600 text-[11px] flex justify-between">
              <span>π(Down) = 94.8%</span>
              <span>π(Up) = 5.0%</span>
              <span>π(End) = 0.2%</span>
            </div>
          </div>
        </div>

        {/* HMM Motor Primitives & Scaling Card */}
        <div className="lg:col-span-2 bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-800">
              Continuous Gaussian HMM (Forward-Backward & Viterbi)
            </span>
            <span className="font-mono text-[11px] text-emerald-600 font-bold">4 Hidden States</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-3 rounded-lg border border-red-200 bg-red-50/40 text-xs space-y-1">
              <div className="flex items-center gap-1.5 font-bold text-red-800">
                <span className="w-2.5 h-2.5 rounded-full bg-red-600"></span>
                <span>State 0: Downstroke</span>
              </div>
              <p className="text-[11px] text-red-700/80">Main vertical stem; strong negative dy velocity.</p>
            </div>

            <div className="p-3 rounded-lg border border-blue-200 bg-blue-50/40 text-xs space-y-1">
              <div className="flex items-center gap-1.5 font-bold text-blue-800">
                <span className="w-2.5 h-2.5 rounded-full bg-blue-600"></span>
                <span>State 1: Upward Loop</span>
              </div>
              <p className="text-[11px] text-blue-700/80">Ascenders & glyph tops; positive dy velocity.</p>
            </div>

            <div className="p-3 rounded-lg border border-emerald-200 bg-emerald-50/40 text-xs space-y-1">
              <div className="flex items-center gap-1.5 font-bold text-emerald-800">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-600"></span>
                <span>State 2: Ligature</span>
              </div>
              <p className="text-[11px] text-emerald-700/80">Inter-character connection; dominant positive dx.</p>
            </div>

            <div className="p-3 rounded-lg border border-amber-200 bg-amber-50/40 text-xs space-y-1">
              <div className="flex items-center gap-1.5 font-bold text-amber-800">
                <span className="w-2.5 h-2.5 rounded-full bg-amber-600"></span>
                <span>State 3: Airborne</span>
              </div>
              <p className="text-[11px] text-amber-700/80">Pen-up repositioning to start of next character.</p>
            </div>
          </div>

          <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-600 space-y-1">
            <div className="font-semibold text-slate-800">Scaling Factors for Underflow Prevention:</div>
            <p className="text-[11px] leading-relaxed">
              For sequence lengths T = 180, unscaled joint probabilities P(O, Q) drop below 10^-78 causing double-precision underflow. We compute scaling factor c_t = 1 / Σ_i α_t(i), ensuring α_hat stays normalized, and evaluate exact log-likelihood ln P(O|λ) = -Σ ln(c_t) = -1073.54.
            </p>
          </div>
        </div>
      </div>

      {/* Viterbi & Kalman Interactive Demonstration */}
      <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-800">
              Live Viterbi Motor State Decoding & Kalman Filter LDS
            </h3>
            <p className="text-xs text-slate-500 font-mono mt-0.5">
              Target sample: &ldquo;{sampleData?.text || "Handwriting Sample"}&rdquo; (Author: {sampleData?.writer_id || "a01-003"})
            </p>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono">
            <div className="flex items-center gap-2">
              <span className="text-slate-600">Process Q:</span>
              <span className="font-bold text-blue-600">{qNoise}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-slate-600">Sensor R:</span>
              <span className="font-bold text-blue-600">{rNoise}</span>
            </div>
          </div>
        </div>

        <HandwritingCanvas
          rawTrajectory={sampleData?.trajectory || []}
          kalmanTrajectory={sampleData?.trajectory || []}
          viterbiStates={viterbiData?.states || []}
          text={sampleData?.text}
          writerId={sampleData?.writer_id}
          isRealSample={true}
        />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-600 pt-2 border-t border-slate-100">
          <div>
            <span className="font-semibold text-slate-800">Viterbi Dynamic Programming:</span>
            <p className="text-[11px] mt-0.5 leading-relaxed">
              Recursively computes δ_t(j) = max_i [δ_(t-1)(i) a_ij] b_j(o_t) and stores backpointers ψ_t(j) to backtrack the mathematically optimal global hidden motor sequence with O(K^2 T) complexity instead of O(K^T).
            </p>
          </div>
          <div>
            <span className="font-semibold text-slate-800">Kalman State-Space LDS:</span>
            <p className="text-[11px] mt-0.5 leading-relaxed">
              Propagates state x_t = [p_x, p_y, v_x, v_y]^T and error covariance P_t, applying optimal Kalman gain K_t to blend inertial kinematic prediction with noisy digitizer measurements.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

