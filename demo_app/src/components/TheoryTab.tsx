"use client";

import React, { useState } from 'react';
import { BookOpen, HelpCircle, CheckCircle2, ChevronDown, ChevronUp } from 'lucide-react';

export default function TheoryTab() {
  const [openFaq, setOpenFaq] = useState<number | null>(0);

  const faqs = [
    {
      q: "Q1: What is the scientific purpose of PCA and Scree Plot analysis in Unit 3?",
      a: "Our Vision Transformer outputs a 512-dimensional embedding on the unit sphere S^511. However, high-dimensional spaces suffer from the curse of dimensionality and collinearity. By performing SVD on the centered covariance matrix Σ = (1/N) X^T X, our Scree Plot reveals that 90% of all handwriting variance across 1,712 writers is captured within 13 latent principal axes, and 95% within 26 axes. This validates that human handwriting variation occupies a compact low-dimensional manifold governed by physical hand kinematics."
    },
    {
      q: "Q2: How does the Expectation-Maximization (EM) algorithm work in Gaussian Mixture Models (GMM)?",
      a: "GMM assumes data is generated from a mixture of K Gaussian components: p(x) = Σ_k π_k N(x | μ_k, Σ_k). In the E-step, the model computes the posterior responsibility γ_nk = (π_k N(x_n | μ_k, Σ_k)) / (Σ_j π_j N(x_n | μ_j, Σ_j)). In the M-step, it re-estimates component weights π_k, means μ_k, and covariances Σ_k using these responsibilities. Furthermore, Phase 2's generative MDN is mathematically a continuous conditional GMM where the parameters are parameterized dynamically by the Transformer decoder!"
    },
    {
      q: "Q3: Why are scaling factors c_t essential in the HMM Forward-Backward Algorithm?",
      a: "For long sequential stroke trajectories (T = 180 to 500 coordinates), joint observation probabilities P(O_1..t, q_t=j) decrease exponentially as a product of T small values, dropping below 10^-80 and triggering double-precision floating point underflow. By computing normalization factor c_t = 1 / Σ_i α_t(i) at each temporal step, the scaled forward variable α_hat remains in [0, 1], and the exact log-likelihood is recovered via ln P(O|λ) = -Σ_t ln(c_t)."
    },
    {
      q: "Q4: How does the Viterbi Algorithm differ from standard posterior decoding?",
      a: "Posterior decoding maximizes the marginal probability at each step P(q_t=j | O), which can yield invalid or illegal state transitions where A_ij = 0. In contrast, the Viterbi algorithm employs dynamic programming to find the globally optimal joint path argmax_Q P(Q | O, λ) over all possible paths in O(K^2 T) time, guaranteeing that every decoded motor sub-gesture transition is kinematically valid."
    },
    {
      q: "Q5: How does the Kalman Filter Linear Dynamical System (LDS) improve handwriting synthesis?",
      a: "Digital digitizers introduce high-frequency quantization and sensor jitter into coordinate streams. We model pen motion with state vector x_t = [p_x, p_y, v_x, v_y]^T and continuous kinematic dynamics F_mat. The Kalman filter predicts next position via inertia, then incorporates the measurement with optimal Kalman gain K_t = P H^T (H P H^T + R)^-1, yielding smooth, human-like pen strokes without artificial robotic jaggedness."
    },
    {
      q: "Q6: Why did the Stacking Meta-Classifier achieve 99.07% accuracy compared to standalone CART (87.15%)?",
      a: "CART uses axis-aligned orthogonal splits that struggle with non-linear, correlated manifolds. Random Forest reduces variance via bootstrap aggregation, AdaBoost reduces bias by focusing on hard cases, and RBF Kernel SVM finds maximum margin boundaries in RKHS. Stacking trains a non-negative Logistic Regression meta-learner on out-of-fold cross-validation predictions, combining their complementary strengths to achieve state-of-the-art 99.07% accuracy."
    }
  ];

  return (
    <div className="space-y-6">
      {/* Overview Banner */}
      <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-sm space-y-2">
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 rounded text-[11px] font-mono font-medium bg-blue-50 text-blue-700 border border-blue-200">
            Defense Master Guide
          </span>
          <h2 className="text-base font-bold text-slate-900 tracking-tight">
            Academic Curriculum Defense, IEEE Formalization & Viva Guide
          </h2>
        </div>
        <p className="text-xs text-slate-600 max-w-4xl leading-relaxed">
          Comprehensive scientific cross-referencing showing how every single syllabus requirement from Units 1 through 5 integrates seamlessly into the 5-Phase Hyper-Realistic Handwriting Synthesis architecture.
        </p>
      </div>

      {/* Syllabus Integration Matrix */}
      <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-800">
            <BookOpen className="w-4 h-4 text-blue-600" />
            <span>Complete Curriculum Algorithm Integration Matrix</span>
          </div>
          <span className="font-mono text-xs text-slate-500">Units 1 through 5</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500 font-mono text-left bg-slate-50/60">
                <th className="py-2.5 px-3">Unit</th>
                <th className="py-2.5 px-3">Curriculum Algorithm</th>
                <th className="py-2.5 px-3">Mathematical Formulation</th>
                <th className="py-2.5 px-3">Engineering Role in Project</th>
                <th className="py-2.5 px-3">Deliverable / Metric</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono text-[11px]">
              <tr>
                <td className="py-2.5 px-3 font-bold text-blue-600">Unit 1 & 2</td>
                <td className="py-2.5 px-3 font-sans text-slate-900">Ridge & Robust Regression</td>
                <td className="py-2.5 px-3 text-slate-600">w* = (X^T X + λI)^-1 X^T y</td>
                <td className="py-2.5 px-3 font-sans text-slate-700">Predicts physical stroke density under scanner outlier noise</td>
                <td className="py-2.5 px-3 text-emerald-600 font-bold">R² = 0.038 (Clean)</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-bold text-blue-600">Unit 2</td>
                <td className="py-2.5 px-3 font-sans text-slate-900">Laplace Bayes LogReg</td>
                <td className="py-2.5 px-3 text-slate-600">S_N = (αI + Σ σ(1-σ) x x^T)^-1</td>
                <td className="py-2.5 px-3 font-sans text-slate-700">Calibrates posterior author verification confidence</td>
                <td className="py-2.5 px-3 text-emerald-600 font-bold">Acc: 88.67% | AUC: 1.0</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-bold text-blue-600">Unit 2</td>
                <td className="py-2.5 px-3 font-sans text-slate-900">RBF Kernel SVM</td>
                <td className="py-2.5 px-3 text-slate-600">K(x, x&apos;) = exp(-γ ||x - x&apos;||^2)</td>
                <td className="py-2.5 px-3 font-sans text-slate-700">Isolates 91 support vector edge cases on style hypersphere</td>
                <td className="py-2.5 px-3 text-emerald-600 font-bold">Acc: 98.67%</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-bold text-purple-600">Unit 3</td>
                <td className="py-2.5 px-3 font-sans text-slate-900">PCA Scree & Latent Dims</td>
                <td className="py-2.5 px-3 text-slate-600">Σ = V Λ V^T, cum_var &ge; 0.95</td>
                <td className="py-2.5 px-3 font-sans text-slate-700">Reduces 512-D ViT space to 26 intrinsic style dimensions</td>
                <td className="py-2.5 px-3 text-emerald-600 font-bold">26 PCs = 95% Var</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-bold text-purple-600">Unit 3</td>
                <td className="py-2.5 px-3 font-sans text-slate-900">GMM with EM Algorithm</td>
                <td className="py-2.5 px-3 text-slate-600">E-step γ_nk · M-step (μ_k, Σ_k, π_k)</td>
                <td className="py-2.5 px-3 font-sans text-slate-700">Clusters 1,712 writers into 5 soft handwriting archetypes</td>
                <td className="py-2.5 px-3 text-emerald-600 font-bold">Sil: 0.289 | DB: 1.38</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-bold text-purple-600">Unit 3</td>
                <td className="py-2.5 px-3 font-sans text-slate-900">Ward Hierarchical Cluster</td>
                <td className="py-2.5 px-3 text-slate-600">ΔESS = (n_A n_B / (n_A+n_B)) ||μ_A - μ_B||^2</td>
                <td className="py-2.5 px-3 font-sans text-slate-700">Builds tree dendrogram of author handwriting genealogies</td>
                <td className="py-2.5 px-3 text-emerald-600 font-bold">Ward Dendrogram</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-bold text-amber-600">Unit 4</td>
                <td className="py-2.5 px-3 font-sans text-slate-900">Markov Pen Dynamics</td>
                <td className="py-2.5 px-3 text-slate-600">P(S_t+1 = j | S_t = i)</td>
                <td className="py-2.5 px-3 font-sans text-slate-700">Models 3-state transition matrix & steady-state distribution</td>
                <td className="py-2.5 px-3 text-emerald-600 font-bold">π_down = 94.8%</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-bold text-amber-600">Unit 4</td>
                <td className="py-2.5 px-3 font-sans text-slate-900">HMM Forward-Backward</td>
                <td className="py-2.5 px-3 text-slate-600">c_t = 1/Σ α_t(i), ln P(O) = -Σ ln(c_t)</td>
                <td className="py-2.5 px-3 font-sans text-slate-700">Underflow-resilient continuous likelihood evaluation</td>
                <td className="py-2.5 px-3 text-emerald-600 font-bold">ln P = -1073.54</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-bold text-amber-600">Unit 4</td>
                <td className="py-2.5 px-3 font-sans text-slate-900">Viterbi Dynamic Program</td>
                <td className="py-2.5 px-3 text-slate-600">δ_t(j) = max_i [δ_t-1(i) a_ij] b_j(o_t)</td>
                <td className="py-2.5 px-3 font-sans text-slate-700">Decodes 4 latent motor sub-gestures (Down, Loop, Ligature, Air)</td>
                <td className="py-2.5 px-3 text-emerald-600 font-bold">O(K^2 T) Optimal Path</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-bold text-amber-600">Unit 4</td>
                <td className="py-2.5 px-3 font-sans text-slate-900">Kalman Filter LDS</td>
                <td className="py-2.5 px-3 text-slate-600">K_t = P H^T (H P H^T + R)^-1</td>
                <td className="py-2.5 px-3 font-sans text-slate-700">Eliminates digitizer sensor jitter via continuous state space</td>
                <td className="py-2.5 px-3 text-emerald-600 font-bold">Optimal Tracking</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-bold text-emerald-600">Unit 5</td>
                <td className="py-2.5 px-3 font-sans text-slate-900">CART Decision Tree</td>
                <td className="py-2.5 px-3 text-slate-600">I_G(t) = 1 - Σ p(k|t)^2</td>
                <td className="py-2.5 px-3 font-sans text-slate-700">Single recursive orthogonal binary partitioning baseline</td>
                <td className="py-2.5 px-3 text-emerald-600 font-bold">Acc: 87.15%</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-bold text-emerald-600">Unit 5</td>
                <td className="py-2.5 px-3 font-sans text-slate-900">Random Forest Ensemble</td>
                <td className="py-2.5 px-3 text-slate-600">Var = ρ σ^2 + ((1-ρ)/B) σ^2</td>
                <td className="py-2.5 px-3 font-sans text-slate-700">100 trees with bootstrap aggregation & random feature subsets</td>
                <td className="py-2.5 px-3 text-emerald-600 font-bold">Acc: 97.20% | OOB: 94.1%</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-bold text-emerald-600">Unit 5</td>
                <td className="py-2.5 px-3 font-sans text-slate-900">AdaBoost & GBDT</td>
                <td className="py-2.5 px-3 text-slate-600">w_i^(m+1) = w_i exp(-α_m y_i h_m)</td>
                <td className="py-2.5 px-3 font-sans text-slate-700">Sequential boosting prioritizing difficult borderline handwriting</td>
                <td className="py-2.5 px-3 text-emerald-600 font-bold">Acc: 90.65% & 95.79%</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-bold text-emerald-600">Unit 5</td>
                <td className="py-2.5 px-3 font-sans text-slate-900">Stacking Meta-Ensemble</td>
                <td className="py-2.5 px-3 text-slate-600">P(y|x) = LogisticReg(RF, GBDT, SVM)</td>
                <td className="py-2.5 px-3 font-sans text-slate-700">Optimal meta-learner convex blending of diverse base classifiers</td>
                <td className="py-2.5 px-3 text-emerald-600 font-bold">Acc: 99.07% (SOTA)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Professor Viva Q&A Accordion */}
      <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-800">
          <HelpCircle className="w-4 h-4 text-blue-600" />
          <span>Professor Defense & Viva Q&A Cheat Sheet</span>
        </div>

        <div className="space-y-3">
          {faqs.map((faq, idx) => {
            const isOpen = openFaq === idx;
            return (
              <div key={idx} className="border border-slate-200 rounded-lg overflow-hidden transition-all">
                <button
                  onClick={() => setOpenFaq(isOpen ? null : idx)}
                  className="w-full p-3.5 text-left bg-slate-50/70 hover:bg-slate-100 flex items-center justify-between gap-3 text-xs font-bold text-slate-800 transition-colors"
                >
                  <span>{faq.q}</span>
                  {isOpen ? <ChevronUp className="w-4 h-4 text-slate-500 flex-shrink-0" /> : <ChevronDown className="w-4 h-4 text-slate-500 flex-shrink-0" />}
                </button>
                {isOpen && (
                  <div className="p-3.5 text-xs text-slate-600 bg-white border-t border-slate-200 leading-relaxed font-sans">
                    {faq.a}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

