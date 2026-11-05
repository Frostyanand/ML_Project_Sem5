# Hyper-Realistic Handwriting Synthesis & ML Curriculum Integration

[![IEEE TPAMI Format](https://img.shields.io/badge/Paper-IEEE%20TPAMI%20Format-blue.svg)](reports/IEEE_Conference_Handwriting_Synthesis_and_ML_Curriculum.pdf)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14.2-black.svg)](https://nextjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)

An end-to-end framework unifying Deep Geometric Metric Learning (Vision Transformer) and Kinematic Trajectory Generation (Style-Disentangled Transformer) with classical probabilistic, sequential, and meta-ensemble machine learning algorithms (Units 1 through 5).

---

## 📖 Table of Contents
1. [Overview & Core Architecture](#-overview--core-architecture)
2. [Syllabus Integration (Units 1 to 5)](#-syllabus-integration-units-1-to-5)
3. [One-Click Setup & Launch](#-one-click-setup--launch)
4. [Master Executed Notebook](#-master-executed-notebook)
5. [Academic Research Paper & Reports](#-academic-research-paper--reports)
6. [Interactive Web Interface](#-interactive-web-interface)
7. [Directory Structure](#-directory-structure)

---

## 🔬 Overview & Core Architecture

Automated handwriting synthesis often suffers from the **Handwriting Uncanny Valley**: font engines produce mechanical, identical glyphs (*"Too Perfect"*), while pixel-based diffusion/GANs create blurred strokes and floating ink fragments (*"Imperfect in the Wrong Way"*).

Our framework parameterizes handwriting as an online temporal kinematic stream:
$$\mathbf{o}_t = [\Delta x_t, \Delta y_t, p_{1,t}, p_{2,t}, p_{3,t}]^T \in \mathbb{R}^5$$

The architecture spans five distinct tiers:
- **Phase 0 (Data Curation):** Multimodal curation of 215,224 line crops from the IAM Offline and Online databases.
- **Phase 1 (The Eye / Style Engine):** Vision Transformer (`vit_base_patch16_224`, 86M params) trained via Batch-Hard Triplet Loss with P-K sampling ($P=16, K=4$) on Google Cloud TPU v5e hardware. Maps handwriting into an invariant 512-dimensional style manifold on $\mathbb{S}^{511}$ (79.96% Top-1, 91.38% Top-5 retrieval across 1,712 authors).
- **Phase 2 (The Hand / Stroke Generator):** Style-Disentangled Transformer (SDT) with Rotary Position Embeddings (RoPE) and 20-component bivariate Gaussian Mixture Density Network (MDN). Resolves the Negative Loss Paradox through differential entropy analysis ($h(X) = \frac{1}{2}\ln(2\pi e \sigma^2)$).
- **Phase 3 (The Paper / Realism Engine):** Procedural Perlin noise baseline wander and ink degradation.
- **Phase 4 (Deployment & Verification):** High-performance FastAPI serving backend and interactive Next.js 14 vector canvas.

---

## 📊 Syllabus Integration (Units 1 to 5)

| Curriculum Unit | Algorithms Implemented | Dataset / Input | Key Metric / Mathematical Result |
| :--- | :--- | :--- | :--- |
| **Unit 1 & 2** | Ridge, Huber, RANSAC | Stroke coordinates + 5% noise | Huber ($\delta=1.35$) & RANSAC achieve 98.7% parameter recovery |
| **Unit 1 & 2** | Bayesian Linear Regression | Conjugate Gaussian Priors | Closed-form epistemic uncertainty $\sigma_{\text{epistemic}} = 0.0051$ |
| **Unit 1 & 2** | Laplace Bayesian Logistic Reg. | Style Manifold Projection | Posterior parameter distribution $\mathbf{w} \sim \mathcal{N}(\hat{\mathbf{w}}, \mathbf{H}^{-1})$ |
| **Unit 1 & 2** | RBF Kernel SVM | 512-D Style Manifold | 91 support vector boundary styles, 98.67% cross-val accuracy |
| **Unit 3** | Principal Component Analysis | 1,712 writers in $\mathbb{R}^{512}$ | **95% variance concentrated in 26 latent dimensions** ($d \ll 512$) |
| **Unit 3** | Factor Analysis | 10 Latent Factors | Negligible sensor noise uniqueness ($\Psi \approx 2 \times 10^{-6}$) |
| **Unit 3** | K-Means Clustering | $k \in [2, 10]$ partitions | Optimal $k^*=5$: Silhouette $s=0.2890$, DB $1.3800$, CH $510.60$ |
| **Unit 3** | Gaussian Mixture Models (EM) | 5 Gaussian Components | Converged in 38 iterations to log-likelihood 4,286.1 |
| **Unit 3** | Ward Hierarchical Clustering | Euclidean Style Distance | Genealogical author tree grouping shared slants and connectivities |
| **Unit 4** | 1st-Order Markov Pen Chain | Pen lift / down events | 94.55% ink continuity; stationary dist $[94.8\%, 5.0\%, 0.2\%]^T$ |
| **Unit 4** | Continuous Gaussian HMM | 4 Latent Motor States | Underflow-resilient scaling factors $c_t = 1/\sum_i \alpha_t(i)$ ($T > 500$) |
| **Unit 4** | Viterbi Dynamic Programming | Trajectory decoding | Optimal state sequence decoding in $O(K^2 T)$ time complexity |
| **Unit 4** | Kalman Filter (LDS) | Digitizer coordinate streams | Forward predictor-corrector equations drop trajectory MSE by **68.4%** |
| **Unit 5** | CART Decision Trees | Gini Impurity Splitting | Baseline non-parametric accuracy 80.33% |
| **Unit 5** | Random Forest ($B=100$) | Decorrelated Bagging | 96.67% test accuracy, 94.08% Out-of-Bag (OOB) score |
| **Unit 5** | AdaBoost (SAMME.R) | Exponential Loss Minimization | 94.33% test accuracy |
| **Unit 5** | Gradient Tree Boosting (GBDT) | Negative Deviance Boosting | 97.67% test accuracy |
| **Unit 5** | **Stacking Meta-Ensemble** | Base Classifiers + Meta-LogReg | **99.07% SOTA Test Accuracy**, **0.9998 ROC-AUC** |
| **Unit 5** | Gini Feature Importance | Mean Decrease in Impurity | Dim 142 (italic slant, 4.82%) & Dim 89 (density, 4.15%) |

---

## ⚡ One-Click Setup & Launch

### Prerequisites
- **Windows OS**
- **Python 3.10+** (with `python` on PATH)
- **Node.js 18+** (with `npm` on PATH)

### Step 1: Automated Dependency Installation
Double-click `setup.bat` (or run in CMD / PowerShell):
```cmd
setup.bat
```
*This verifies Python & Node.js, installs all PyTorch / Scikit-Learn dependencies via `requirements.txt`, installs Next.js npm packages, and compiles the production bundle.*

### Step 2: One-Click Demonstration Launch
Double-click `run_demo.bat`:
```cmd
run_demo.bat
```
*This automatically checks and frees ports `8000` and `3000`, starts the FastAPI PyTorch server, launches the Next.js frontend, opens your default browser at `http://localhost:3000`, and streams all backend & frontend logs with unified prefixes (`[API]` / `[WEB]`). When you close the terminal, both processes are cleanly terminated.*

---

## 📓 Master Executed Notebook

The complete end-to-end experimental journey is encapsulated in:
- [`Phase_1_2_Units3_4_5_Master_Integrated.ipynb`](file:///c:/Vault%201/Frosty%20Coder/Micro%20Projects/ML_Project_Sem5/Phase_1_2_Units3_4_5_Master_Integrated.ipynb) (3.58 MB)

Every cell has been executed sequentially with outputs, loss curves, confusion matrices, dendrograms, and trajectory plots embedded.

---

## 📄 Academic Research Paper & Reports

The research manuscript is prepared according to the **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)** / **ICDAR** publication specification:
- **PDF (6 Pages, 0 text overlap, embedded 300 DPI figures):** [`reports/IEEE_Conference_Handwriting_Synthesis_and_ML_Curriculum.pdf`](file:///c:/Vault%201/Frosty%20Coder/Micro%20Projects/ML_Project_Sem5/reports/IEEE_Conference_Handwriting_Synthesis_and_ML_Curriculum.pdf)
- **Word Document (.docx):** [`reports/IEEE_Conference_Handwriting_Synthesis_and_ML_Curriculum.docx`](file:///c:/Vault%201/Frosty%20Coder/Micro%20Projects/ML_Project_Sem5/reports/IEEE_Conference_Handwriting_Synthesis_and_ML_Curriculum.docx)
- **Markdown (.md):** [`reports/IEEE_Conference_Handwriting_Synthesis_and_ML_Curriculum.md`](file:///c:/Vault%201/Frosty%20Coder/Micro%20Projects/ML_Project_Sem5/reports/IEEE_Conference_Handwriting_Synthesis_and_ML_Curriculum.md)
- **Printable HTML Source:** [`reports/IEEE_Pristine_Manuscript.html`](file:///c:/Vault%201/Frosty%20Coder/Micro%20Projects/ML_Project_Sem5/reports/IEEE_Pristine_Manuscript.html)

---

## 🖥️ Interactive Web Interface

The Next.js 14 web app (`demo_app`) features a clean, light-themed editorial UI:
1. **Synthesis Studio:** Interactive lined notebook canvas with SVG Bezier trajectory rendering, temperature bias slider, and instant Kalman filter toggle.
2. **Unit 3 (Manifold & Clustering):** Interactive 2D PCA scatter manifold of 1,712 writers, GMM Gaussian ellipses, and K-Means archetype selector.
3. **Unit 4 (Sequential Dynamics & HMM):** 3-state Markov transition matrix, 4-state HMM Viterbi trellis decoder, and raw vs Kalman denoised trajectory comparison.
4. **Unit 5 (Ensemble Benchmark):** Model accuracy progression chart, ROC curves, Stacking Meta-Classifier inference, and Random Forest Gini feature importance ranking.
5. **Academic Defense:** Full theoretical derivations, differential entropy proofs, and model architecture diagrams.

---

## 📁 Directory Structure

```text
ML_Project_Sem5/
├── data/
│   └── writer_style_vectors.pt        # 1,712 author embeddings on S^511 (3.8 MB)
├── demo_app/
│   ├── backend/
│   │   └── server.py                  # FastAPI PyTorch inference backend
│   ├── src/
│   │   ├── app/page.tsx               # Next.js 14 interactive UI
│   │   └── components/                # Modular UI components
│   ├── package.json
│   └── run_demo.py                    # Multi-process orchestrator
├── reports/
│   ├── figures/                       # High-resolution publication plots
│   ├── IEEE_Conference_Handwriting_Synthesis_and_ML_Curriculum.pdf
│   ├── IEEE_Conference_Handwriting_Synthesis_and_ML_Curriculum.docx
│   ├── IEEE_Conference_Handwriting_Synthesis_and_ML_Curriculum.md
│   └── IEEE_Pristine_Manuscript.html
├── Phase_1_2_Units3_4_5_Master_Integrated.ipynb  # Master executed notebook (3.58 MB)
├── Phase_1_Unit2_Integrated.ipynb                # Baseline Phase 1 notebook
├── requirements.txt                   # Complete Python dependencies
├── setup.bat                          # One-click dependency setup
├── run_demo.bat                       # One-click platform launcher
└── .gitignore                         # Standard exclusions (node_modules, caches, large weights)
```
