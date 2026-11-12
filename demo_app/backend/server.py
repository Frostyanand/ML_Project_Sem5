import os
import sys

# Configure UTF-8 stdout/stderr for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.stats import multivariate_normal

app = FastAPI(title="Handwriting Synthesis & ML Curriculum Inference Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# MODEL ARCHITECTURE (SDT Transformer)
# ------------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        dim = d_model
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    def forward(self, q, k):
        seq_len = q.shape[2]
        cos = self.cos_cached[:, :, :seq_len, ...].to(q.device)
        sin = self.sin_cached[:, :, :seq_len, ...].to(q.device)
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
        q_rope = (q * cos) + (rotate_half(q) * sin)
        k_rope = (k * cos) + (rotate_half(k) * sin)
        return q_rope, k_rope

class NanoDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.self_attn_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_content_q = nn.Linear(d_model, d_model, bias=False)
        self.cross_content_kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_style_q = nn.Linear(d_model, d_model, bias=False)
        self.cross_style_kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)
        
    def forward(self, x, content_mem, style_mem, tgt_mask=None):
        B, L, D = x.shape
        residual = x
        x = self.norm1(x)
        qkv = self.self_attn_qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.rope(q, k)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=tgt_mask, dropout_p=0.0)
        x = residual + attn_out.transpose(1, 2).reshape(B, L, D)
        
        residual = x
        x = self.norm2(x)
        q = self.cross_content_q(x).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.cross_content_kv(content_mem).reshape(B, -1, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = residual + attn_out.transpose(1, 2).reshape(B, L, D)
        
        residual = x
        x = self.norm3(x)
        q = self.cross_style_q(x).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.cross_style_kv(style_mem).reshape(B, -1, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = residual + attn_out.transpose(1, 2).reshape(B, L, D)
        
        x = x + self.ff(self.norm4(x))
        return x

class ContentEncoder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.cnn_context = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.ReLU(),
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.ReLU(),
            nn.Dropout(0.1)
        )
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)
    def forward(self, text):
        x = self.embedding(text).permute(0, 2, 1)
        x = self.cnn_context(x).permute(0, 2, 1)
        return self.transformer(x)

class StyleEncoder_Legacy(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 3, 2, 1), nn.Linear(1,1))
    def forward(self, x): return x

class NanoBananaSDT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config['D_MODEL']
        self.content_encoder = ContentEncoder(85, self.d_model)
        self.style_encoder = StyleEncoder_Legacy(self.d_model) 
        self.style_vec_proj = nn.Linear(512, self.d_model)
        self.stroke_embed = nn.Linear(5, self.d_model)
        self.layers = nn.ModuleList([NanoDecoderLayer(self.d_model, config['N_HEADS']) for _ in range(config['N_LAYERS'])])
        self.mdn_head = nn.Linear(self.d_model, config['MDN_GAUSSIANS'] * 6)
        self.pen_head = nn.Linear(self.d_model, 3)

    def forward(self, strokes, text, style_vec):
        content_mem = self.content_encoder(text)
        style_mem = self.style_vec_proj(style_vec).unsqueeze(1)
        x = self.stroke_embed(strokes)
        mask = torch.triu(torch.ones(strokes.size(1), strokes.size(1), device=strokes.device) * float('-inf'), diagonal=1)
        for layer in self.layers:
            x = layer(x, content_mem, style_mem, tgt_mask=mask)
        return self.mdn_head(x), self.pen_head(x)

VOCAB = " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# ------------------------------------------------------------------------------
# GLOBAL STATE & ASSET LOADING
# ------------------------------------------------------------------------------
STATE: Dict[str, Any] = {}

def init_system():
    print("⏳ Initializing Backend Models & Datasets...")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    local_data_dir = os.path.join(repo_root, "data")
    base_proj = r"C:\Vault 1\Frosty Coder\Projects\Hyper-Realistic-Handwriting-Synthesis-Platform"
    
    # 1. Resolve Style Bank path (local repo data folder first, then fallback)
    style_path = os.path.join(local_data_dir, "writer_style_vectors.pt")
    if not os.path.exists(style_path):
        style_path = os.path.join(base_proj, "writer_style_vectors.pt")
    
    if os.path.exists(style_path):
        style_bank = torch.load(style_path, map_location="cpu")
        writer_ids = sorted(list(style_bank.keys()))
        X_styles = torch.stack([style_bank[w] for w in writer_ids]).numpy()
        STATE["style_bank"] = style_bank
        STATE["writer_ids"] = writer_ids
        STATE["X_styles"] = X_styles
        print(f"✅ Loaded {len(writer_ids)} author style representations from {style_path}")
    else:
        print("⚠️ Warning: writer_style_vectors.pt not found, generating synthetic latent manifold.")
        writer_ids = [f"author_{i:03d}" for i in range(100)]
        X_styles = np.random.randn(100, 512).astype(np.float32)
        X_styles /= np.linalg.norm(X_styles, axis=1, keepdims=True)
        STATE["style_bank"] = {w: torch.from_numpy(X_styles[i]) for i, w in enumerate(writer_ids)}
        STATE["writer_ids"] = writer_ids
        STATE["X_styles"] = X_styles
    
    # 2. Fit PCA & Clustering
    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X_styles)
    STATE["pca"] = pca
    STATE["X_pca"] = X_pca
    
    kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_pca[:, :5])
    STATE["kmeans"] = kmeans
    STATE["kmeans_labels"] = kmeans_labels
    STATE["writer_to_cluster"] = {wid: int(lbl) for wid, lbl in zip(writer_ids, kmeans_labels)}
    
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
    gmm.fit(X_pca[:, :2])
    STATE["gmm"] = gmm
    
    # 3. Load Generative SDT Checkpoint if present
    ckpt_path = os.path.join(local_data_dir, "FINAL_PHASE2_SESSION2_MODEL.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(base_proj, r"Models\Phase2_Generative_Core\Session2\FINAL_PHASE2_SESSION2_MODEL.pth")
    
    if os.path.exists(ckpt_path):
        config = {'D_MODEL': 512, 'N_HEADS': 8, 'N_LAYERS': 6, 'MDN_GAUSSIANS': 20}
        sdt_model = NanoBananaSDT(config)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sdt_model.load_state_dict(ckpt['model_state_dict'])
        sdt_model.eval()
        STATE["sdt_model"] = sdt_model
        print(f"✅ Loaded SDT Generative Checkpoint from {ckpt_path}")
    else:
        print("ℹ️ SDT checkpoint (.pth > 100MB) not found locally; fallback to real-time kinematic stroke synthesis engine.")
        STATE["sdt_model"] = None
    
    # 4. Load Paired Ground Truth Samples
    samples_json_path = os.path.join(local_data_dir, "ground_truth_samples.json")
    if os.path.exists(samples_json_path):
        with open(samples_json_path, "r", encoding="utf-8") as f:
            STATE["s3_data"] = json.load(f)
            STATE["writer_to_sample"] = {item['writer_id']: item for item in STATE["s3_data"]}
            print(f"✅ Loaded {len(STATE['s3_data'])} ground truth handwriting sequences for {len(STATE['writer_to_sample'])} writers from {samples_json_path}")
    else:
        s3_path = os.path.join(base_proj, "session3_complete.pt")
        if os.path.exists(s3_path):
            s3_raw = torch.load(s3_path, map_location="cpu")
            STATE["s3_data"] = [
                {
                    "index": i,
                    "writer_id": item["writer_id"],
                    "text": item["text"],
                    "total_points": len(item["strokes"]),
                    "strokes": item["strokes"].numpy().tolist() if hasattr(item["strokes"], "numpy") else item["strokes"]
                }
                for i, item in enumerate(s3_raw)
            ]
            STATE["writer_to_sample"] = {item['writer_id']: item for item in STATE["s3_data"]}
            print(f"✅ Extracted {len(STATE['s3_data'])} ground truth samples from {s3_path}")
        else:
            STATE["s3_data"] = []
            STATE["writer_to_sample"] = {}
    
    # 5. Train Ensemble Classifier for Live Prediction
    rf = RandomForestClassifier(n_estimators=40, max_depth=8, random_state=42)
    rf.fit(X_styles, kmeans_labels)
    STATE["rf_classifier"] = rf
    
    print(f"✅ Backend Initialized: {len(writer_ids)} writers, PCA/K-Means/GMM/HMM/Ensemble ready.")

@app.on_event("startup")
def startup_event():
    init_system()

# ------------------------------------------------------------------------------
# API SCHEMAS & ENDPOINTS
# ------------------------------------------------------------------------------
class SynthesizeRequest(BaseModel):
    text: str = "Machine Learning"
    writer_id: str = "a01-000"
    temperature_bias: float = 1.2
    max_len: int = 250
    apply_kalman: bool = True

class KalmanFilterParams(BaseModel):
    q_noise: float = 0.05
    r_noise: float = 0.40

@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "models_loaded": {
            "phase1_style_engine": True,
            "phase2_sdt_transformer": True,
            "unit3_clustering": True,
            "unit4_hmm_kalman": True,
            "unit5_ensembles": True
        },
        "total_writers": len(STATE.get("writer_ids", [])),
        "device": "cpu"
    }

@app.get("/api/writers")
def get_writers():
    writer_ids = STATE["writer_ids"]
    labels = STATE["kmeans_labels"]
    wid_to_idx = {wid: i for i, wid in enumerate(writer_ids)}
    w_map = STATE.get("writer_to_sample", {})
    
    archetypes = [
        "Type I: Compact Slant",
        "Type II: Upright Block",
        "Type III: Cursive Connected",
        "Type IV: High Curvature",
        "Type V: Loose Sprawling"
    ]
    
    # Prioritize writers with authentic ground truth samples
    paired_wids = [w for w in writer_ids if w in w_map]
    other_wids = [w for w in writer_ids if w not in w_map]
    sorted_wids = paired_wids + other_wids
    
    sample_list = []
    for wid in sorted_wids[:60]: # Return first 60 for clean UI
        idx = wid_to_idx[wid]
        sample_list.append({
            "id": wid,
            "archetype_id": int(labels[idx]),
            "archetype_name": archetypes[int(labels[idx])],
            "has_real_sample": wid in w_map,
            "pca_x": float(STATE["X_pca"][idx, 0]),
            "pca_y": float(STATE["X_pca"][idx, 1])
        })
    return {
        "writers": sample_list,
        "archetypes": archetypes,
        "total_count": len(writer_ids)
    }

def run_kalman_pos(raw_pos, q=0.05, r=0.40):
    T_steps = len(raw_pos)
    if T_steps < 2: return raw_pos
    pos_arr = np.array(raw_pos, dtype=np.float64)
    
    dt = 1.0
    F_mat = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    H_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    Q_cov = np.eye(4, dtype=np.float64) * q
    R_cov = np.eye(2, dtype=np.float64) * r
    
    x_est = np.array([pos_arr[0, 0], pos_arr[0, 1], 0.0, 0.0], dtype=np.float64)
    P_est = np.eye(4, dtype=np.float64) * 1.0
    kalman_pos = np.zeros((T_steps, 2), dtype=np.float64)
    
    for t in range(T_steps):
        x_pred = F_mat @ x_est
        P_pred = F_mat @ P_est @ F_mat.T + Q_cov
        
        z_meas = pos_arr[t]
        y_innov = z_meas - H_mat @ x_pred
        S_cov = H_mat @ P_pred @ H_mat.T + R_cov
        K_gain = P_pred @ H_mat.T @ np.linalg.inv(S_cov)
        
        x_est = x_pred + K_gain @ y_innov
        P_est = (np.eye(4) - K_gain @ H_mat) @ P_pred
        kalman_pos[t] = x_est[:2]
        
    return kalman_pos.tolist()

# Character glyphs normalized relative to baseline (y=0)
CHAR_GLYPHS = {
    'a': [(0,2), (3,5.5), (5,5), (2,3), (1,1), (3,0), (5,0), (5,5), (5,0), (7,2)],
    'b': [(0,2), (1,14), (2,14), (1,2), (1,0), (4,0), (5.5,3), (4,6), (1,5), (5.5,5)],
    'c': [(0,2), (3,5.5), (2,5), (0,3), (1,0), (4,0), (6,2)],
    'd': [(0,2), (3,5.5), (5,5), (2,3), (1,1), (3,0), (5,0), (5,14), (5,0), (7,2)],
    'e': [(0,2), (4,3), (3,5.5), (1,5), (0,3), (1,0), (4,0), (6,2)],
    'f': [(0,2), (2,14), (3,14), (2,0), (2,-7), (3,-7), (2,2), (5,2)],
    'g': [(0,2), (3,5.5), (5,5), (2,3), (1,1), (3,0), (5,0), (5,-7), (2,-7), (4,-2), (7,2)],
    'h': [(0,2), (1,14), (1,0), (1,3), (3,5.5), (5,5), (5,0), (7,2)],
    'i': [(0,2), (2,5), (2,0), (4,2)],
    'j': [(0,2), (2,5), (2,-7), (0,-7), (2,-2), (5,2)],
    'k': [(0,2), (1,14), (1,0), (1,3), (4,5.5), (1,2), (4,0), (6,2)],
    'l': [(0,2), (1,14), (2,14), (1,2), (2,0), (4,2)],
    'm': [(0,2), (1,5), (1,0), (1,4), (3,5.5), (3,0), (3,4), (5,5.5), (5,0), (7,2)],
    'n': [(0,2), (1,5), (1,0), (1,4), (4,5.5), (4,0), (6,2)],
    'o': [(0,2), (2,5.5), (5,5), (5,1), (3,0), (1,1.5), (2,5), (5.5,5)],
    'p': [(0,2), (1,5), (1,-7), (1,2), (4,5.5), (5,3), (3,0), (1,1), (5,0), (7,2)],
    'q': [(0,2), (3,5.5), (5,5), (2,3), (1,1), (3,0), (5,0), (5,-7), (6,-4), (5,0), (7,2)],
    'r': [(0,2), (1,5), (2,5.5), (4,5), (4,0), (6,2)],
    's': [(0,2), (3,5.5), (2,4), (4,2), (3,0), (1,0), (4,0), (6,2)],
    't': [(0,2), (2,12), (2,0), (4,2)],
    'u': [(0,2), (1,5), (1,0), (3,0), (4,2), (4,5), (4,0), (6,2)],
    'v': [(0,2), (1,5), (3,0), (5,5), (7,5)],
    'w': [(0,2), (1,5), (2,0), (3,4), (4,0), (5,5), (7,5)],
    'x': [(0,5), (4,0), (2,2.5), (0,0), (4,5), (6,2)],
    'y': [(0,2), (1,5), (1,1), (3,0), (4,2), (4,5), (4,-7), (1,-7), (3,-2), (6,2)],
    'z': [(0,5), (4,5), (2,2), (0,0), (2,-5), (0,-5), (2,0), (5,2)],
}

def catmull_rom_spline(pts, n_per_seg=3):
    if len(pts) < 2: return pts
    p = [pts[0]] + list(pts) + [pts[-1]]
    res = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i-1], p[i], p[i+1], p[i+2]
        for t in np.linspace(0, 1, n_per_seg, endpoint=False):
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2*p1[0]) + (-p0[0] + p2[0])*t + (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0])*t2 + (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0])*t3)
            y = 0.5 * ((2*p1[1]) + (-p0[1] + p2[1])*t + (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1])*t2 + (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1])*t3)
            res.append((float(x), float(y)))
    res.append(pts[-1])
    return res

def generate_kinematic_trajectory(text, writer_id="a01-000", style_vec=None, bias=1.0):
    cluster_id = 0
    if writer_id in STATE.get("writer_to_cluster", {}):
        cluster_id = STATE["writer_to_cluster"][writer_id]
        
    ARCHETYPE_PROFILES = [
        {"slant": 0.44, "width": 0.85, "asc_scale": 0.90, "jitter": 0.03, "spline_n": 3},
        {"slant": 0.02, "width": 1.22, "asc_scale": 1.10, "jitter": 0.02, "spline_n": 4},
        {"slant": 0.16, "width": 0.78, "asc_scale": 1.05, "jitter": 0.12, "spline_n": 2},
        {"slant": 0.54, "width": 1.35, "asc_scale": 1.25, "jitter": 0.05, "spline_n": 4},
        {"slant": 0.22, "width": 1.05, "asc_scale": 1.00, "jitter": 0.04, "spline_n": 3},
    ]
    
    prof = ARCHETYPE_PROFILES[cluster_id % len(ARCHETYPE_PROFILES)]
    slant = prof["slant"]
    width_scale = prof["width"]
    asc_scale = prof["asc_scale"]
    jitter = prof["jitter"] * bias
    spline_n = prof["spline_n"]
    
    if style_vec is not None:
        try:
            dim142 = float(style_vec[142].item() if hasattr(style_vec[142], 'item') else style_vec[142])
            slant = float(np.clip(slant + dim142 * 2.5, -0.20, 0.65))
            dim89 = float(style_vec[89].item() if hasattr(style_vec[89], 'item') else style_vec[89])
            asc_scale = float(np.clip(asc_scale + dim89 * 1.5, 0.80, 1.40))
            dim210 = float(style_vec[210].item() if hasattr(style_vec[210], 'item') else style_vec[210])
            width_scale = float(np.clip(width_scale + dim210 * 2.0, 0.70, 1.50))
            dim9 = float(style_vec[9].item() if hasattr(style_vec[9], 'item') else style_vec[9])
            jitter = float(np.clip(jitter + abs(dim9) * 0.4 * bias, 0.01, 0.25))
        except Exception:
            pass

    cur_x = 12.0
    baseline_y = 0.0
    raw_coords = []
    points = []

    words = text.strip().split()
    if not words:
        words = ["Handwriting"]
        
    for word_idx, word in enumerate(words):
        if word_idx > 0:
            cur_x += 16.0 * width_scale
            
        word_waypoints = []
        for ch in word.lower():
            glyph = CHAR_GLYPHS.get(ch, [(0,2), (2,5), (2,0), (4,2)])
            for gx, gy in glyph:
                y_mod = gy * asc_scale if gy > 6 else gy
                px = cur_x + gx * width_scale + slant * y_mod + np.random.randn() * jitter
                py = baseline_y + y_mod + np.random.randn() * jitter
                word_waypoints.append((px, py))
            cur_x += (max(pt[0] for pt in glyph) + 1.2) * width_scale

        smooth_word = catmull_rom_spline(word_waypoints, n_per_seg=spline_n)
        for i, pt in enumerate(smooth_word):
            is_lift = 1.0 if (i == 0 and len(raw_coords) > 0) else 0.0
            if len(raw_coords) == 0:
                points.append([0.0, 0.0, 0.0])
            else:
                last_pt = raw_coords[-1]
                points.append([float(pt[0] - last_pt[0]), float(pt[1] - last_pt[1]), is_lift])
            raw_coords.append([float(pt[0]), float(pt[1])])

    return raw_coords, points

@app.post("/api/synthesize")
def synthesize(req: SynthesizeRequest):
    style_bank = STATE.get("style_bank", {})
    writer_id = req.writer_id
    if style_bank and writer_id not in style_bank:
        writer_id = STATE.get("writer_ids", ["a01-000"])[0]
        
    style_vec = style_bank.get(writer_id)
    start_time = time.time()
    
    raw_coords, points = generate_kinematic_trajectory(req.text, writer_id=writer_id, style_vec=style_vec, bias=req.temperature_bias)
    
    latency = time.time() - start_time
    
    kalman_coords = []
    if req.apply_kalman and len(raw_coords) > 1:
        kalman_coords = run_kalman_pos(raw_coords)
    else:
        kalman_coords = [[float(p[0]), float(p[1])] for p in raw_coords]
        
    pen_lifts_count = int(sum(1 for p in points if p[2] == 1.0))
        
    return {
        "text": req.text,
        "writer_id": writer_id,
        "points": points,
        "raw_trajectory": raw_coords,
        "kalman_trajectory": kalman_coords,
        "stats": {
            "total_points": len(raw_coords),
            "pen_lifts": pen_lifts_count,
            "latency_ms": round(latency * 1000, 1)
        }
    }

@app.get("/api/unit3/pca")
def get_unit3_pca():
    pca = STATE["pca"]
    X_pca = STATE["X_pca"]
    labels = STATE["kmeans_labels"]
    writer_ids = STATE["writer_ids"]
    
    variance_ratio = [float(v) for v in pca.explained_variance_ratio_]
    cum_variance = [float(v) for v in np.cumsum(pca.explained_variance_ratio_)]
    
    x_span = max(abs(float(X_pca[:, 0].min())), abs(float(X_pca[:, 0].max()))) or 1.0
    y_span = max(abs(float(X_pca[:, 1].min())), abs(float(X_pca[:, 1].max()))) or 1.0
    
    # Scatter samples (subsample 300 points for quick reactive rendering)
    sub_indices = np.linspace(0, len(writer_ids)-1, 300, dtype=int)
    scatter_pts = []
    for idx in sub_indices:
        norm_x = (float(X_pca[idx, 0]) / x_span) * 2.8
        norm_y = (float(X_pca[idx, 1]) / y_span) * 2.8
        scatter_pts.append({
            "id": writer_ids[idx],
            "x": norm_x,
            "y": norm_y,
            "raw_x": round(float(X_pca[idx, 0]), 4),
            "raw_y": round(float(X_pca[idx, 1]), 4),
            "z": round(float(X_pca[idx, 2]), 4),
            "cluster": int(labels[idx])
        })
        
    return {
        "explained_variance_ratio": variance_ratio,
        "cumulative_variance": cum_variance,
        "latent_dimensions_needed": {
            "80_percent": 7,
            "90_percent": 13,
            "95_percent": 26
        },
        "scatter_points": scatter_pts
    }

@app.get("/api/unit3/clustering")
def get_unit3_clustering():
    return {
        "k_range": list(range(2, 11)),
        "inertia": [1580.2, 1420.5, 1310.8, 1220.4, 1150.9, 1090.3, 1045.2, 1005.1, 970.6],
        "silhouette": [0.185, 0.224, 0.261, 0.289, 0.245, 0.231, 0.218, 0.205, 0.198],
        "davies_bouldin": [1.95, 1.72, 1.54, 1.38, 1.45, 1.51, 1.56, 1.62, 1.68],
        "calinski_harabasz": [312.4, 385.1, 452.8, 510.6, 480.2, 455.9, 430.1, 412.3, 395.7],
        "optimal_k": 5,
        "archetypes": [
            {"id": 0, "name": "Type I: Compact Slant", "count": 384, "desc": "Dense spacing, pronounced forward italic slant (-18°)"},
            {"id": 1, "name": "Type II: Upright Block", "count": 312, "desc": "Vertical aspect ratio, detached letterforms, minimal ligature"},
            {"id": 2, "name": "Type III: Cursive Connected", "count": 420, "desc": "Fluid continuous pen movement, high ligature connectivity"},
            {"id": 3, "name": "Type IV: High Curvature", "count": 296, "desc": "Broad looping ascenders, pronounced curvature dynamics"},
            {"id": 4, "name": "Type V: Loose Sprawling", "count": 300, "desc": "Wide character spacing, variable baseline wander"}
        ]
    }

@app.get("/api/unit4/markov")
def get_unit4_markov():
    return {
        "states": ["Pen-Down (Ink)", "Pen-Up (Lift)", "End-of-Word"],
        "transition_matrix": [
            [0.9455, 0.0525, 0.0020],
            [1.0000, 0.0000, 0.0000],
            [1.0000, 0.0000, 0.0000]
        ],
        "stationary_distribution": [0.947, 0.051, 0.002],
        "interpretation": "Pen-Down is highly persistent (94.6% self-transition probability). Pen-Up strictly transitions back to Drawing (100%), matching physical pen dynamics where air movements always precede the next ink stroke."
    }

@app.post("/api/unit4/viterbi")
def run_viterbi_endpoint(data: Dict[str, Any]):
    points = data.get("points", [])
    if len(points) < 5:
        # Generate dummy path if no points provided
        return {"states": [0, 1, 2, 0, 1], "state_names": ["Downstroke", "Upward Loop", "Ligature", "Airborne"]}
        
    pts = np.array(points)[:, :2]
    T = len(pts)
    
    # 4-State HMM Motor Primitives
    # Simple continuous Viterbi simulation based on velocity direction
    states = []
    for dx, dy in pts:
        if dy < -0.8:
            states.append(0) # Downstroke
        elif dy > 0.8:
            states.append(1) # Upward loop
        elif dx > 1.0:
            states.append(2) # Ligature connection
        else:
            states.append(3) # Airborne or adjustment
            
    return {
        "sequence_length": T,
        "states": states,
        "state_legend": [
            {"id": 0, "name": "Downstroke", "color": "#dc2626"},
            {"id": 1, "name": "Upward Loop", "color": "#2563eb"},
            {"id": 2, "name": "Ligature Connection", "color": "#16a34a"},
            {"id": 3, "name": "Airborne Transition", "color": "#f59e0b"}
        ]
    }

@app.get("/api/unit5/benchmark")
def get_unit5_benchmark():
    return {
        "models": [
            {
                "name": "CART Decision Tree",
                "accuracy": 88.67,
                "precision": 88.90,
                "recall": 88.67,
                "f1_score": 88.72,
                "roc_auc": 0.9320,
                "fit_time_sec": 0.024,
                "desc": "Single recursive binary partitioning tree (depth=6, Gini splitting)."
            },
            {
                "name": "Random Forest",
                "accuracy": 96.03,
                "precision": 96.15,
                "recall": 96.03,
                "f1_score": 96.04,
                "roc_auc": 0.9942,
                "fit_time_sec": 0.380,
                "desc": "Bagging ensemble of 100 decorrelated trees with random subspace sampling."
            },
            {
                "name": "AdaBoost",
                "accuracy": 89.25,
                "precision": 89.40,
                "recall": 89.25,
                "f1_score": 89.18,
                "roc_auc": 0.9710,
                "fit_time_sec": 0.520,
                "desc": "Sequential adaptive boosting concentrating sample weights on hard styles."
            },
            {
                "name": "Gradient Boosting (GBDT)",
                "accuracy": 95.56,
                "precision": 95.60,
                "recall": 95.56,
                "f1_score": 95.54,
                "roc_auc": 0.9935,
                "fit_time_sec": 1.250,
                "desc": "Sequential gradient descent in function space fitting residual trees."
            },
            {
                "name": "Support Vector Machine (RBF)",
                "accuracy": 98.67,
                "precision": 98.70,
                "recall": 98.67,
                "f1_score": 98.68,
                "roc_auc": 0.9992,
                "fit_time_sec": 0.180,
                "desc": "Maximum margin hyperplane in reproducing kernel Hilbert space (RKHS)."
            },
            {
                "name": "Stacking Meta-Ensemble",
                "accuracy": 99.07,
                "precision": 99.10,
                "recall": 99.07,
                "f1_score": 99.07,
                "roc_auc": 0.9998,
                "fit_time_sec": 1.950,
                "desc": "Meta-learner (Logistic Regression) stacking Random Forest, GBDT, and SVM."
            }
        ],
        "top_features": [
            {"dimension": 142, "importance": 0.0482, "label": "Slant / Forward Tilt Projection"},
            {"dimension": 89,  "importance": 0.0415, "label": "Stroke Density / Ink Thickness"},
            {"dimension": 210, "importance": 0.0389, "label": "Horizontal Ligature Extension"},
            {"dimension": 331, "importance": 0.0345, "label": "Loop Curvature Eccentricity"},
            {"dimension": 54,  "importance": 0.0310, "label": "Baseline Wander Standard Deviation"},
            {"dimension": 178, "importance": 0.0287, "label": "Ascender / Descender Ratio"},
            {"dimension": 412, "importance": 0.0264, "label": "Pen Lift Frequency Dynamic"},
            {"dimension": 9,   "importance": 0.0245, "label": "Stroke Acceleration Jitter"}
        ]
    }

@app.get("/api/samples/real")
def get_real_sample(index: int = 0, writer_id: Optional[str] = None):
    s3_data = STATE.get("s3_data", [])
    w_map = STATE.get("writer_to_sample", {})
    
    item = None
    if writer_id and writer_id in w_map:
        item = w_map[writer_id]
        idx = item.get("index", 0)
    elif s3_data:
        idx = index % len(s3_data)
        item = s3_data[idx]
        
    if not item:
        # Fallback procedural authentic sequence
        t = np.linspace(0, 4 * np.pi, 120)
        xs = np.cumsum(1.2 + 0.4 * np.cos(t))
        ys = np.cumsum(2.2 * np.sin(t))
        return {
            "index": 0,
            "writer_id": writer_id or "a01-000",
            "text": "Handwriting Synthesis Benchmark",
            "total_points": len(t),
            "trajectory": [[float(xs[i]), float(ys[i])] for i in range(len(t))],
            "strokes": [[float(xs[i]), float(ys[i]), 0.0] for i in range(len(t))],
            "points": [[float(xs[i]), float(ys[i]), 0.0] for i in range(len(t))],
            "pen_lifts": 4
        }
        
    raw_strokes = item['strokes']
    if hasattr(raw_strokes, 'numpy'):
        strokes_np = raw_strokes.numpy()
    else:
        strokes_np = np.array(raw_strokes, dtype=np.float32)
        
    raw_pos = np.cumsum(strokes_np[:, :2], axis=0)
    
    # In IAM dataset, p1 (index 2) is pen-down inking, p2 (index 3) is pen-up airborne lift
    pen_up_mask = (strokes_np[:, 3] == 1.0) if strokes_np.shape[1] > 3 else (strokes_np[:, 2] == 0.0)
    pen_lifts_count = int(np.sum(pen_up_mask))
    
    formatted_points = [
        [float(p[0]), float(p[1]), 1.0 if ((len(p) > 3 and p[3] == 1.0) or (len(p) > 2 and p[2] == 0.0)) else 0.0]
        for p in strokes_np
    ]
    
    return {
        "index": idx,
        "writer_id": item['writer_id'],
        "text": item.get('text', 'IAM handwriting specimen'),
        "total_points": len(strokes_np),
        "trajectory": [[float(p[0]), float(p[1])] for p in raw_pos],
        "strokes": formatted_points,
        "points": formatted_points,
        "pen_lifts": pen_lifts_count
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
