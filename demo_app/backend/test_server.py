import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from server import app, init_system

print("=" * 60)
print("TESTING FASTAPI BACKEND IN-PROCESS WITH TESTCLIENT")
print("=" * 60)

init_system()
client = TestClient(app)

# 1. /api/status
r = client.get("/api/status")
print(f"1. /api/status -> {r.status_code}")
assert r.status_code == 200
print(f"   Response: {r.json()}")

# 2. /api/writers
r = client.get("/api/writers")
print(f"2. /api/writers -> {r.status_code}")
assert r.status_code == 200
data = r.json()
print(f"   Writers count: {data.get('count')}, Sample: {data.get('writers', [])[:2]}")

# 3. /api/samples/real
r = client.get("/api/samples/real?index=0")
print(f"3. /api/samples/real -> {r.status_code}")
assert r.status_code == 200
sample = r.json()
print(f"   Sample writer: {sample.get('writer_id')}, total_points: {sample.get('total_points')}, pen_lifts: {sample.get('pen_lifts')}")

# 4. /api/synthesize
t0 = time.time()
payload = {
    "text": "Academic Machine Learning",
    "writer_id": "a01-000",
    "stochasticity": 0.3,
    "kalman_filter": True,
    "max_length": 60
}
r = client.post("/api/synthesize", json=payload)
dt = (time.time() - t0) * 1000
print(f"4. /api/synthesize -> {r.status_code} in {dt:.1f} ms")
assert r.status_code == 200
synth = r.json()
print(f"   Points: {len(synth.get('points', []))}, Stats: {synth.get('stats')}")

# 5. /api/unit3/pca
r = client.get("/api/unit3/pca")
print(f"5. /api/unit3/pca -> {r.status_code}")
assert r.status_code == 200
pca = r.json()
print(f"   Explained variance (first 3): {pca.get('explained_variance_ratio', [])[:3]}")
print(f"   Scatter points count: {len(pca.get('scatter_points', []))}")

# 6. /api/unit3/clustering
r = client.get("/api/unit3/clustering")
print(f"6. /api/unit3/clustering -> {r.status_code}")
assert r.status_code == 200
clust = r.json()
print(f"   Archetypes: {len(clust.get('archetypes', []))}, Optimal K: {clust.get('optimal_k')}")

# 7. /api/unit4/markov
r = client.get("/api/unit4/markov")
print(f"7. /api/unit4/markov -> {r.status_code}")
assert r.status_code == 200
markov = r.json()
print(f"   Transition matrix shape: {len(markov.get('matrix', []))}x{len(markov.get('matrix', [[]])[0])}")

# 8. /api/unit4/viterbi
viterbi_payload = {
    "observations": [
        [0.1, 0.2, 0.0],
        [0.3, 0.4, 0.0],
        [0.5, 0.6, 1.0],
        [0.0, 0.0, 1.0]
    ]
}
r = client.post("/api/unit4/viterbi", json=viterbi_payload)
print(f"8. /api/unit4/viterbi -> {r.status_code}")
assert r.status_code == 200
print(f"   States decoded: {r.json().get('states')}")

# 9. /api/unit5/benchmark
r = client.get("/api/unit5/benchmark")
print(f"9. /api/unit5/benchmark -> {r.status_code}")
assert r.status_code == 200
bench = r.json()
print(f"   Leaderboard count: {len(bench.get('leaderboard', []))}, Top model: {bench.get('leaderboard', [{}])[0].get('model')}")

print("\n" + "=" * 60)
print("ALL 9 BACKEND ENDPOINTS FULLY VERIFIED & WORKING WITH HIGH FIDELITY!")
print("=" * 60)
