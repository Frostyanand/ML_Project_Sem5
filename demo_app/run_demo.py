import os
import sys
import time
import subprocess
import threading
import webbrowser
import signal

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(PROJECT_DIR, "backend")

print("=" * 78)
print("  HYPER-REALISTIC HANDWRITING SYNTHESIS PLATFORM - DEMO LAUNCHER")
print("  Machine Learning Curriculum & Generative Synthesis Demonstration")
print("=" * 78)
print("  [1/4] Checking and freeing ports 8000 and 3000...")

def free_port(port):
    try:
        # Find PID using port
        cmd = f"netstat -ano | findstr :{port}"
        out = subprocess.check_output(cmd, shell=True, text=True, errors="ignore")
        for line in out.strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 5 and "LISTENING" in parts:
                pid = parts[-1]
                subprocess.run(f"taskkill /F /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"        Freed port {port} (killed stale PID {pid})")
    except Exception:
        pass

free_port(8000)
free_port(3000)

print("  [2/4] Starting FastAPI Model Server (PyTorch SDT & ViT Manifold) on :8000...")
api_cmd = [sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "info"]
p_api = subprocess.Popen(
    api_cmd,
    cwd=BACKEND_DIR,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="replace",
    bufsize=1
)

def stream_api_logs():
    for line in iter(p_api.stdout.readline, ''):
        if line.strip():
            print(f"  [API] {line.strip()}", flush=True)

threading.Thread(target=stream_api_logs, daemon=True).start()

# Wait 3 seconds for server startup
time.sleep(3.5)

print("  [3/4] Starting Next.js Production Web Interface on :3000...")
# On Windows, npm is npm.cmd
npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
ui_cmd = [npm_cmd, "start"]
p_ui = subprocess.Popen(
    ui_cmd,
    cwd=PROJECT_DIR,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="replace",
    bufsize=1
)

def stream_ui_logs():
    for line in iter(p_ui.stdout.readline, ''):
        if line.strip():
            print(f"  [WEB] {line.strip()}", flush=True)

threading.Thread(target=stream_ui_logs, daemon=True).start()

# Wait 2 seconds and open browser
time.sleep(2.5)
print("  [4/4] Opening Interactive Prototype in Default Browser: http://localhost:3000")
print("=" * 78)
print("  ✨ System is LIVE! Press Ctrl+C at any time to shutdown and free resources.")
print("=" * 78)

webbrowser.open("http://localhost:3000")

def cleanup(*args):
    print("\n\n" + "=" * 78)
    print("  🛑 SHUTTING DOWN DEMO PLATFORM & FREEING ALL ALLOCATED RESOURCES...")
    print("=" * 78)
    try:
        if p_ui.poll() is None:
            subprocess.run(f"taskkill /F /T /PID {p_ui.pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("  ✓ Next.js web server stopped.")
    except Exception:
        pass
    try:
        if p_api.poll() is None:
            subprocess.run(f"taskkill /F /T /PID {p_api.pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("  ✓ FastAPI PyTorch model server stopped.")
    except Exception:
        pass
    free_port(8000)
    free_port(3000)
    print("  ✓ Ports 8000 and 3000 released.")
    print("  ✓ All resources successfully freed. Goodbye!\n")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

try:
    while True:
        time.sleep(1)
        if p_api.poll() is not None:
            print("  ⚠️ API Server exited unexpectedly.")
            break
        if p_ui.poll() is not None:
            print("  ⚠️ Next.js Server exited.")
            break
except KeyboardInterrupt:
    cleanup()
finally:
    cleanup()

