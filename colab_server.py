# ==============================================================================
# 🚀 BUCKBUCK AI • ULTIMATE ENTERPRISE COLAB BACKEND
# WITH SMART GPU PRESERVATION & SESSION TIMER WATCHDOG
# ==============================================================================

# ------------------------------------------------------------------------------
# ⏱️ GPU QUOTA & SESSION PRESERVATION SETTINGS (Customize as needed)
# ------------------------------------------------------------------------------
SESSION_LIMIT_MINUTES = 120       # Max continuous session (e.g. 2 hours)
AUTO_SHUTDOWN_ON_IDLE = True     # Auto-stop if you leave without closing
IDLE_TIMEOUT_MINUTES = 25        # Shut down after 25 mins of zero activity
AUTO_RELEASE_GPU = True          # Release Colab VM on expiry to preserve daily quota
# ------------------------------------------------------------------------------

import os, sys, time, subprocess, threading, re, json, io, base64, traceback

SERVER_START_TIME = time.time()
LAST_ACTIVITY_TIME = time.time()

# [1/6] MOUNT GOOGLE DRIVE FOR PERMANENT MODEL PERSISTENCE
try:
    from google.colab import drive
    print("📁 [1/6] Mounting Google Drive for permanent model cache...")
    drive.mount('/content/drive')
    os.makedirs('/content/drive/MyDrive/ollama_models', exist_ok=True)
    os.environ['OLLAMA_MODELS'] = '/content/drive/MyDrive/ollama_models'
except Exception as e:
    print("⚠️ Running outside Colab or Drive skipped. Using local disk.")

# [2/6] CONFIGURE HIGH-PERFORMANCE GPU ENVIRONMENT
os.environ['OLLAMA_ORIGINS'] = '*'
os.environ['OLLAMA_HOST'] = '127.0.0.1:11434'
os.environ['OLLAMA_FLASH_ATTENTION'] = '1'
os.environ['OLLAMA_NUM_PARALLEL'] = '2'
os.environ['OLLAMA_KEEP_ALIVE'] = '24h'

# [3/6] INSTALL SYSTEM DEPENDENCIES & PACKAGES
print("⏳ [2/6] Installing Ollama, Cloudflare Tunnel & AI Python Suite...")
os.system("sudo apt-get update -qq && sudo apt-get install -y zstd pciutils > /dev/null 2>&1")
os.system("curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1")
os.system("curl -s -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb && sudo dpkg -i cloudflared.deb > /dev/null 2>&1")
os.system("pip install -q fastapi uvicorn httpx pydantic matplotlib numpy sentence-transformers > /dev/null 2>&1")

# [4/6] START OLLAMA ENGINE IN BACKGROUND
print("⚡ [3/6] Starting Ollama Engine with FlashAttention & GPU Locking...")
subprocess.Popen(["ollama", "serve"], env=dict(os.environ))
time.sleep(3)

# Pull primary models to Google Drive
print("📥 [4/6] Ensuring core models are cached in Google Drive...")
subprocess.run(["ollama", "pull", "deepseek-r1:7b"])
subprocess.run(["ollama", "pull", "qwen2.5:7b"])
subprocess.run(["ollama", "pull", "deepseek-r1:1.5b"])

# [5/6] FASTAPI GATEWAY WITH CODE EXECUTION & WATCHDOG
print("🧠 [5/6] Initializing Enterprise Backend Gateway (FastAPI + GPU Executor)...")

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import httpx, uvicorn

app = FastAPI(title="BuckBuck Neural Backend Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_INTERNAL_URL = "http://127.0.0.1:11434"

def record_activity():
    global LAST_ACTIVITY_TIME
    LAST_ACTIVITY_TIME = time.time()

class CodeExecutionRequest(BaseModel):
    code: str

# 1. PYTHON GPU CODE EXECUTION (INTERCEPTS PRINT STATEMENTS & PLOTS)
@app.post("/api/exec")
async def execute_python_code(req: CodeExecutionRequest):
    record_activity()
    code = req.code
    start_time = time.time()
    exec_scope = {"__name__": "__main__", "sys": sys, "os": os}
    old_stdout, old_stderr = sys.stdout, sys.stderr
    redirected_stdout, redirected_stderr = io.StringIO(), io.StringIO()
    plots_base64 = []
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.close('all')
        
        sys.stdout, sys.stderr = redirected_stdout, redirected_stderr
        exec(code, exec_scope)
        
        for fig in [plt.figure(i) for i in plt.get_fignums()]:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
            buf.seek(0)
            plots_base64.append(base64.b64encode(buf.read()).decode('utf-8'))
            plt.close(fig)
            
        stdout_text = redirected_stdout.getvalue()
        stderr_text = redirected_stderr.getvalue()
        is_success = True
        error_msg = ""
    except Exception as e:
        stdout_text = redirected_stdout.getvalue()
        stderr_text = traceback.format_exc()
        is_success = False
        error_msg = str(e)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    return {
        "success": is_success,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "error": error_msg,
        "plots": plots_base64,
        "execution_time_seconds": round(time.time() - start_time, 3)
    }

# 2. SYSTEM HEALTH & GPU VRAM STATS (WITH SESSION COUNTDOWN)
@app.get("/api/health")
async def get_system_health():
    import torch
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_free = "15.0 GB"
    if torch.cuda.is_available():
        free_b, _ = torch.cuda.mem_get_info(0)
        vram_free = f"{free_b / (1024**3):.1f} GB"
        
    elapsed_minutes = (time.time() - SERVER_START_TIME) / 60
    remaining_minutes = max(0, int(SESSION_LIMIT_MINUTES - elapsed_minutes))
    idle_minutes = int((time.time() - LAST_ACTIVITY_TIME) / 60)
    
    return {
        "status": "healthy",
        "gpu": gpu_name,
        "vram_free": vram_free,
        "session_remaining_minutes": remaining_minutes,
        "idle_minutes": idle_minutes
    }

# 3. REVERSE PROXY FOR OLLAMA (/api/chat, /api/tags, etc.)
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
async def reverse_proxy_ollama(request: Request, path: str):
    record_activity()
    target_url = f"{OLLAMA_INTERNAL_URL}/{path}"
    headers = dict(request.headers)
    headers.pop("host", None)
    client = httpx.AsyncClient(timeout=None)
    req_body = await request.body()
    try:
        ollama_req = client.build_request(
            method=request.method, url=target_url, headers=headers,
            params=request.query_params, content=req_body
        )
        response = await client.send(ollama_req, stream=True)
        return StreamingResponse(
            response.aiter_raw(), status_code=response.status_code,
            headers=dict(response.headers), background=client.aclose
        )
    except Exception as e:
        await client.aclose()
        return JSONResponse({"error": str(e)}, status_code=502)

threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning"), daemon=True).start()
time.sleep(2)

# [6/6] CLOUDFLARE TUNNEL
tunnel_cmd = ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000"]
tunnel_process = subprocess.Popen(tunnel_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

for line in tunnel_process.stderr:
    match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
    if match:
        url = match.group(0)
        print("\n" + "="*65)
        print("🎉 BUCKBUCK ENTERPRISE GPU BACKEND IS ONLINE!")
        print(f"🔗 YOUR BACKEND URL: {url}")
        print(f"⏱️ SESSION LIMIT: {SESSION_LIMIT_MINUTES} mins | IDLE AUTO-STOP: {IDLE_TIMEOUT_MINUTES} mins")
        print("="*65)
        print("👉 Copy this URL and paste it into Settings (⚙️) on https://buckbuck.pages.dev")
        print("="*65)
        break

# ------------------------------------------------------------------------------
# 🛡️ AUTOMATED GPU PRESERVATION WATCHDOG
# ------------------------------------------------------------------------------
def session_watchdog():
    while True:
        time.sleep(30)
        elapsed = (time.time() - SERVER_START_TIME) / 60
        idle = (time.time() - LAST_ACTIVITY_TIME) / 60

        # Check for Session Expiry
        if elapsed >= SESSION_LIMIT_MINUTES:
            print("\n" + "="*65)
            print(f"🛑 [WATCHDOG] Session limit of {SESSION_LIMIT_MINUTES} mins reached!")
            print("💾 Models safely preserved in Google Drive.")
            print("🛡️ Releasing Colab GPU to protect your daily compute quota...")
            print("="*65)
            shutdown_backend()
            break

        # Check for Idle Timeout
        if AUTO_SHUTDOWN_ON_IDLE and idle >= IDLE_TIMEOUT_MINUTES:
            print("\n" + "="*65)
            print(f"💤 [WATCHDOG] Inactive for {IDLE_TIMEOUT_MINUTES} mins. Auto-stopping GPU...")
            print("💾 Models safely preserved in Google Drive.")
            print("="*65)
            shutdown_backend()
            break

def shutdown_backend():
    try:
        tunnel_process.kill()
    except:
        pass
    if AUTO_RELEASE_GPU:
        try:
            from google.colab import runtime
            runtime.unassign()
        except:
            os._exit(0)
    else:
        os._exit(0)

threading.Thread(target=session_watchdog, daemon=True).start()

while True:
    time.sleep(60)
