# ==============================================================================
# 🚀 BUCKBUCK AI • ENTERPRISE COLAB BACKEND (HARDENED)
# WITH AUTH, NON-BLOCKING EXECUTION, GPU PRESERVATION & SESSION WATCHDOG
# ==============================================================================

# ------------------------------------------------------------------------------
# ⚙️ CONFIG — customize as needed
# ------------------------------------------------------------------------------
SESSION_LIMIT_MINUTES = 120        # Max continuous session (e.g. 2 hours)
AUTO_SHUTDOWN_ON_IDLE = True       # Auto-stop if you leave without closing
IDLE_TIMEOUT_MINUTES = 25          # Shut down after N mins of zero activity
AUTO_RELEASE_GPU = True            # Release Colab VM on expiry to preserve daily quota
EXEC_TIMEOUT_SECONDS = 30          # Hard cap on any single /api/exec call
ALLOWED_ORIGIN = "https://buckbuck.pages.dev"  # Lock CORS to your actual frontend
# ------------------------------------------------------------------------------

import os, sys, time, subprocess, threading, re, json, io, base64, secrets, traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

SERVER_START_TIME = time.time()
LAST_ACTIVITY_TIME = time.time()

# ------------------------------------------------------------------------------
# 🔑 API KEY — generated fresh each run, printed once. Put it in the frontend's
# settings alongside the backend URL. Every request must send:
#   Authorization: Bearer <API_KEY>
# ------------------------------------------------------------------------------
API_KEY = secrets.token_urlsafe(32)

# [1/6] MOUNT GOOGLE DRIVE FOR PERMANENT MODEL PERSISTENCE
try:
    from google.colab import drive
    print("📁 [1/6] Mounting Google Drive for permanent model cache...")
    drive.mount('/content/drive')
    os.makedirs('/content/drive/MyDrive/ollama_models', exist_ok=True)
    os.environ['OLLAMA_MODELS'] = '/content/drive/MyDrive/ollama_models'
except Exception:
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
os.system(
    "curl -s -L https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-linux-amd64.deb -o cloudflared.deb && sudo dpkg -i cloudflared.deb > /dev/null 2>&1"
)
os.system(
    "pip install -q fastapi uvicorn httpx pydantic matplotlib numpy "
    "sentence-transformers torch > /dev/null 2>&1"
)

# [4/6] START OLLAMA ENGINE IN BACKGROUND
print("⚡ [3/6] Starting Ollama Engine with FlashAttention & GPU Locking...")
subprocess.Popen(["ollama", "serve"], env=dict(os.environ))
time.sleep(3)

# Pull primary models to Google Drive — skip anything already cached
print("📥 [4/6] Ensuring core models are cached in Google Drive...")
REQUIRED_MODELS = ["deepseek-r1:7b", "qwen2.5:7b", "deepseek-r1:1.5b"]

def already_pulled(model: str) -> bool:
    try:
        listed = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=15)
        return model.split(":")[0] in listed.stdout
    except Exception:
        return False

for model in REQUIRED_MODELS:
    if already_pulled(model):
        print(f"   ✅ {model} already cached, skipping pull.")
        continue
    print(f"   ⬇️  Pulling {model} ...")
    result = subprocess.run(["ollama", "pull", model])
    if result.returncode != 0:
        print(f"   ⚠️ Failed to pull {model} (exit code {result.returncode}) — continuing anyway.")

# [5/6] FASTAPI GATEWAY WITH CODE EXECUTION & WATCHDOG
print("🧠 [5/6] Initializing Enterprise Backend Gateway (FastAPI + GPU Executor)...")

from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import httpx, uvicorn

app = FastAPI(title="BuckBuck Neural Backend Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN, "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_INTERNAL_URL = "http://127.0.0.1:11434"

# One shared client for the lifetime of the app
http_client: httpx.AsyncClient | None = None

@app.on_event("startup")
async def _startup():
    global http_client
    http_client = httpx.AsyncClient(timeout=60.0)

@app.on_event("shutdown")
async def _shutdown():
    if http_client:
        await http_client.aclose()

# Thread pool so /api/exec runs blocking user code without freezing async event loop
exec_pool = ThreadPoolExecutor(max_workers=2)

def record_activity():
    global LAST_ACTIVITY_TIME
    LAST_ACTIVITY_TIME = time.time()

def require_api_key(authorization: str = Header(default=""), x_api_key: str = Header(default="")):
    """Validates Authorization: Bearer <key> or X-API-Key: <key>"""
    token = ""
    if authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()

    if not token or token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    record_activity()

class CodeExecutionRequest(BaseModel):
    code: str

def _run_user_code(code: str) -> dict:
    """Runs in a worker thread with execution timeout and chart interception."""
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
        "execution_time_seconds": round(time.time() - start_time, 3),
    }

# 1. PYTHON GPU CODE EXECUTION (NON-BLOCKING & TIMEOUT-BOUNDED)
@app.post("/api/exec")
async def execute_python_code(req: CodeExecutionRequest, _=Depends(require_api_key)):
    loop = __import__("asyncio").get_event_loop()
    future = loop.run_in_executor(exec_pool, _run_user_code, req.code)
    try:
        result = await __import__("asyncio").wait_for(future, timeout=EXEC_TIMEOUT_SECONDS)
    except __import__("asyncio").TimeoutError:
        return JSONResponse(
            {"success": False, "error": f"Execution exceeded {EXEC_TIMEOUT_SECONDS}s timeout"},
            status_code=408,
        )
    return result

# 2. SYSTEM HEALTH & GPU VRAM STATS
@app.get("/api/health")
async def get_system_health(_=Depends(require_api_key)):
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU"
        if gpu_available:
            free_b, _total = torch.cuda.mem_get_info(0)
            vram_free = f"{free_b / (1024**3):.1f} GB"
        else:
            vram_free = "N/A"
    except Exception as e:
        gpu_name, vram_free = "unknown (torch unavailable)", "N/A"

    elapsed_minutes = (time.time() - SERVER_START_TIME) / 60
    remaining_minutes = max(0, int(SESSION_LIMIT_MINUTES - elapsed_minutes))
    idle_minutes = int((time.time() - LAST_ACTIVITY_TIME) / 60)

    return {
        "status": "healthy",
        "gpu": gpu_name,
        "vram_free": vram_free,
        "session_remaining_minutes": remaining_minutes,
        "idle_minutes": idle_minutes,
    }

# 3. REVERSE PROXY FOR OLLAMA (/api/chat, /api/tags, etc.) — AUTH-GATED
ALLOWED_OLLAMA_ROUTES = ["api/chat", "api/generate", "api/tags", "api/show", "api/version"]

@app.api_route("/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def reverse_proxy_ollama(request: Request, path: str, _=Depends(require_api_key)):
    if path not in ALLOWED_OLLAMA_ROUTES and not path.startswith("api/chat"):
        raise HTTPException(status_code=403, detail=f"Access to '{path}' is blocked for security.")

    target_url = f"{OLLAMA_INTERNAL_URL}/{path}"
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("authorization", None)
    headers.pop("x-api-key", None)
    req_body = await request.body()
    try:
        ollama_req = http_client.build_request(
            method=request.method, url=target_url, headers=headers,
            params=request.query_params, content=req_body,
        )
        response = await http_client.send(ollama_req, stream=True)
        return StreamingResponse(
            response.aiter_raw(), status_code=response.status_code,
            headers=dict(response.headers), background=response.aclose,
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)

threading.Thread(
    target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning"),
    daemon=True,
).start()
time.sleep(2)

# [6/6] CLOUDFLARE TUNNEL WITH EVENT DETECTION
tunnel_cmd = ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000"]
tunnel_process = subprocess.Popen(tunnel_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

tunnel_url_found = threading.Event()
tunnel_url_holder = {"url": None}

def watch_tunnel_output():
    for line in tunnel_process.stderr:
        match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
        if match:
            tunnel_url_holder["url"] = match.group(0)
            tunnel_url_found.set()
            break

threading.Thread(target=watch_tunnel_output, daemon=True).start()

if tunnel_url_found.wait(timeout=30):
    url = tunnel_url_holder["url"]
    print("\n" + "=" * 68)
    print("🎉 BUCKBUCK ENTERPRISE GPU BACKEND IS ONLINE!")
    print(f"🔗 BACKEND URL : {url}")
    print(f"🔑 API KEY     : {API_KEY}")
    print(f"⏱️ SESSION LIMIT: {SESSION_LIMIT_MINUTES} mins | IDLE AUTO-STOP: {IDLE_TIMEOUT_MINUTES} mins")
    print("=" * 68)
    print("👉 Paste both the URL and API key into Settings (⚙️) on https://buckbuck.pages.dev")
    print("🛡️ Security Active: CORS locked, threadpool execution, destructive routes blocked.")
    print("=" * 68)
else:
    print("⚠️ Cloudflare tunnel did not report a URL within 30s. Check logs.")

# ------------------------------------------------------------------------------
# 🛡️ AUTOMATED GPU PRESERVATION WATCHDOG
# ------------------------------------------------------------------------------
def shutdown_backend():
    try:
        tunnel_process.kill()
    except Exception:
        pass
    exec_pool.shutdown(wait=False, cancel_futures=True)
    if AUTO_RELEASE_GPU:
        try:
            from google.colab import runtime
            runtime.unassign()
        except Exception:
            os._exit(0)
    else:
        os._exit(0)

def session_watchdog():
    while True:
        time.sleep(30)
        elapsed = (time.time() - SERVER_START_TIME) / 60
        idle = (time.time() - LAST_ACTIVITY_TIME) / 60

        if elapsed >= SESSION_LIMIT_MINUTES:
            print("\n" + "=" * 68)
            print(f"🛑 [WATCHDOG] Session limit of {SESSION_LIMIT_MINUTES} mins reached!")
            print("💾 Models safely preserved in Google Drive.")
            print("🛡️ Releasing Colab GPU to protect your daily compute quota...")
            print("=" * 68)
            shutdown_backend()
            break

        if AUTO_SHUTDOWN_ON_IDLE and idle >= IDLE_TIMEOUT_MINUTES:
            print("\n" + "=" * 68)
            print(f"💤 [WATCHDOG] Inactive for {IDLE_TIMEOUT_MINUTES} mins. Auto-stopping GPU...")
            print("💾 Models safely preserved in Google Drive.")
            print("=" * 68)
            shutdown_backend()
            break

threading.Thread(target=session_watchdog, daemon=True).start()

while True:
    time.sleep(60)
