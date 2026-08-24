# ==============================================================================
# 🚀 BUCKBUCK AI • FLAGSHIP WEB & CODING GPU BACKEND (LEAN v6.0)
# POWERED BY QWEN 2.5 CODER (7B) · 5-SECOND BOOT · 1-CLICK LAUNCH · ZERO QUOTA WASTE
# ==============================================================================

import os, sys, time, subprocess, threading, re, json, io, base64, secrets, traceback, asyncio, signal, tempfile, shutil, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

# ------------------------------------------------------------------------------
# ⚙️ CONFIGURATION
# ------------------------------------------------------------------------------
PRIMARY_MODEL = "qwen2.5-coder:7b"   # 🏆 #1 Open-Source Web & Coding Model
SESSION_LIMIT_MINUTES = 120         # Max session duration (mins)
AUTO_SHUTDOWN_ON_IDLE = True        # Auto-stop on inactivity
IDLE_TIMEOUT_MINUTES = 30           # Inactivity threshold
AUTO_RELEASE_GPU = True             # Release VM on session end to preserve quota
EXEC_TIMEOUT_SECONDS = 35           # Python code execution timeout
EXEC_MAX_CONCURRENT = 1             # Concurrent Python sandbox executions
EXEC_MAX_PER_MINUTE = 30            # Rate limit
ALLOWED_ORIGIN = "https://buckbuck.pages.dev"
FRONTEND_BASE_URL = "https://buckbuck.pages.dev"

WEEKLY_BUDGET_MINUTES = 12 * 60     # 12 hours/week quota protector
GPU_TRUE_IDLE_UTIL_PCT = 3          # Idle GPU util threshold

SERVER_START_TIME = time.time()
LAST_REAL_ACTIVITY_TIME = time.time()

# [1/4] MOUNT GOOGLE DRIVE & PERSISTENT KEY SETUP
DRIVE_AVAILABLE = False
DRIVE_ROOT = '/content/drive/MyDrive'
DRIVE_MODELS = f"{DRIVE_ROOT}/ollama_models"
DRIVE_BIN = f"{DRIVE_ROOT}/buckbuck_bin"
DRIVE_KEY_PATH = f"{DRIVE_ROOT}/buckbuck_api_key.txt"
LOCAL_MODELS = '/root/.ollama/models'

print("📁 [1/4] Connecting Google Drive Storage...")
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    os.makedirs(DRIVE_MODELS, exist_ok=True)
    os.makedirs(DRIVE_BIN, exist_ok=True)
    os.makedirs(LOCAL_MODELS, exist_ok=True)
    
    # Persistent API Key: reused across all sessions
    if os.path.exists(DRIVE_KEY_PATH):
        with open(DRIVE_KEY_PATH, 'r') as f:
            API_KEY = f.read().strip()
    else:
        API_KEY = secrets.token_urlsafe(32)
        with open(DRIVE_KEY_PATH, 'w') as f:
            f.write(API_KEY)

    # Sync model cache from Drive to local NVMe SSD
    if os.path.exists(os.path.join(DRIVE_MODELS, "manifests")):
        shutil.copytree(DRIVE_MODELS, LOCAL_MODELS, dirs_exist_ok=True)
    
    os.environ['OLLAMA_MODELS'] = LOCAL_MODELS
    DRIVE_AVAILABLE = True
except Exception:
    API_KEY = secrets.token_urlsafe(32)

USAGE_LOG_PATH = f"{DRIVE_ROOT}/buckbuck_usage_log.json" if DRIVE_AVAILABLE else "/content/buckbuck_usage_log.json"

def load_usage_log():
    try:
        with open(USAGE_LOG_PATH, 'r') as f: return json.load(f)
    except Exception: return []

def save_usage_log(entries):
    try:
        with open(USAGE_LOG_PATH, 'w') as f: json.dump(entries, f)
    except Exception: pass

def _safe_parse(e):
    try: return datetime.fromisoformat(e["end"])
    except Exception: return None

def minutes_used_last_7_days(entries):
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    return sum(e["minutes"] for e in entries if _safe_parse(e) and _safe_parse(e) >= cutoff)

_usage_entries = load_usage_log()
_minutes_used_this_week = minutes_used_last_7_days(_usage_entries)
_minutes_remaining_this_week = max(0, WEEKLY_BUDGET_MINUTES - _minutes_used_this_week)
EFFECTIVE_SESSION_LIMIT_MINUTES = int(min(SESSION_LIMIT_MINUTES, _minutes_remaining_this_week))

# [2/4] RESTORE ENGINE & FAST PIP PACKAGES
print("⚡ [2/4] Setting up Ollama Engine & Cloudflare Tunnel...")

os.environ['PATH'] = f"/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:{os.environ.get('PATH', '')}"
os.environ['LD_LIBRARY_PATH'] = f"/usr/local/lib/ollama:/usr/local/cuda/lib64:/usr/lib64-nvidia:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OLLAMA_ORIGINS'] = '*'
os.environ['OLLAMA_HOST'] = '127.0.0.1:11434'
os.environ['OLLAMA_FLASH_ATTENTION'] = '1'
os.environ['OLLAMA_NUM_PARALLEL'] = '1'
os.environ['OLLAMA_KEEP_ALIVE'] = '24h'
os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'

def restore_binaries_from_drive() -> bool:
    if not DRIVE_AVAILABLE: return False
    ollama_drive = os.path.join(DRIVE_BIN, "ollama")
    cloudflared_drive = os.path.join(DRIVE_BIN, "cloudflared")
    if os.path.exists(ollama_drive) and os.path.exists(cloudflared_drive):
        try:
            shutil.copy(ollama_drive, "/usr/local/bin/ollama")
            shutil.copy(cloudflared_drive, "/usr/local/bin/cloudflared")
            os.system("chmod +x /usr/local/bin/ollama /usr/local/bin/cloudflared >/dev/null 2>&1")
            lib_d = os.path.join(DRIVE_BIN, "ollama_lib")
            if os.path.exists(lib_d):
                shutil.copytree(lib_d, "/usr/local/lib/ollama", dirs_exist_ok=True)
            return True
        except Exception: pass
    return False

def save_binaries_to_drive():
    if not DRIVE_AVAILABLE: return
    try:
        if os.path.exists("/usr/local/bin/ollama"):
            shutil.copy("/usr/local/bin/ollama", os.path.join(DRIVE_BIN, "ollama"))
        cf = shutil.which("cloudflared") or "/usr/local/bin/cloudflared"
        if os.path.exists(cf):
            shutil.copy(cf, os.path.join(DRIVE_BIN, "cloudflared"))
        if os.path.exists("/usr/local/lib/ollama"):
            shutil.copytree("/usr/local/lib/ollama", os.path.join(DRIVE_BIN, "ollama_lib"), dirs_exist_ok=True)
    except Exception: pass

def install_binaries_from_web():
    os.system("sudo apt-get update -qq >/dev/null 2>&1 && sudo apt-get install -y -q zstd >/dev/null 2>&1")
    os.system("curl -s -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cf.deb && sudo dpkg -i /tmp/cf.deb >/dev/null 2>&1")
    rc = os.system("curl -fsSL https://ollama.com/install.sh | sh >/dev/null 2>&1")
    if rc != 0 or not os.path.exists("/usr/local/bin/ollama"):
        asset = "ollama-linux-amd64.tar.zst"
        os.system(f"curl -sS -L 'https://github.com/ollama/ollama/releases/latest/download/{asset}' -o '/tmp/{asset}'")
        os.system(f"tar --zstd -xf '/tmp/{asset}' -C /usr/local >/dev/null 2>&1 || (zstd -d -f '/tmp/{asset}' -o /tmp/o.tar && tar -xf /tmp/o.tar -C /usr/local >/dev/null 2>&1)")
        os.system("chmod +x /usr/local/bin/ollama >/dev/null 2>&1")
    save_binaries_to_drive()

if not restore_binaries_from_drive():
    install_binaries_from_web()

OLLAMA_BIN = shutil.which("ollama") or "/usr/local/bin/ollama"

# Fast pip install
os.system("pip install -q fastapi uvicorn httpx pydantic nest_asyncio >/dev/null 2>&1")

# Background installer for Whisper audio and extra ML packages
def _bg_install_ml():
    os.system("pip install -q openai-whisper torchaudio seaborn sympy >/dev/null 2>&1")
threading.Thread(target=_bg_install_ml, daemon=True).start()

# Free port 8000
os.system("fuser -k 8000/tcp >/dev/null 2>&1 || true")

# Start Ollama Engine
subprocess.Popen([OLLAMA_BIN, "serve"], env=dict(os.environ), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(2)

# [3/4] VERIFY & WARM UP FLAGSHIP MODEL (Qwen 2.5 Coder 7B)
print(f"📥 [3/4] Validating Flagship Model ({PRIMARY_MODEL})...")

def already_pulled(model: str) -> bool:
    try:
        listed = subprocess.run([OLLAMA_BIN, "list"], capture_output=True, text=True, timeout=10)
        return any(parts and (parts[0] == model or parts[0] == f"{model}:latest" or model in parts[0])
                   for line in listed.stdout.splitlines() if (parts := line.split()))
    except Exception:
        return False

def verify_and_pull_primary():
    need_pull = not already_pulled(PRIMARY_MODEL)
    if not need_pull:
        # Test integrity with a 1-token dry run
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/generate",
                data=json.dumps({"model": PRIMARY_MODEL, "prompt": "1", "options": {"num_predict": 1}}).encode(),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                pass
            print(f"   ✅ {PRIMARY_MODEL} (verified in Google Drive)")
            return
        except Exception as e:
            print(f"   ⚠️ Cache integrity check failed ({e}). Re-pulling fresh copy...")
            subprocess.run([OLLAMA_BIN, "rm", PRIMARY_MODEL], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            need_pull = True

    if need_pull:
        print(f"   ⬇️ Downloading {PRIMARY_MODEL} into Google Drive cache (one-time setup)...")
        res = subprocess.run([OLLAMA_BIN, "pull", PRIMARY_MODEL])
        if res.returncode == 0 and DRIVE_AVAILABLE:
            shutil.copytree(LOCAL_MODELS, DRIVE_MODELS, dirs_exist_ok=True)
            print(f"   💾 Saved {PRIMARY_MODEL} permanently into Google Drive!")

verify_and_pull_primary()

# Lock flagship model into GPU VRAM with 24h keep-alive
print(f"🔥 Locking {PRIMARY_MODEL} into GPU VRAM for instant 0.1s response...")
try:
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps({"model": PRIMARY_MODEL, "prompt": "hi", "keep_alive": "24h"}).encode(),
        headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req, timeout=60)
    print(f"   ⚡ {PRIMARY_MODEL} is resident in Tesla T4 VRAM.")
except Exception:
    pass

# [4/4] FASTAPI SERVER SETUP & 1-CLICK LAUNCH
import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI, Request, HTTPException, Depends, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import httpx, uvicorn

OLLAMA_INTERNAL_URL = "http://127.0.0.1:11434"
http_client: httpx.AsyncClient | None = None
whisper_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=300.0)
    yield
    if http_client:
        await http_client.aclose()

app = FastAPI(title="BuckBuck Neural Gateway", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

exec_semaphore = asyncio.Semaphore(EXEC_MAX_CONCURRENT)
exec_pool = ThreadPoolExecutor(max_workers=EXEC_MAX_CONCURRENT + 1)
_rate_lock = threading.Lock()
_rate_window_start = time.time()
_rate_count = 0

def check_rate_limit():
    global _rate_window_start, _rate_count
    with _rate_lock:
        now = time.time()
        if now - _rate_window_start >= 60:
            _rate_window_start = now
            _rate_count = 0
        _rate_count += 1
        if _rate_count > EXEC_MAX_PER_MINUTE:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

def record_real_activity():
    global LAST_REAL_ACTIVITY_TIME
    LAST_REAL_ACTIVITY_TIME = time.time()

def require_api_key(authorization: str = Header(default=""), x_api_key: str = Header(default="")):
    supplied = authorization[7:].strip() if authorization.startswith("Bearer ") else x_api_key.strip()
    if not secrets.compare_digest(supplied, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")

class CodeExecutionRequest(BaseModel):
    code: str

_RUNNER_TEMPLATE = r"""
import sys, io, json, base64, traceback
def _main():
    code = sys.argv[1] if len(sys.argv) > 1 else ""
    exec_scope = {"__name__": "__main__"}
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
        result = {"success": True, "stdout": redirected_stdout.getvalue(), "stderr": redirected_stderr.getvalue(), "error": "", "plots": plots_base64}
    except Exception as e:
        result = {"success": False, "stdout": redirected_stdout.getvalue(), "stderr": traceback.format_exc(), "error": str(e), "plots": plots_base64}
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
    sys.stdout.write("\n__RESULT_JSON__" + json.dumps(result))
_main()
"""

def _run_code_in_subprocess(code: str, timeout: float) -> dict:
    start = time.time()
    with tempfile.NamedTemporaryFile("w", suffix="_runner.py", delete=False) as f:
        f.write(_RUNNER_TEMPLATE)
        runner_path = f.name
    proc = subprocess.Popen([sys.executable, runner_path, code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception: proc.kill()
        proc.wait(timeout=5)
        return {"success": False, "stdout": "", "stderr": "", "error": f"Timeout of {timeout}s exceeded", "plots": [], "execution_time_seconds": round(time.time()-start, 3)}
    finally:
        try: os.unlink(runner_path)
        except Exception: pass

    marker = "__RESULT_JSON__"
    if marker in stdout:
        try: result = json.loads(stdout.split(marker, 1)[1])
        except Exception: result = {"success": False, "stdout": stdout, "stderr": stderr, "error": "Parse error", "plots": []}
    else:
        result = {"success": False, "stdout": stdout, "stderr": stderr, "error": "No output", "plots": []}
    result["execution_time_seconds"] = round(time.time() - start, 3)
    return result

@app.post("/api/exec")
async def execute_python_code(req: CodeExecutionRequest, _=Depends(require_api_key)):
    check_rate_limit()
    record_real_activity()
    async with exec_semaphore:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(exec_pool, _run_code_in_subprocess, req.code, EXEC_TIMEOUT_SECONDS)
    return result

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...), _=Depends(require_api_key)):
    global whisper_model
    record_real_activity()
    try:
        import whisper
        if whisper_model is None:
            whisper_model = whisper.load_model("base", device="cuda" if os.path.exists('/dev/nvidia0') else "cpu")
        with tempfile.NamedTemporaryFile("wb", suffix=".webm", delete=False) as f:
            f.write(await file.read())
            tmp_p = f.name
        tr = whisper_model.transcribe(tmp_p)
        try: os.unlink(tmp_p)
        except Exception: pass
        return {"success": True, "text": tr.get("text", "").strip()}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/health")
async def get_system_health(_=Depends(require_api_key)):
    gpu_name, vram_free = "Tesla T4", "14.5 GB"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            free_b, _ = torch.cuda.mem_get_info(0)
            vram_free = f"{free_b / (1024**3):.1f} GB"
    except Exception: pass

    elapsed = (time.time() - SERVER_START_TIME) / 60
    return {
        "status": "healthy",
        "gpu": gpu_name,
        "vram_free": vram_free,
        "model": PRIMARY_MODEL,
        "features": ["qwen2.5_coder_7b", "whisper_audio_gpu", "1click_autoconnect", "sandboxed_exec"],
        "session_remaining_minutes": max(0, int(EFFECTIVE_SESSION_LIMIT_MINUTES - elapsed)),
        "idle_minutes": int((time.time() - LAST_REAL_ACTIVITY_TIME) / 60),
    }

ALLOWED_OLLAMA_ROUTES = ["api/chat", "api/generate", "api/tags", "api/show", "api/version"]

@app.api_route("/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def reverse_proxy_ollama(request: Request, path: str, _=Depends(require_api_key)):
    if path not in ALLOWED_OLLAMA_ROUTES and not path.startswith("api/chat"):
        raise HTTPException(status_code=403, detail="Access blocked")
    record_real_activity()
    target_url = f"{OLLAMA_INTERNAL_URL}/{path}"
    headers = dict(request.headers)
    headers.pop("host", None); headers.pop("authorization", None); headers.pop("x-api-key", None)
    req_body = await request.body()

    # Route requests to flagship model if requested model is alias
    if request.method == "POST" and (path == "api/chat" or path == "api/generate"):
        try:
            body_json = json.loads(req_body.decode())
            req_model = body_json.get("model", "")
            if not req_model or req_model != PRIMARY_MODEL:
                if not already_pulled(req_model):
                    body_json["model"] = PRIMARY_MODEL
                    req_body = json.dumps(body_json).encode()
        except Exception: pass

    try:
        ollama_req = http_client.build_request(method=request.method, url=target_url, headers=headers, params=request.query_params, content=req_body)
        response = await http_client.send(ollama_req, stream=True)
        resp_headers = dict(response.headers)
        resp_headers["X-Accel-Buffering"] = "no"; resp_headers["Cache-Control"] = "no-cache"
        return StreamingResponse(response.aiter_raw(), status_code=response.status_code, headers=resp_headers, media_type="application/x-ndjson", background=response.aclose)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)

# Cloudflare Tunnel & 1-Click Launch
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
    one_click_url = f"{FRONTEND_BASE_URL}/?colab={url}&key={API_KEY}"
    
    print("\n" + "═" * 70)
    print(f"🎉 BUCKBUCK FLAGSHIP GPU BACKEND IS ONLINE ({PRIMARY_MODEL})!")
    print("═" * 70)
    print(f"👉 1-CLICK LAUNCH : {one_click_url}")
    print("═" * 70)
    print("💡 Click the link above to open BuckBuck AI with GPU connected automatically.")
    print("═" * 70)
else:
    print("⚠️ Cloudflare tunnel timeout. Please rerun cell.")

# Watchdog thread
def session_watchdog():
    while True:
        time.sleep(30)
        elapsed = (time.time() - SERVER_START_TIME) / 60
        idle = (time.time() - LAST_REAL_ACTIVITY_TIME) / 60
        if elapsed >= EFFECTIVE_SESSION_LIMIT_MINUTES or (AUTO_SHUTDOWN_ON_IDLE and idle >= IDLE_TIMEOUT_MINUTES):
            try: tunnel_process.kill()
            except Exception: pass
            if AUTO_RELEASE_GPU:
                try:
                    from google.colab import runtime
                    runtime.unassign()
                except Exception: os._exit(0)
            else: os._exit(0)
            break

threading.Thread(target=session_watchdog, daemon=True).start()

# Main server run loop
config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="error")
server = uvicorn.Server(config)
loop = asyncio.get_event_loop()
loop.run_until_complete(server.serve())
