# ==============================================================================
# 🚀 BUCKBUCK AI • ENTERPRISE COLAB BACKEND (HARDENED v4)
# VISION AI · DATA SCIENCE SUITE · AUTO-PIP · WHISPER GPU · HARDENED SANDBOX
# ==============================================================================

# ------------------------------------------------------------------------------
# ⚙️ CONFIG — customize as needed
# ------------------------------------------------------------------------------
SESSION_LIMIT_MINUTES = 120        # Hard cap for THIS session
AUTO_SHUTDOWN_ON_IDLE = True       # Auto-stop if inactive
IDLE_TIMEOUT_MINUTES = 20          # Shut down after N mins of zero real activity
AUTO_RELEASE_GPU = True            # Release Colab VM on expiry to preserve daily quota
EXEC_TIMEOUT_SECONDS = 35          # Hard cap on any single /api/exec call
EXEC_MAX_CONCURRENT = 1            # Max concurrent /api/exec calls
EXEC_MAX_PER_MINUTE = 25           # Rate limit per API key
ALLOWED_ORIGIN = "https://buckbuck.pages.dev"  # Lock CORS to your frontend

# --- Colab 7-day quota protection --------------------------------------------
WEEKLY_BUDGET_MINUTES = 12 * 60     # 12 hours/week budget
USAGE_LOG_PATH_DRIVE = '/content/drive/MyDrive/buckbuck_usage_log.json'
USAGE_LOG_PATH_LOCAL = '/content/buckbuck_usage_log.json'
GPU_TRUE_IDLE_UTIL_PCT = 3          # nvidia-smi util% below which GPU is idle
# ------------------------------------------------------------------------------

import os, sys, time, subprocess, threading, re, json, io, base64, secrets, traceback, asyncio, signal, tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

SERVER_START_TIME = time.time()
LAST_REAL_ACTIVITY_TIME = time.time()

# 🔑 API KEY
API_KEY = secrets.token_urlsafe(32)

# [1/6] MOUNT GOOGLE DRIVE
DRIVE_AVAILABLE = False
try:
    from google.colab import drive
    print("📁 [1/6] Mounting Google Drive for permanent model cache...")
    drive.mount('/content/drive')
    os.makedirs('/content/drive/MyDrive/ollama_models', exist_ok=True)
    os.environ['OLLAMA_MODELS'] = '/content/drive/MyDrive/ollama_models'
    DRIVE_AVAILABLE = True
except Exception:
    print("⚠️ Running outside Colab or Drive skipped. Using local disk.")

USAGE_LOG_PATH = USAGE_LOG_PATH_DRIVE if DRIVE_AVAILABLE else USAGE_LOG_PATH_LOCAL

# 📊 WEEKLY USAGE BUDGET
def load_usage_log():
    try:
        with open(USAGE_LOG_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return []

def save_usage_log(entries):
    try:
        with open(USAGE_LOG_PATH, 'w') as f:
            json.dump(entries, f)
    except Exception as e:
        print(f"⚠️ Could not write usage log: {e}")

def _safe_parse(e):
    try:
        return datetime.fromisoformat(e["end"])
    except Exception:
        return None

def minutes_used_last_7_days(entries):
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    total = 0.0
    for e in entries:
        ts = _safe_parse(e)
        if ts and ts >= cutoff:
            total += e["minutes"]
    return total

_usage_entries = load_usage_log()
_minutes_used_this_week = minutes_used_last_7_days(_usage_entries)
_minutes_remaining_this_week = max(0, WEEKLY_BUDGET_MINUTES - _minutes_used_this_week)

print(f"📊 Colab quota guard: {_minutes_used_this_week:.0f}/{WEEKLY_BUDGET_MINUTES} mins used in last 7 days "
      f"({_minutes_remaining_this_week:.0f} mins remaining).")

if _minutes_remaining_this_week <= 0:
    print("\n" + "=" * 65)
    print("🛑 WEEKLY BUDGET EXHAUSTED — refusing to start a new GPU session.")
    print("=" * 65)
    raise SystemExit(1)

EFFECTIVE_SESSION_LIMIT_MINUTES = int(min(SESSION_LIMIT_MINUTES, _minutes_remaining_this_week))

# [2/6] CONFIGURE HIGH-PERFORMANCE GPU ENVIRONMENT
os.environ['PATH'] = f"/usr/local/bin:/usr/bin:/bin:{os.environ.get('PATH', '')}"
os.environ['OLLAMA_ORIGINS'] = '*'
os.environ['OLLAMA_HOST'] = '127.0.0.1:11434'
os.environ['OLLAMA_FLASH_ATTENTION'] = '1'
os.environ['OLLAMA_NUM_PARALLEL'] = '2'
os.environ['OLLAMA_KEEP_ALIVE'] = '24h'
os.environ['OLLAMA_MAX_LOADED_MODELS'] = '2'

import shutil

def get_ollama_path() -> str:
    for candidate in ["/usr/local/bin/ollama", "/usr/bin/ollama", "/bin/ollama"]:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which("ollama")
    if found:
        return found
    print("⏳ Installing Ollama binary...")
    os.system("curl -fsSL https://ollama.com/install.sh | sh")
    for candidate in ["/usr/local/bin/ollama", "/usr/bin/ollama", "/bin/ollama"]:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which("ollama")
    if found:
        return found
    # Fallback to direct tar extraction
    print("⚠️ Standard script missed, extracting direct linux-amd64 binary...")
    os.system("curl -s -L https://ollama.com/download/ollama-linux-amd64.tgz -o /tmp/ollama.tgz && tar -C /usr -xzf /tmp/ollama.tgz > /dev/null 2>&1")
    return "/usr/bin/ollama" if os.path.exists("/usr/bin/ollama") else "/usr/local/bin/ollama"

# [3/6] INSTALL SYSTEM DEPENDENCIES & DATA SCIENCE/VISION/VOICE PACKAGES
print("⏳ [2/6] Installing Ollama, Cloudflare Tunnel, ML/Vision & Audio Suite...")
OLLAMA_BIN = get_ollama_path()
os.system(
    "curl -s -L https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-linux-amd64.deb -o cloudflared.deb && sudo dpkg -i cloudflared.deb > /dev/null 2>&1"
)
os.system(
    "pip install -q fastapi uvicorn httpx pydantic matplotlib numpy pandas scipy "
    "scikit-learn seaborn sympy pillow torch torchaudio openai-whisper > /dev/null 2>&1"
)

# [4/6] START OLLAMA ENGINE IN BACKGROUND
print(f"⚡ [3/6] Starting Ollama Engine ({OLLAMA_BIN}) with FlashAttention & GPU Locking...")
subprocess.Popen([OLLAMA_BIN, "serve"], env=dict(os.environ))
time.sleep(3)

print("📥 [4/6] Ensuring core & Vision models are cached in Google Drive...")
REQUIRED_MODELS = ["deepseek-r1:7b", "qwen2.5:7b", "llava:7b", "deepseek-r1:1.5b"]

def already_pulled(model: str) -> bool:
    try:
        listed = subprocess.run([OLLAMA_BIN, "list"], capture_output=True, text=True, timeout=15)
        return model.split(":")[0] in listed.stdout
    except Exception:
        return False

for model in REQUIRED_MODELS:
    if already_pulled(model):
        print(f"   ✅ {model} already cached, skipping pull.")
        continue
    print(f"   ⬇️  Pulling {model} ...")
    result = subprocess.run([OLLAMA_BIN, "pull", model])
    if result.returncode != 0:
        print(f"   ⚠️ Failed to pull {model} (exit code {result.returncode}) — continuing anyway.")

# [5/6] FASTAPI GATEWAY WITH MULTIMODAL, VOICE, AND AUTO-PIP EXECUTION
print("🧠 [5/6] Initializing Enterprise Backend Gateway (FastAPI + Vision + GPU Audio)...")

from fastapi import FastAPI, Request, HTTPException, Depends, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import httpx, uvicorn

app = FastAPI(title="BuckBuck Neural Multimodal Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN, "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_INTERNAL_URL = "http://127.0.0.1:11434"
http_client: httpx.AsyncClient | None = None
whisper_model = None

@app.on_event("startup")
async def _startup():
    global http_client
    http_client = httpx.AsyncClient(timeout=60.0)

@app.on_event("shutdown")
async def _shutdown():
    if http_client:
        await http_client.aclose()

exec_semaphore = asyncio.Semaphore(EXEC_MAX_CONCURRENT)
exec_pool = ThreadPoolExecutor(max_workers=EXEC_MAX_CONCURRENT + 1)

# Rate limiter
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
            raise HTTPException(status_code=429, detail="Rate limit exceeded, slow down")

# Audit log
AUDIT_LOG_PATH = (
    '/content/drive/MyDrive/buckbuck_audit_log.jsonl' if DRIVE_AVAILABLE
    else '/content/buckbuck_audit_log.jsonl'
)

def audit_log(event: dict):
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    try:
        with open(AUDIT_LOG_PATH, 'a') as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        pass

def record_real_activity():
    global LAST_REAL_ACTIVITY_TIME
    LAST_REAL_ACTIVITY_TIME = time.time()

def require_api_key(authorization: str = Header(default=""), x_api_key: str = Header(default="")):
    supplied = ""
    if authorization.startswith("Bearer "):
        supplied = authorization[7:].strip()
    elif x_api_key:
        supplied = x_api_key.strip()

    if not secrets.compare_digest(supplied, API_KEY):
        audit_log({"event": "auth_failed"})
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

class CodeExecutionRequest(BaseModel):
    code: str

# ------------------------------------------------------------------------------
# 🧪 PROCESS-SANDBOXED EXECUTION WITH DYNAMIC PIP AUTO-INSTALLATION
# ------------------------------------------------------------------------------
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

        result = {
            "success": True,
            "stdout": redirected_stdout.getvalue(),
            "stderr": redirected_stderr.getvalue(),
            "error": "",
            "plots": plots_base64,
        }
    except Exception as e:
        result = {
            "success": False,
            "stdout": redirected_stdout.getvalue(),
            "stderr": traceback.format_exc(),
            "error": str(e),
            "plots": plots_base64,
        }
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
    sys.stdout.write("\n__RESULT_JSON__" + json.dumps(result))

_main()
"""

def _run_code_in_subprocess(code: str, timeout: float, allow_pip_retry: bool = True) -> dict:
    start = time.time()
    with tempfile.NamedTemporaryFile("w", suffix="_runner.py", delete=False) as f:
        f.write(_RUNNER_TEMPLATE)
        runner_path = f.name

    proc = subprocess.Popen(
        [sys.executable, runner_path, code],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=5)
        try:
            os.unlink(runner_path)
        except Exception:
            pass
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "error": f"Execution exceeded {timeout}s timeout and was killed.",
            "plots": [],
            "execution_time_seconds": round(time.time() - start, 3),
        }
    finally:
        try:
            os.unlink(runner_path)
        except Exception:
            pass

    marker = "__RESULT_JSON__"
    if marker in stdout:
        payload = stdout.split(marker, 1)[1]
        try:
            result = json.loads(payload)
        except Exception:
            result = {"success": False, "stdout": stdout, "stderr": stderr, "error": "Failed to parse result", "plots": []}
    else:
        result = {"success": False, "stdout": stdout, "stderr": stderr, "error": "No output produced", "plots": []}

    # DYNAMIC PIP AUTO-INSTALLER: If missing module, auto-install and retry!
    if not result.get("success") and allow_pip_retry:
        err_text = result.get("stderr", "") + result.get("error", "")
        missing_match = re.search(r"No module named '([a-zA-Z0-9_-]+)'", err_text)
        if missing_match:
            pkg = missing_match.group(1)
            print(f"📦 [AUTO-PIP] Detected missing package '{pkg}', auto-installing on GPU...")
            pip_res = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], capture_output=True, timeout=40)
            if pip_res.returncode == 0:
                print(f"✅ [AUTO-PIP] Successfully installed '{pkg}'! Re-running code...")
                return _run_code_in_subprocess(code, timeout, allow_pip_retry=False)

    result["execution_time_seconds"] = round(time.time() - start, 3)
    return result

@app.post("/api/exec")
async def execute_python_code(req: CodeExecutionRequest, _=Depends(require_api_key)):
    check_rate_limit()
    record_real_activity()
    async with exec_semaphore:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            exec_pool, _run_code_in_subprocess, req.code, EXEC_TIMEOUT_SECONDS
        )
    audit_log({
        "event": "exec",
        "success": result.get("success"),
        "execution_time_seconds": result.get("execution_time_seconds"),
        "code_preview": req.code[:200],
        "code_length": len(req.code),
    })
    return result

# 🎙️ GPU WHISPER AUDIO TRANSCRIPTION ENDPOINT
@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...), _=Depends(require_api_key)):
    global whisper_model
    record_real_activity()
    try:
        import whisper
        if whisper_model is None:
            whisper_model = whisper.load_model("base", device="cuda" if os.path.exists('/dev/nvidia0') else "cpu")
            
        with tempfile.NamedTemporaryFile("wb", suffix=".webm", delete=False) as f:
            content = await file.read()
            f.write(content)
            tmp_audio_path = f.name
            
        transcription = whisper_model.transcribe(tmp_audio_path)
        try:
            os.unlink(tmp_audio_path)
        except Exception:
            pass
            
        return {"success": True, "text": transcription.get("text", "").strip()}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

# 2. SYSTEM HEALTH & GPU VRAM STATS
try:
    import torch
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

@app.get("/api/health")
async def get_system_health(_=Depends(require_api_key)):
    gpu_name, vram_free = "unknown", "N/A"
    if _TORCH_OK:
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                free_b, _total = torch.cuda.mem_get_info(0)
                vram_free = f"{free_b / (1024**3):.1f} GB"
            else:
                gpu_name = "CPU"
        except Exception:
            pass

    elapsed_minutes = (time.time() - SERVER_START_TIME) / 60
    remaining_minutes = max(0, int(EFFECTIVE_SESSION_LIMIT_MINUTES - elapsed_minutes))
    idle_minutes = int((time.time() - LAST_REAL_ACTIVITY_TIME) / 60)

    return {
        "status": "healthy",
        "gpu": gpu_name,
        "vram_free": vram_free,
        "features": ["vision_llava", "whisper_audio_gpu", "auto_pip_install", "pandas_seaborn_ml"],
        "session_remaining_minutes": remaining_minutes,
        "idle_minutes": idle_minutes,
        "weekly_budget_remaining_minutes": max(
            0, int(_minutes_remaining_this_week - elapsed_minutes)
        ),
    }

# 3. REVERSE PROXY FOR OLLAMA (/api/chat, /api/tags, etc.) — AUTH-GATED
ALLOWED_OLLAMA_ROUTES = ["api/chat", "api/generate", "api/tags", "api/show", "api/version"]

@app.api_route("/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def reverse_proxy_ollama(request: Request, path: str, _=Depends(require_api_key)):
    if path not in ALLOWED_OLLAMA_ROUTES and not path.startswith("api/chat"):
        raise HTTPException(status_code=403, detail=f"Access to '{path}' is blocked for security.")

    record_real_activity()
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

# [6/6] CLOUDFLARE TUNNEL
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
    print("🎉 BUCKBUCK ENTERPRISE GPU BACKEND IS ONLINE (v4 MULTIMODAL)!")
    print(f"🔗 BACKEND URL   : {url}")
    print(f"🔑 API KEY       : {API_KEY}")
    print(f"⏱️ SESSION LIMIT  : {EFFECTIVE_SESSION_LIMIT_MINUTES} mins (weekly-budget adjusted)")
    print(f"💤 IDLE AUTO-STOP: {IDLE_TIMEOUT_MINUTES} mins of real inactivity")
    print(f"📊 WEEKLY BUDGET : {_minutes_remaining_this_week:.0f} / {WEEKLY_BUDGET_MINUTES} mins remaining")
    print("=" * 68)
    print("👉 Paste both the URL and API key into Settings (⚙️) on https://buckbuck.pages.dev")
    print("⚡ New: Vision AI (LLaVA), Auto-Pip, Whisper GPU Voice & ML pre-installed!")
    print("=" * 68)
else:
    print("⚠️ Cloudflare tunnel did not report a URL within 30s. Check logs.")

# ------------------------------------------------------------------------------
# 🛡️ AUTOMATED GPU PRESERVATION WATCHDOG
# ------------------------------------------------------------------------------
def _record_this_session_usage():
    elapsed_min = (time.time() - SERVER_START_TIME) / 60
    entries = load_usage_log()
    entries.append({
        "end": datetime.now(timezone.utc).isoformat(),
        "minutes": round(elapsed_min, 2),
    })
    cutoff = datetime.now(timezone.utc) - timedelta(days=14)
    entries = [e for e in entries if _safe_parse(e) and _safe_parse(e) >= cutoff]
    save_usage_log(entries)

def shutdown_backend():
    _record_this_session_usage()
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

def gpu_truly_idle() -> bool:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        util = int(out.stdout.strip().splitlines()[0])
        return util < GPU_TRUE_IDLE_UTIL_PCT
    except Exception:
        return True

def session_watchdog():
    while True:
        time.sleep(30)
        elapsed = (time.time() - SERVER_START_TIME) / 60
        idle = (time.time() - LAST_REAL_ACTIVITY_TIME) / 60

        if elapsed >= EFFECTIVE_SESSION_LIMIT_MINUTES:
            print("\n" + "=" * 68)
            print(f"🛑 [WATCHDOG] Session cap of {EFFECTIVE_SESSION_LIMIT_MINUTES} mins reached.")
            print("💾 Models safely preserved in Google Drive.")
            print("🛡️ Releasing Colab GPU to protect your weekly compute quota...")
            print("=" * 68)
            shutdown_backend()
            break

        if AUTO_SHUTDOWN_ON_IDLE and idle >= IDLE_TIMEOUT_MINUTES and gpu_truly_idle():
            print("\n" + "=" * 68)
            print(f"💤 [WATCHDOG] No real requests for {IDLE_TIMEOUT_MINUTES} mins and GPU is idle. Auto-stopping...")
            print("💾 Models safely preserved in Google Drive.")
            print("=" * 68)
            shutdown_backend()
            break

threading.Thread(target=session_watchdog, daemon=True).start()

while True:
    time.sleep(60)
