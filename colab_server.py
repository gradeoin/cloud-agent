# ==============================================================================
# 🚀 BULLETPROOF GOOGLE COLAB BACKEND (ZERO 403 ERRORS)
# Uses Cloudflare Tunnel: 100% Free, Zero Tokens Needed, Zero 403 Issues!
# ==============================================================================

from google.colab import drive
import os, subprocess, time, re, threading

print("📁 [1/4] Mounting Google Drive (for permanent model storage)...")
drive.mount('/content/drive')

# CRITICAL: Allow all website origins and bind to all network interfaces
os.environ['OLLAMA_MODELS'] = '/content/drive/MyDrive/ollama_models'
os.environ['OLLAMA_ORIGINS'] = '*'
os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'

print("⏳ [2/4] Installing Ollama & Cloudflare Tunnel...")
!sudo apt-get update -qq && sudo apt-get install -y zstd pciutils > /dev/null
!curl -fsSL https://ollama.com/install.sh | sh
!curl -s -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb && sudo dpkg -i cloudflared.deb > /dev/null

print("⚡ [3/4] Starting Ollama Engine...")
subprocess.Popen(["ollama", "serve"], env=dict(os.environ))
time.sleep(4)

# Load model from Google Drive
MODEL = "deepseek-r1:7b"
print(f"📥 Loading {MODEL} from Google Drive...")
!ollama pull {MODEL}

print("🌐 [4/4] Starting Cloudflare Tunnel...")
tunnel_process = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://127.0.0.1:11434"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

def extract_tunnel_url():
    for line in tunnel_process.stderr:
        match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
        if match:
            url = match.group(0)
            print("\n" + "="*60)
            print("🎉 GOOGLE COLAB GPU IS READY AND ONLINE!")
            print(f"🔗 YOUR BACKEND URL: {url}")
            print("="*60)
            print("👉 Copy this URL and paste it into Settings (⚙️) on your website.")
            break

threading.Thread(target=extract_tunnel_url, daemon=True).start()

# Keep execution active
while True:
    time.sleep(60)
