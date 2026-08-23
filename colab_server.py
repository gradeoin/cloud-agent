# ==============================================================================
# 🚀 GOOGLE COLAB BACKEND SCRIPT
# 1. Select: Runtime > Change runtime type > T4 GPU
# 2. Paste this complete code block into Colab and click 'Run'
# ==============================================================================

from google.colab import drive
import os, subprocess, time
from pyngrok import ngrok

print("📁 [1/4] Mounting Google Drive (for permanent model storage)...")
drive.mount('/content/drive')

# Tell Ollama to save all downloaded models inside your Google Drive
os.environ['OLLAMA_MODELS'] = '/content/drive/MyDrive/ollama_models'

print("⏳ [2/4] Installing Ollama & Pyngrok...")
!curl -fsSL https://ollama.com/install.sh | sh
!pip install pyngrok -q

print("⚡ [3/4] Starting Ollama Engine...")
subprocess.Popen(["ollama", "serve"])
time.sleep(4)

# Choose your model (Downloads once to Google Drive, loads in seconds next time)
MODEL = "deepseek-r1:7b"
print(f"📥 Loading {MODEL} from Google Drive...")
!ollama pull {MODEL}

# ==============================================================================
# 4. EXPOSE PUBLIC TUNNEL VIA NGROK
# Get your free token and 1 free static domain from: https://dashboard.ngrok.com
# ==============================================================================
NGROK_AUTH_TOKEN = "3IKKvEZXCZ4a9TWHkDIWdt7BDR4_5RhQUAikDCYvBSynDCFfA"  # 👈 Paste your token here
STATIC_DOMAIN = ""  # 👈 (Optional) e.g., "my-ai-gpu.ngrok-free.app"

ngrok.set_auth_token(NGROK_AUTH_TOKEN)

if STATIC_DOMAIN:
    tunnel = ngrok.connect(11434, "http", domain=STATIC_DOMAIN)
else:
    tunnel = ngrok.connect(11434, "http")

print("\n" + "="*60)
print(f"🎉 GOOGLE COLAB GPU IS READY AND ONLINE!")
print(f"🔗 YOUR BACKEND URL: {tunnel.public_url}")
print("="*60)
print("👉 Copy this URL and paste it into Settings (⚙️) on your website.")

# Keep execution active
while True:
    time.sleep(60)
