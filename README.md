# 🚀 Private Cloud AI Setup & Testing Guide

This guide walks you through testing your full **Private Cloud AI Workstation** with **0% load on your laptop** and **zero API subscription fees**.

---

## 📁 Files Created for You:
1. [**`index.html`**](file:///C:/Users/divya/.gemini/antigravity/brain/f09082e5-1353-47d2-aad4-574dac9a11bc/index.html) — The modern web chat frontend with:
   - DeepSeek-R1 integration
   - **🌐 Live Web Search toggle (DuckDuckGo)**
   - Markdown & code syntax highlighting
   - Settings modal for instant backend switching
2. [**`colab_server.py`**](file:///C:/Users/divya/.gemini/antigravity/brain/f09082e5-1353-47d2-aad4-574dac9a11bc/colab_server.py) — The ready-to-run Google Colab GPU script.

---

## 🧪 How to Test It in 3 Easy Steps

### Step 1: Start Your Google Colab GPU Backend
1. Open a new notebook at [colab.research.google.com](https://colab.research.google.com).
2. Go to **Runtime > Change runtime type** and select **T4 GPU**.
3. Copy all the code from [`colab_server.py`](file:///C:/Users/divya/.gemini/antigravity/brain/f09082e5-1353-47d2-aad4-574dac9a11bc/colab_server.py), paste it into Colab, and insert your free **Ngrok Auth Token** (from [dashboard.ngrok.com](https://dashboard.ngrok.com)).
4. Click **Run**.
5. When finished, Colab will print:
   ```
   🎉 GOOGLE COLAB GPU IS READY AND ONLINE!
   🔗 YOUR BACKEND URL: https://xxxx-xx-xx.ngrok-free.app
   ```

---

### Step 2: Open Your Website
1. Double-click [**`index.html`**](file:///C:/Users/divya/.gemini/antigravity/brain/f09082e5-1353-47d2-aad4-574dac9a11bc/index.html) to open it in your browser (Chrome, Edge, Brave, etc.).
2. Click the **Settings (⚙️)** icon in the top right.
3. Paste your Colab Backend URL (`https://xxxx.ngrok-free.app`) and click **Save Settings**.
4. The status badge will turn **🟢 Colab Online (T4 GPU)**.

---

### Step 3: Test DeepSeek-R1 & Live Web Search!
- **Regular Mode:** Type any complex reasoning or coding question:
  > *"Write a Python script to scrape stock prices and explain the logic."*
- **Live Search Mode:** Turn ON the **"🌐 Live Search"** toggle switch and ask about recent events:
  > *"What are the latest updates in AI this week?"*

---

## 🌐 How to Deploy to Cloudflare Pages (Free Permanent Link)
When you're ready to access this website from your mobile phone or from any computer:
1. Create a free repository on [github.com](https://github.com) and upload `index.html`.
2. Go to [dash.cloudflare.com](https://dash.cloudflare.com) > **Workers & Pages** > **Create application** > **Pages** > **Connect to Git**.
3. Select your repository and click **Deploy**.
4. Cloudflare gives you a permanent URL (e.g. `https://my-private-ai.pages.dev`).
