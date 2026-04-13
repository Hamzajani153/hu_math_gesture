# ✏️ Math Gesture AI

Draw math problems in the air using your finger — AI solves them in real time.  
Powered by **OpenAI GPT-4o**, **Anthropic Claude**, or **Google Gemini** with hand tracking via **MediaPipe + cvzone**.

---

## 📋 Table of Contents

- [What This App Does](#what-this-app-does)
- [How It Works](#how-it-works)
- [Gesture Guide](#gesture-guide)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation & Setup](#installation--setup)
- [Running the App](#running-the-app)
- [API Keys & Model Priority](#api-keys--model-priority)
- [Troubleshooting](#troubleshooting)
- [Technical Deep Dive](#technical-deep-dive)

---

## What This App Does

You open the app in your browser, point your webcam at your hand, and:

1. Raise your **index finger** to draw a math problem in the air (e.g. `2 + 2`)
2. Raise **all 5 fingers** to send what you drew to an AI model
3. The AI reads your drawing and returns a **step-by-step solution** in the answer panel

Everything runs locally on your machine. Only the canvas image is sent to the AI API.

---

## How It Works

```
Webcam Feed
    │
    ▼
MediaPipe Hand Tracking  (via cvzone)
    │
    ├── Detects 21 hand landmarks per frame
    ├── fingersUp() → tells which fingers are raised
    │
    ▼
Gesture Logic
    │
    ├── Index only      → draw green line on canvas
    ├── Index + Middle  → lift pen (move without drawing)
    ├── Pinky only      → clear canvas
    └── All 5 fingers   → send canvas image to AI
            │
            ▼
      AI API Call
            │
            ├── OpenAI   → gpt-4o    (vision)
            ├── Anthropic → claude-sonnet-4-5  (vision)
            └── Google   → gemini-1.5-flash   (vision)
            │
            ▼
      Answer displayed
      in Streamlit UI
```

### Smoothing Algorithm

Raw fingertip positions from MediaPipe jitter by ±5–10 pixels per frame due to camera noise.  
To fix this, the app uses **Exponential Smoothing**:

```
smoothed = α × raw_position + (1 − α) × previous_smoothed
```

- `α = 0.5` means 50% weight on the current position, 50% on history
- This removes jitter while keeping the line responsive with near-zero lag
- Tunable at the top of `app.py` via the `ALPHA` constant

### Canvas Architecture

The canvas is a `numpy` array (same size as the webcam frame) initialized **once** at startup.  
It is never auto-reset — only cleared when you use the pinky gesture.  
Drawings persist across frames by blending the canvas onto each webcam frame:

```python
combined[mask] = cv2.addWeighted(img, 0.15, canvas, 0.85, 0)[mask]
```

Only pixels where something is drawn are blended — the rest of the webcam feed stays clear and bright.

---

## Gesture Guide

| Gesture | Fingers State | Action |
|---------|--------------|--------|
| ☝️ Index finger only | `[?, 1, 0, 0, 0]` | Draw green line |
| ✌️ Index + Middle | `[?, 1, 1, 0, 0]` | Lift pen — move freely without drawing |
| 🤙 Pinky only | `[?, 0, 0, 0, 1]` | Clear the entire canvas |
| 🖐️ All 5 fingers | `sum = 5` | Send drawing to AI and get answer |

> **Tip:** Use ✌️ to reposition your hand between numbers/symbols without drawing a connecting line.

---

## Project Structure

```
hu_FYP/
│
├── app.py                  ← Main Streamlit application
├── main.py                 ← Entry point (can be same as app.py)
├── .env                    ← Your API keys (never commit this)
├── .env.example            ← Template — copy to .env and fill in keys
├── requirements.txt        ← All Python dependencies
├── README.md               ← This file
│
└── venv/                   ← Python virtual environment (not committed)
```

---

## Requirements

- **Python** 3.9 or higher
- **Webcam** (built-in or external USB)
- **Windows 10/11** (instructions below use PowerShell)
- At least **one API key** from OpenAI, Anthropic, or Google

---

## Installation & Setup

### Step 1 — Clone or download the project

```
Your project folder should be at:
E:\hamza_files_2\hu_FYP\
```

### Step 2 — Open PowerShell in the project folder

Right-click inside the project folder → **Open in Terminal**  
Or open PowerShell and navigate manually:

```powershell
cd E:\hamza_files_2\hu_FYP
```

### Step 3 — Create a virtual environment

Run this once to create the `venv` folder:

```powershell
python -m venv venv
```

This creates an isolated Python environment so your packages don't conflict with other projects.

### Step 4 — Activate the virtual environment

**Every time you open a new terminal**, activate the venv before running anything:

```powershell
.\venv\Scripts\Activate.ps1
```

You will see `(venv)` appear at the start of your prompt:

```
(venv) PS E:\hamza_files_2\hu_FYP>
```

> ⚠️ **If you get an execution policy error**, run this once in PowerShell as Administrator:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then try activating again.

### Step 5 — Install dependencies

With the venv activated, install all required packages:

```powershell
pip install -r requirements.txt
```

This installs: `streamlit`, `opencv-python`, `cvzone`, `numpy`, `openai`, `anthropic`, `google-generativeai`, `Pillow`, `python-dotenv`.

> This may take 2–5 minutes the first time. You only need to do this once.

### Step 6 — Set up your API keys

Copy the example env file:

```powershell
copy .env.example .env
```

Open `.env` in any text editor (Notepad, VS Code, etc.) and fill in your keys:

```env
# Add whichever key(s) you have — you only need ONE

OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxx
```

**Where to get keys:**
| Provider | URL |
|----------|-----|
| OpenAI | https://platform.openai.com/api-keys |
| Anthropic | https://console.anthropic.com/settings/api-keys |
| Google Gemini | https://aistudio.google.com/app/apikey |

---

## Running the App

### Every time you want to run the app:

**1. Open PowerShell in your project folder**

**2. Activate the virtual environment:**
```powershell
.\venv\Scripts\Activate.ps1
```

**3. Start the app:**
```powershell
streamlit run app.py
```

**4. The browser opens automatically at:**
```
http://localhost:8501
```

**5. In the browser:**
- Set the correct **camera index** (0 = built-in laptop camera, 1 or 2 = external USB webcam)
- Check **▶ Run camera**
- The live webcam feed appears
- Start drawing!

### To stop the app:
Press `Ctrl + C` in the PowerShell terminal.

---

## API Keys & Model Priority

The app automatically selects the AI model based on which key is present in `.env`.  
Priority order:

```
1st choice → OpenAI    (GPT-4o)          if OPENAI_API_KEY is set
2nd choice → Anthropic (Claude 3.5)      if ANTHROPIC_API_KEY is set
3rd choice → Google    (Gemini 1.5)      if GOOGLE_API_KEY is set
```

The active model is shown as a badge in the Answer panel.  
You only need **one key** to run the app. If you have multiple keys, OpenAI is always used first.

---

## Troubleshooting

### ❌ "Cannot open camera at index 0"
The default camera index might be wrong for your setup.
- Try index `1` or `2` in the camera index field
- Make sure no other app (Zoom, Teams, OBS) is using the camera
- Unplug and replug external webcams, then restart the app

### ❌ "Failed to read frame"
- Close all other applications that might be accessing the camera
- Try a different camera index
- Restart the Streamlit app (`Ctrl+C` then `streamlit run app.py`)

### ❌ Drawing is not appearing
- Make sure only your **index finger** is raised — all others must be down
- Check the **Gesture** pill below the camera — it shows what the app is detecting in real time
- Improve lighting so your hand is clearly visible to the camera

### ❌ All 5 fingers raised but no AI answer
- Check the Gesture pill shows `🖐️ Sending to AI...`
- Make sure your `.env` file has a valid API key
- There is a **4-second cooldown** between AI calls — hold the gesture for a moment
- Check the PowerShell terminal for any error messages

### ❌ "Execution policy" error when activating venv
Run this once in PowerShell as Administrator:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### ❌ Warnings in the terminal (TensorFlow, protobuf, etc.)
These are **normal** — they come from MediaPipe and Google's libraries internally.  
They do not affect functionality. The app works fine with these warnings present.

---

## Technical Deep Dive

### Libraries Used

| Library | Purpose |
|---------|---------|
| `streamlit` | Web UI — renders camera feed, answer panel, controls |
| `opencv-python` | Camera capture, image processing, drawing lines on canvas |
| `cvzone` | Wrapper around MediaPipe for hand detection and finger counting |
| `mediapipe` | Google's ML library — detects 21 hand landmarks per frame |
| `numpy` | Canvas as a pixel array, blending operations |
| `Pillow` | Converts numpy canvas to PIL Image for AI API calls |
| `openai` | GPT-4o vision API |
| `anthropic` | Claude vision API |
| `google-generativeai` | Gemini vision API |
| `python-dotenv` | Loads API keys from `.env` file |

### Why Streamlit's `while` Loop Pattern?

Normally Streamlit reruns the entire script on every interaction (button click, checkbox, etc.).  
This would re-open the camera every rerun and cause flickering.

The fix: run the camera inside a **single execution** of the script using a `while True` loop,  
and update `st.empty()` placeholders **in-place** rather than re-creating widgets.  
This means no reruns during the camera loop → no flickering → smooth 30fps feed.

### Canvas Blending

```python
mask     = canvas.astype(bool)          # True where pixels are drawn
combined = img.copy()                   # start from clean webcam frame
combined[mask] = cv2.addWeighted(       # only blend where drawing exists
    img, 0.15, canvas, 0.85, 0
)[mask]
```

Result: drawn lines appear at 85% opacity on top of the webcam feed.  
Undrawn areas are 100% clean webcam — no darkening effect.

### AI Vision Call Flow

```
Canvas (numpy BGR array)
    │
    ▼
PIL Image → PNG bytes → Base64 string
    │
    ▼
API call with image + prompt
    │
    ▼
Text response → displayed in answer panel
```

The prompt instructs the model to identify the math expression from the rough finger-drawn image,  
solve it step by step, and end with `ANSWER: <result>` on the final line.

---

## Author

**Hamza** — Final Year Project, Hamdard University  
Built with Python, Streamlit, MediaPipe, and OpenAI/Anthropic/Google APIs.