import os, base64, io, time
import cv2
import numpy as np
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
from dotenv import load_dotenv

# ── Keys ───────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY",    "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY",    "").strip()

def get_default_provider():
    if OPENAI_API_KEY:    return "openai",    "GPT-4o (OpenAI)"
    if ANTHROPIC_API_KEY: return "anthropic", "Claude 3.5 Sonnet"
    if GOOGLE_API_KEY:    return "gemini",    "Gemini 1.5 Flash"
    return None, None

DEFAULT_PROVIDER, DEFAULT_LABEL = get_default_provider()

# ── Page ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Math Gesture AI", layout="wide", page_icon="✏️")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: #080810;
}
[data-testid="stHeader"] { background: transparent; }

h1,h2,h3 { color: #e8e3f8 !important; }

/* ── Answer card ── */
.answer-card {
    background: linear-gradient(135deg, #0f1923 0%, #111a2e 100%);
    border: 1px solid #1e3a5f;
    border-left: 4px solid #00d4aa;
    border-radius: 14px;
    padding: 22px 26px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    color: #a8f0d8;
    min-height: 180px;
    line-height: 1.85;
    white-space: pre-wrap;
    box-shadow: 0 0 30px rgba(0,212,170,0.08);
}
.answer-card .label {
    font-size: 0.7rem;
    color: #00d4aa;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 10px;
    display: block;
}

/* ── Thinking animation ── */
.thinking-card {
    background: linear-gradient(135deg, #0f1f18 0%, #111a14 100%);
    border: 1px solid #1e5f3a;
    border-left: 4px solid #00ff88;
    border-radius: 14px;
    padding: 22px 26px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    color: #88ffcc;
    min-height: 80px;
    line-height: 1.7;
}

/* ── Gesture live display ── */
.gesture-pill {
    display: inline-block;
    background: #12122a;
    border: 1px solid #333388;
    border-radius: 999px;
    padding: 6px 18px;
    font-size: 0.85rem;
    color: #9988ff;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 8px;
    width: 100%;
    text-align: center;
}
.gesture-pill.drawing  { border-color:#00ff88; color:#00ff88; background:#001a10; }
.gesture-pill.clearing { border-color:#ff6644; color:#ff6644; background:#1a0a00; }
.gesture-pill.solving  { border-color:#ffcc00; color:#ffcc00; background:#1a1400;
                         animation: blink 0.6s ease-in-out infinite alternate; }
@keyframes blink { from{opacity:1} to{opacity:0.3} }

/* ── Legend ── */
.legend {
    background: #0c0c1a;
    border: 1px solid #222244;
    border-radius: 12px;
    padding: 16px 20px;
    font-size: 0.82rem;
    color: #8888bb;
    line-height: 2.4;
}
.legend b { color: #ccccff; }

/* ── Provider badge ── */
.badge {
    background: #111130;
    border: 1px solid #4444aa;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: #8888ee;
    display: inline-block;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

if DEFAULT_PROVIDER is None:
    st.error("No API keys found. Add OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY to your .env")
    st.stop()

# ── Cached AI clients ──────────────────────────────────────────────────────────
@st.cache_resource
def get_detector():
    return HandDetector(
        staticMode=False, maxHands=1,
        modelComplexity=1, detectionCon=0.7, minTrackCon=0.5
    )

@st.cache_resource
def get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def get_anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

@st.cache_resource
def get_gemini():
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")


# ── AI solver ──────────────────────────────────────────────────────────────────
PROMPT = """This image contains a handwritten math problem drawn with a finger in the air on a dark background.
The drawing may look rough. Please:
1. Identify the math expression or problem
2. Solve it step by step
3. State the final answer clearly on the last line as: ANSWER: <result>

Be concise and clear."""

def canvas_to_b64(canvas: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(canvas).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def is_canvas_empty(canvas: np.ndarray) -> bool:
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    return cv2.countNonZero(gray) < 100   # less than 100 non-black pixels = empty

def solve(canvas: np.ndarray, provider: str) -> str:
    if is_canvas_empty(canvas):
        return "⚠️  Nothing drawn yet!\nDraw your math problem first, then raise all 5 fingers."
    try:
        if provider == "openai":
            b64 = canvas_to_b64(canvas)
            r = get_openai_client().chat.completions.create(
                model="gpt-4o", max_tokens=1024,
                messages=[{"role": "user", "content": [
                    {"type": "text",      "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]}]
            )
            return r.choices[0].message.content

        elif provider == "anthropic":
            b64 = canvas_to_b64(canvas)
            r = get_anthropic_client().messages.create(
                model="claude-sonnet-4-5", max_tokens=1024,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": b64
                    }},
                    {"type": "text", "text": PROMPT}
                ]}]
            )
            return r.content[0].text

        elif provider == "gemini":
            return get_gemini().generate_content(
                [PROMPT, Image.fromarray(canvas)]
            ).text

    except Exception as e:
        return f"❌ API Error ({provider}):\n{str(e)}"
    return "No response received."


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## ✏️ Math Gesture AI")
st.markdown("---")

col_cam, col_ans = st.columns([3, 2], gap="large")

with col_cam:
    cam_index = int(st.number_input(
        "📷 Camera index  (0 = built-in · 1 or 2 = external)",
        min_value=0, max_value=5, value=0, step=1
    ))
    run = st.checkbox("▶  Run camera", value=False)
    status_ph  = st.empty()
    frame_ph   = st.image([])
    gesture_ph = st.empty()

with col_ans:
    st.markdown("### 🤖 AI Answer")

    model_options = {}
    if OPENAI_API_KEY:    model_options["🟢 GPT-4o (OpenAI)"]               = "openai"
    if ANTHROPIC_API_KEY: model_options["🟠 Claude 3.5 Sonnet (Anthropic)"] = "anthropic"
    if GOOGLE_API_KEY:    model_options["🔵 Gemini 1.5 Flash (Google)"]     = "gemini"

    selected_label = st.selectbox(
        "Model",
        list(model_options.keys()),
        index=list(model_options.values()).index(DEFAULT_PROVIDER)
    )
    selected_provider = model_options[selected_label]

    st.markdown(
        f'<div class="badge">Default: {DEFAULT_LABEL}</div>',
        unsafe_allow_html=True
    )

    answer_ph = st.empty()
    answer_ph.markdown(
        '<div class="answer-card">'
        '<span class="label">Waiting for input</span>'
        'Draw a math problem ✏️\nthen open all 5 fingers 🖐️ to solve'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="legend">
        ☝️ &nbsp;<b>Index finger only</b> &nbsp;&nbsp;&nbsp;→ Draw (green line)<br>
        ✌️ &nbsp;<b>Index + Middle</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ Move without drawing<br>
        👍 &nbsp;<b>Thumb only</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ Clear canvas<br>
        🖐️ &nbsp;<b>All 5 fingers open</b> &nbsp;→ Solve with AI
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA LOOP
# ══════════════════════════════════════════════════════════════════════════════
# Drawing settings
LINE_COLOR    = (0, 255, 80)    # bright green
LINE_THICK    = 12              # bold line
TIP_RADIUS    = 14              # fingertip circle radius
TIP_COLOR     = (0, 255, 80)

if run:
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        status_ph.error(f"❌ Cannot open camera {cam_index}. Try 0, 1, or 2.")
        st.stop()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cap.set(cv2.CAP_PROP_FPS,          30)

    status_ph.success("✅ Camera running — uncheck to stop.")

    detector     = get_detector()
    canvas       = None
    prev_pos     = None
    output_text  = ""
    last_ai_time = 0
    AI_COOLDOWN  = 4.0    # seconds — prevent repeat firing

    TARGET_FPS = 30
    frame_time = 1.0 / TARGET_FPS

    while True:
        t0 = time.time()

        success, img = cap.read()
        if not success or img is None:
            status_ph.error("❌ Frame read failed. Check camera / close other apps using it.")
            break

        img = cv2.flip(img, 1)   # mirror

        # Init canvas
        if canvas is None or canvas.shape != img.shape:
            canvas = np.zeros_like(img)

        # ── Detect hand ────────────────────────────────────────────────────────
        hands, img = detector.findHands(img, draw=True, flipType=True)

        gesture_text  = "🤚 No hand detected"
        gesture_class = ""

        if hands:
            hand    = hands[0]
            lmList  = hand["lmList"]
            fingers = detector.fingersUp(hand)
            total   = sum(fingers)

            # ── DRAW  — index finger only [0,1,0,0,0] ─────────────────────────
            if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                # Use index fingertip (landmark 8)
                current_pos = lmList[8][0:2]

                if prev_pos is not None:
                    # Draw thick green line — works in ANY direction
                    cv2.line(canvas,
                             (prev_pos[0],    prev_pos[1]),
                             (current_pos[0], current_pos[1]),
                             LINE_COLOR, LINE_THICK)

                # Glowing fingertip circle
                cv2.circle(img,
                           (current_pos[0], current_pos[1]),
                           TIP_RADIUS, TIP_COLOR, cv2.FILLED)
                cv2.circle(img,
                           (current_pos[0], current_pos[1]),
                           TIP_RADIUS + 6, (0, 180, 50), 2)  # outer glow ring

                prev_pos      = current_pos
                gesture_text  = "☝️  Drawing..."
                gesture_class = "drawing"

            # ── MOVE (pen up) — index + middle [?,1,1,0,0] ───────────────────
            elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
                prev_pos      = None   # lift pen — no line drawn
                gesture_text  = "✌️  Pen lifted (moving)"
                gesture_class = ""

            # ── CLEAR — thumb only [1,0,0,0,0] ───────────────────────────────
            elif fingers[0] == 1 and total == 1:
                canvas        = np.zeros_like(img)
                prev_pos      = None
                output_text   = ""
                gesture_text  = "👍  Canvas cleared!"
                gesture_class = "clearing"
                answer_ph.markdown(
                    '<div class="answer-card">'
                    '<span class="label">Canvas cleared</span>'
                    'Draw a new problem ✏️'
                    '</div>',
                    unsafe_allow_html=True
                )

            # ── SOLVE — all 5 fingers open (total == 5) ───────────────────────
            elif total == 5:
                now = time.time()
                gesture_text  = "🖐️  Sending to AI..."
                gesture_class = "solving"

                if now - last_ai_time > AI_COOLDOWN:
                    last_ai_time = now
                    answer_ph.markdown(
                        f'<div class="thinking-card">'
                        f'⏳ Solving with {selected_label}…\n'
                        f'Please hold still for a moment.'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    result = solve(canvas, selected_provider)
                    output_text = result

            else:
                prev_pos     = None   # any other gesture lifts pen
                gesture_text = f"🤟 Fingers: {fingers}  (sum={total})"

        else:
            prev_pos = None   # hand left frame — reset pen

        # ── Show answer ────────────────────────────────────────────────────────
        if output_text and gesture_class != "solving":
            answer_ph.markdown(
                f'<div class="answer-card">'
                f'<span class="label">✅ Answer — {selected_label}</span>'
                f'{output_text}'
                f'</div>',
                unsafe_allow_html=True
            )

        # ── Gesture pill ───────────────────────────────────────────────────────
        gesture_ph.markdown(
            f'<div class="gesture-pill {gesture_class}">{gesture_text}</div>',
            unsafe_allow_html=True
        )

        # ── Blend canvas onto frame ────────────────────────────────────────────
        # Only blend where canvas has actual drawing (avoids darkening whole frame)
        canvas_mask  = canvas.astype(bool)
        combined     = img.copy()
        combined[canvas_mask] = cv2.addWeighted(img, 0.2, canvas, 0.8, 0)[canvas_mask]

        frame_ph.image(combined, channels="BGR")

        # ── FPS cap ────────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        wait    = frame_time - elapsed
        if wait > 0:
            time.sleep(wait)

    cap.release()