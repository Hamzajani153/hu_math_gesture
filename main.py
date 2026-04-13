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
[data-testid="stAppViewContainer"] { background: #080810; }
[data-testid="stHeader"]           { background: transparent; }
h1,h2,h3                           { color: #e8e3f8 !important; }

.answer-card {
    background: linear-gradient(135deg, #0f1923, #111a2e);
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
.answer-card .lbl {
    font-size: 0.7rem; color: #00d4aa;
    text-transform: uppercase; letter-spacing: 2px;
    margin-bottom: 12px; display: block;
}
.thinking-card {
    background: linear-gradient(135deg, #0f1f18, #111a14);
    border: 1px solid #1e5f3a;
    border-left: 4px solid #00ff88;
    border-radius: 14px;
    padding: 22px 26px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem; color: #88ffcc;
    min-height: 80px; line-height: 1.7;
    animation: pulse 0.8s ease-in-out infinite alternate;
}
@keyframes pulse { from{opacity:1} to{opacity:0.3} }

.gesture-pill {
    display: block; background: #12122a;
    border: 1px solid #333388; border-radius: 999px;
    padding: 7px 20px; font-size: 0.85rem; color: #9988ff;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 8px; text-align: center;
}
.gesture-pill.drawing  { border-color:#00ff88; color:#00ff88; background:#001a10; }
.gesture-pill.clearing { border-color:#ff6644; color:#ff6644; background:#1a0a00; }
.gesture-pill.solving  { border-color:#ffcc00; color:#ffcc00; background:#1a1400;
                         animation: pulse 0.6s ease-in-out infinite alternate; }
.model-badge {
    background:#111130; border:1px solid #4444aa; border-radius:999px;
    padding:5px 16px; font-size:0.78rem; color:#8888ee;
    display:inline-block; margin-bottom:14px;
}
.legend {
    background:#0c0c1a; border:1px solid #222244; border-radius:12px;
    padding:16px 20px; font-size:0.82rem; color:#8888bb;
    line-height:2.6; margin-top:16px;
}
.legend b { color:#ccccff; }
</style>
""", unsafe_allow_html=True)

if DEFAULT_PROVIDER is None:
    st.error("No API keys found. Add at least one key to your .env file.")
    st.stop()

# ── Cached resources ───────────────────────────────────────────────────────────
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
PROMPT = """This image shows a handwritten math problem drawn with a finger on a dark background.
The lines are bright green on black. Please:
1. Identify the math expression
2. Solve it step by step
3. State the final answer clearly as:  ANSWER: <result>
Be concise."""

def canvas_to_b64(canvas: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(canvas).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def is_empty(canvas: np.ndarray) -> bool:
    return cv2.countNonZero(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)) < 100

def solve(canvas: np.ndarray, provider: str) -> str:
    if is_empty(canvas):
        return "⚠️  Nothing drawn yet!\nDraw your math problem first,\nthen open all 5 fingers."
    try:
        if provider == "openai":
            b64 = canvas_to_b64(canvas)
            r = get_openai_client().chat.completions.create(
                model="gpt-4o", max_tokens=1024,
                messages=[{"role": "user", "content": [
                    {"type": "text",      "text": PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64}"
                    }}
                ]}]
            )
            return r.choices[0].message.content
        elif provider == "anthropic":
            b64 = canvas_to_b64(canvas)
            r = get_anthropic_client().messages.create(
                model="claude-sonnet-4-5", max_tokens=1024,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64
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
#  EXPONENTIAL SMOOTHING  — near-zero lag, removes jitter
#  Formula: smooth = alpha * raw + (1 - alpha) * prev_smooth
#  alpha = 0.5  →  50% new position, 50% history  (responsive + stable)
#  Higher alpha = more responsive but more jitter
#  Lower alpha  = smoother but more lag
# ══════════════════════════════════════════════════════════════════════════════
ALPHA = 0.5   # tune: 0.4 = ultra smooth, 0.6 = more responsive

class ExpSmoother:
    def __init__(self, alpha: float = ALPHA):
        self.alpha = alpha
        self.sx    = None
        self.sy    = None

    def update(self, x: int, y: int):
        if self.sx is None:          # first point — no history yet
            self.sx, self.sy = float(x), float(y)
        else:
            self.sx = self.alpha * x + (1 - self.alpha) * self.sx
            self.sy = self.alpha * y + (1 - self.alpha) * self.sy

    def get(self):
        return int(self.sx), int(self.sy)

    def reset(self):
        self.sx = None
        self.sy = None

    @property
    def ready(self):
        return self.sx is not None


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
    run        = st.checkbox("▶  Run camera", value=False)
    status_ph  = st.empty()
    frame_ph   = st.image([])
    gesture_ph = st.empty()

with col_ans:
    st.markdown("### 🤖 AI Answer")

    answer_ph = st.empty()
    answer_ph.markdown(
        '<div class="answer-card">'
        '<span class="lbl">Waiting for input</span>'
        'Draw a math problem ✏️\nthen open all 5 fingers 🖐️ to solve.'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="legend">
        ☝️ &nbsp;<b>Index finger only</b> &nbsp;&nbsp;&nbsp;→ Draw<br>
        ✌️ &nbsp;<b>Index + Middle</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ Lift pen (move freely)<br>
        🤙 &nbsp;<b>Pinky only</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ Clear canvas<br>
        🖐️ &nbsp;<b>All 5 fingers</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ Solve with AI
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA LOOP
# ══════════════════════════════════════════════════════════════════════════════
LINE_COLOR = (0, 255, 80)    # bright green
LINE_THICK = 14
TIP_R      = 15

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
    smoother     = ExpSmoother(ALPHA)

    # ── FIX 2: Canvas created ONCE here, never auto-reset ─────────────────────
    # We read one frame just to get dimensions, then build canvas from that
    ok, first_frame = cap.read()
    if not ok:
        status_ph.error("❌ Could not read first frame.")
        st.stop()
    first_frame = cv2.flip(first_frame, 1)
    canvas      = np.zeros_like(first_frame)   # permanent canvas
    h, w        = canvas.shape[:2]             # fixed size — never changes

    prev_smooth  = None
    output_text  = ""
    last_ai_time = 0
    AI_COOLDOWN  = 4.0

    TARGET_FPS = 30
    frame_time = 1.0 / TARGET_FPS

    while True:
        t0 = time.time()

        success, img = cap.read()
        if not success or img is None:
            status_ph.error("❌ Frame read failed. Close other apps using the camera.")
            break

        img = cv2.flip(img, 1)

        # ── FIX 2: Resize frame to match canvas if camera fluctuates ──────────
        # Never reset canvas — just resize the frame to match
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))

        # ── Detect hand ────────────────────────────────────────────────────────
        hands, img = detector.findHands(img, draw=True, flipType=True)

        gesture_text  = "🤚 No hand detected"
        gesture_class = ""

        if hands:
            hand    = hands[0]
            lmList  = hand["lmList"]
            fingers = detector.fingersUp(hand)
            # fingers = [thumb, index, middle, ring, pinky]
            total   = sum(fingers)

            # ── ☝️ DRAW — index only, rest down ───────────────────────────────
            # FIX 1: Use exponential smoother instead of averaging buffer
            if (fingers[1] == 1 and
                fingers[2] == 0 and
                fingers[3] == 0 and
                fingers[4] == 0):

                raw_x, raw_y = int(lmList[8][0]), int(lmList[8][1])
                smoother.update(raw_x, raw_y)
                sx, sy = smoother.get()

                if prev_smooth is not None and smoother.ready:
                    cv2.line(canvas,
                             prev_smooth, (sx, sy),
                             LINE_COLOR, LINE_THICK,
                             lineType=cv2.LINE_AA)

                # Fingertip glow
                cv2.circle(img, (sx, sy), TIP_R,      LINE_COLOR,   cv2.FILLED)
                cv2.circle(img, (sx, sy), TIP_R + 7,  (0, 180, 50), 2)

                prev_smooth   = (sx, sy)
                gesture_text  = "☝️  Drawing..."
                gesture_class = "drawing"

            # ── ✌️ PEN LIFT — index + middle ──────────────────────────────────
            elif (fingers[1] == 1 and
                  fingers[2] == 1 and          
                  fingers[3] == 0 and
                  fingers[4] == 0):
                smoother.reset()
                prev_smooth   = None
                gesture_text  = "✌️  Pen lifted"
                gesture_class = ""

            # ── 🤙 CLEAR — pinky only [?,0,0,0,1] ─────────────────────────────
            # FIX 3: Removed thumb-clear. Now pinky only clears.
            elif (fingers[4] == 1 and
                  fingers[1] == 0 and
                  fingers[2] == 0 and
                  fingers[3] == 0):
                canvas        = np.zeros((h, w, 3), dtype=np.uint8)
                smoother.reset()
                prev_smooth   = None
                output_text   = ""
                gesture_text  = "🤙  Canvas cleared!"
                gesture_class = "clearing"
                answer_ph.markdown(
                    '<div class="answer-card">'
                    '<span class="lbl">Canvas cleared</span>'
                    'Draw a new problem ✏️'
                    '</div>',
                    unsafe_allow_html=True
                )

            # ── 🖐️ SOLVE — all 5 fingers ──────────────────────────────────────
            elif total == 5:
                smoother.reset()
                prev_smooth   = None
                now           = time.time()
                gesture_text  = "🖐️  Sending to AI..."
                gesture_class = "solving"

                if now - last_ai_time > AI_COOLDOWN:
                    last_ai_time = now
                    answer_ph.markdown(
                        f'<div class="thinking-card">'
                        f'⏳ Solving with …\n'
                        f'Please hold still.'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    result = solve(canvas, DEFAULT_PROVIDER)
                    if result:
                        output_text = result

            # ── Any other gesture — just lift pen ──────────────────────────────
            else:
                smoother.reset()
                prev_smooth  = None
                gesture_text = f"🤟 Fingers up: {fingers}  (total={total})"

        else:
            # Hand left frame
            smoother.reset() 
            prev_smooth = None

        # ── Show answer ────────────────────────────────────────────────────────
        if output_text and gesture_class != "solving":
            answer_ph.markdown(
                f'<div class="answer-card">'
                f'<span class="lbl">✅ Answer — </span>'
                f'{output_text}'
                f'</div>',
                unsafe_allow_html=True
            )

        # ── Gesture pill ───────────────────────────────────────────────────────
        gesture_ph.markdown(
            f'<div class="gesture-pill {gesture_class}">{gesture_text}</div>',
            unsafe_allow_html=True
        )

        # ── Blend: draw only where canvas has pixels ───────────────────────────
        mask     = canvas.astype(bool)
        combined = img.copy()
        combined[mask] = cv2.addWeighted(img, 0.15, canvas, 0.85, 0)[mask]

        frame_ph.image(combined, channels="BGR")

        # ── FPS cap ────────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        wait    = frame_time - elapsed
        if wait > 0:
            time.sleep(wait)

    cap.release()