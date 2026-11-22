import os
import streamlit as st
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
except ImportError:
    webrtc_streamer = None
    class VideoProcessorBase: pass
    class RTCConfiguration:
        def __init__(self, *args, **kwargs): pass

try:
    from twilio.rest import Client
except ImportError:
    Client = None

import av
import cv2
try:
    import mediapipe as mp
except ImportError:
    mp = None

import joblib
import numpy as np
import math
import threading
import time
try:
    from gtts import gTTS
except ImportError:
    gTTS = None
import io
import base64

# --- Page Config ---
st.set_page_config(
    page_title="ISL Live Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling & GUI Theme ---
st.markdown("""
<style>
    /* Global layout & typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Card */
    .main-header {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2b40 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 24px 30px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align: center;
    }
    
    .badge-bar {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-top: 10px;
        flex-wrap: wrap;
    }
    
    .status-badge {
        font-size: 12px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        background: rgba(76, 175, 80, 0.15);
        color: #4CAF50;
        border: 1px solid #4CAF50;
    }

    .status-badge-blue {
        font-size: 12px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        background: rgba(33, 150, 243, 0.15);
        color: #2196F3;
        border: 1px solid #2196F3;
    }

    .status-badge-purple {
        font-size: 12px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        background: rgba(156, 39, 176, 0.15);
        color: #BA68C8;
        border: 1px solid #BA68C8;
    }

    /* Sentence Card */
    .sentence-card {
        background: #181926;
        border: 2px solid #2e3046;
        border-radius: 16px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4);
    }

    .sentence-title {
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8b92ad;
        margin-bottom: 8px;
    }

    .sentence-display {
        font-size: 26px;
        font-weight: 700;
        color: #00E676;
        min-height: 48px;
        line-height: 1.4;
        display: flex;
        align-items: center;
    }

    .sentence-placeholder {
        color: #565d7a;
        font-style: italic;
        font-size: 18px;
        font-weight: 400;
    }

    /* Blink Indicator */
    .blink-card {
        background: #181926;
        border: 2px solid #2e3046;
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        margin: 15px 0;
    }

    .blink-dots {
        font-size: 28px;
        margin: 8px 0;
        letter-spacing: 6px;
    }

    .alert-box {
        font-size: 16px;
        font-weight: 600;
        color: #ff5252;
        background: rgba(255, 82, 82, 0.12);
        padding: 12px 18px;
        border-radius: 10px;
        border: 1px solid #ff5252;
        margin: 12px 0;
        text-align: center;
    }

    .camera-banner {
        font-size: 13px;
        color: #9aa2c2;
        background: rgba(45, 55, 72, 0.6);
        border: 1px dashed rgba(255, 255, 255, 0.15);
        padding: 10px 16px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🤟 ISL Recognition")
    st.markdown("### 📋 Supported Signs (13 Total)")

    signs_data = [
        ("👋 Hello", "Open Palm (all fingers up)"),
        ("👍 Good", "Thumbs Up gesture"),
        ("👎 No", "Thumbs Down gesture"),
        ("👉 Look", "Pointing Index finger"),
        ("✊ Stop", "Closed Fist"),
        ("🙏 Thank You", "Flat hand from chin outward"),
        ("😔 Sorry", "Fist circular motion"),
        ("🤲 Please", "Open hand on chest"),
        ("😢 Sad", "Index trace cheek"),
        ("💧 Water", "W hand shape"),
        ("🍎 Eat", "Fingertips pinched together"),
        ("🤟 I Love You", "Thumb + Index + Pinky"),
        ("👌 OK", "Thumb + Index circle"),
    ]

    for sign, desc in signs_data:
        st.markdown(f"**{sign}** — *{desc}*")

    st.divider()

    st.markdown("### 💡 How It Works")
    st.markdown("""
    1. Click **START** in video area and allow camera.
    2. Hold any sign steadily for 3 frames.
    3. Words automatically append to your sentence.
    4. **Blink 3 times** to speak the sentence hands-free!
    """)

    st.divider()

    st.markdown("### 👤 Author Information")
    st.markdown("""
    **Student:** Jigar J Langaliya  
    **Institute:** Sir Bhavsinhji Polytechnic Institute  
    **Center:** UFOS Training Center, Bhavnagar  
    **Mentor:** Aditya Dodiya  
    """)

# --- Main Header ---
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 32px; font-weight: 800; color: #ffffff;">
        🤟 Indian Sign Language — Live Translator
    </h1>
    <p style="margin: 6px 0 0 0; color: #a5b4fc; font-size: 16px;">
        Real-time gesture recognition with hands-free eye-blink speech trigger
    </p>
    <div class="badge-bar">
        <span class="status-badge">● 13 Gestures Active</span>
        <span class="status-badge-blue">⚡ Low Latency WebRTC</span>
        <span class="status-badge-purple">👁️ Blink-to-Speech v2.0</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="camera-banner">
    📷 <strong>Camera Access:</strong> Click <strong>START</strong> below to begin. Allow browser camera permission when prompted.<br>
    <em>Tip: If blocked, click the lock 🔒 icon in your browser address bar and enable Camera access.</em>
</div>
""", unsafe_allow_html=True)


# --- Load Model ---
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    for candidate in ['isl_master_model.pkl', 'isl_final_fully_model.pkl']:
        candidate_path = os.path.join(BASE_DIR, candidate)
        if os.path.exists(candidate_path):
            try:
                return joblib.load(candidate_path)
            except Exception as e:
                st.error(f"Error loading {candidate}: {e}")
    st.error("Model error: No model file found in Day_14_Final_Day.")
    return None


model = load_model()

SIGN_LABELS = {
    'open_palm': 'Hello', 'thumbs_up': 'Good', 'thumbs_down': 'No',
    'pointing': 'Look', 'fist': 'Stop', 'thank_you': 'Thank You',
    'sorry': 'Sorry', 'please': 'Please', 'unhappy': 'Sad',
    'water': 'Water', 'eat': 'Eat', 'ily': 'I Love You',
    'cool': 'OK', 'ok': 'OK',
}


def translate(label):
    return SIGN_LABELS.get(label, label.replace('_', ' ').title())


if mp:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
else:
    mp_hands = None
    mp_drawing = None

# Import BlinkDetector (robust to casing)
try:
    from blink_Detector import BlinkDetector
except ImportError:
    try:
        from blink_detector import BlinkDetector
    except ImportError:
        class BlinkDetector:
            def __init__(self): pass

            def check(self, img, rgb_frame=None): return 0

            def reset(self): pass


class SignProcessor(VideoProcessorBase):
    def __init__(self):
        if mp_hands:
            self.hands = mp_hands.Hands(
                static_image_mode=False, max_num_hands=1, model_complexity=0,
                min_detection_confidence=0.6, min_tracking_confidence=0.5
            )
        else:
            self.hands = None
        self.detector = BlinkDetector()
        self.last_capture_time = 0
        self.sentence = []
        self.last_prediction = None
        self.stable_count = 0
        self.captured_this_sign = False
        self.current_prediction_text = ""
        self.current_blink_count = 0
        self.lock = threading.Lock()
        self.text_to_speak = ""
        self.show_alert = False
        self.speak_triggered = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # Convert to RGB once and reuse
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blink_count = self.detector.check(img, rgb_frame=rgb_img) or 0

        raw_prediction = None
        confidence = 0.0

        if self.hands:
            rgb_img.flags.writeable = False
            results = self.hands.process(rgb_img)
            rgb_img.flags.writeable = True

            if results.multi_hand_landmarks and model is not None:
                for hand_landmarks in results.multi_hand_landmarks:
                    if mp_drawing:
                        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    lm = hand_landmarks.landmark
                    wrist_x, wrist_y = lm[0].x, lm[0].y
                    coordinates = []
                    for i in range(21):
                        coordinates.extend([lm[i].x - wrist_x, lm[i].y - wrist_y])

                    def dist(i1, i2):
                        return math.sqrt((lm[i1].x - lm[i2].x) ** 2 + (lm[i1].y - lm[i2].y) ** 2)

                    features = coordinates + [dist(4, 8), dist(8, 12), dist(0, 8)]
                    feature_array = np.array(features).reshape(1, -1)

                    try:
                        probs = model.predict_proba(feature_array)[0]
                        probs_sum = np.sum(probs)
                        if probs_sum > 0:
                            probs = probs / probs_sum
                        max_idx = int(np.argmax(probs))
                        raw_prediction = model.classes_[max_idx]
                        confidence = float(probs[max_idx]) * 100.0
                    except Exception:
                        pass

        with self.lock:
            self.current_blink_count = blink_count

            if raw_prediction:
                if raw_prediction == self.last_prediction:
                    self.stable_count += 1
                else:
                    self.stable_count = 0
                    self.last_prediction = raw_prediction
                    self.captured_this_sign = False
            else:
                self.stable_count = 0
                self.last_prediction = None
                self.captured_this_sign = False
                self.current_prediction_text = ""

            if self.stable_count >= 3 and raw_prediction:
                ct = time.time()
                human_text = translate(raw_prediction)
                self.current_prediction_text = human_text

                if confidence >= 70.0 and not self.captured_this_sign and ct - self.last_capture_time >= 1.5:
                    self.sentence.append(human_text)
                    self.captured_this_sign = True
                    self.last_capture_time = ct

            # Trigger TTS when 3 blinks detected
            if self.current_blink_count >= 3 and not self.speak_triggered:
                self.speak_triggered = True
                if self.sentence:
                    self.text_to_speak = " ".join(self.sentence)
                    self.sentence = []
                    self.captured_this_sign = False
                    self.current_prediction_text = ""
                else:
                    self.show_alert = True
                    self.text_to_speak = "No sentence to speak. Please make a sign first."

            # Reset speak trigger when blinks are reset
            if self.current_blink_count == 0:
                self.speak_triggered = False

        cv2.putText(img, f"Blinks: {self.current_blink_count}/3", (img.shape[1] - 180, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        # Display Predicted Sign (top-left)
        if self.current_prediction_text:
            sign_text = f"Sign: {self.current_prediction_text}"
            sign_size = cv2.getTextSize(sign_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(img,
                          (10, 10),
                          (10 + sign_size[0] + 20, 10 + sign_size[1] + 20),
                          (0, 0, 0), -1)
            cv2.rectangle(img,
                          (10, 10),
                          (10 + sign_size[0] + 20, 10 + sign_size[1] + 20),
                          (255, 255, 0), 2)
            cv2.putText(img, sign_text,
                        (20, 10 + sign_size[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Display NO SENTENCE ALERT on video (center)
        if self.show_alert:
            alert_text = "NO SENTENCE!"
            alert_text2 = "Please make a sign first"

            cv2.rectangle(img,
                          (img.shape[1] // 2 - 200, img.shape[0] // 2 - 60),
                          (img.shape[1] // 2 + 200, img.shape[0] // 2 + 60),
                          (0, 0, 139), -1)
            cv2.rectangle(img,
                          (img.shape[1] // 2 - 200, img.shape[0] // 2 - 60),
                          (img.shape[1] // 2 + 200, img.shape[0] // 2 + 60),
                          (0, 0, 255), 3)

            cv2.putText(img, alert_text,
                        (img.shape[1] // 2 - 130, img.shape[0] // 2 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(img, alert_text2,
                        (img.shape[1] // 2 - 180, img.shape[0] // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display Sentence (bottom-left)
        if self.sentence:
            sentence_text = f"Sentence: {' '.join(self.sentence)}"
            if len(sentence_text) > 40:
                sentence_text = sentence_text[:37] + "..."
            sent_size = cv2.getTextSize(sentence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img,
                          (10, img.shape[0] - sent_size[1] - 30),
                          (10 + sent_size[0] + 20, img.shape[0] - 10),
                          (0, 0, 0), -1)
            cv2.rectangle(img,
                          (10, img.shape[0] - sent_size[1] - 30),
                          (10 + sent_size[0] + 20, img.shape[0] - 10),
                          (0, 255, 0), 2)
            cv2.putText(img, sentence_text,
                        (20, img.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- Twilio Config ---
@st.cache_resource
def get_ice_servers():
    try:
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "ACd5cb71986c3f7771ddb2cc4a313376fa")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "145f7dc2179cf7dd6675505ceb8287e0")
        if Client:
            client = Client(account_sid, auth_token)
            token = client.tokens.create()
            return token.ice_servers
        return [
            {"urls": ["stun:global.stun.twilio.com:3478"]},
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    except Exception:
        return [
            {"urls": ["stun:global.stun.twilio.com:3478"]},
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]


# Two-column layout: Video on Left, Controls & Output on Right
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    st.markdown("### 📹 Live Camera Feed")
    if webrtc_streamer:
        ctx = webrtc_streamer(
            key="isl-predict",
            video_processor_factory=SignProcessor,
            rtc_configuration=RTCConfiguration({"iceServers": get_ice_servers()}),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "facingMode": "user"
                },
                "audio": False
            },
            async_processing=True
        )
    else:
        ctx = None
        st.warning("WebRTC streamer module is not available in this environment.")

with right_col:
    st.markdown("### 👁️ Hands-Free Blink Trigger")
    blink_placeholder = st.empty()

    st.markdown("### 📝 Building Sentence")
    sentence_placeholder = st.empty()
    alert_placeholder = st.empty()

    st.markdown("#### ⚙️ Sentence Controls")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        backspace_btn = st.button("⌫ Backspace", use_container_width=True)
        clear_btn = st.button("🗑️ Clear All", use_container_width=True)
    with btn_col2:
        reset_blinks_btn = st.button("🔄 Reset Blinks", use_container_width=True)
        manual_speak_btn = st.button("🔊 Speak Now", use_container_width=True)

if ctx and ctx.state.playing and ctx.video_processor:
    speak_this_frame = ""

    with ctx.video_processor.lock:
        if backspace_btn and ctx.video_processor.sentence:
            ctx.video_processor.sentence.pop()
        if clear_btn:
            ctx.video_processor.sentence = []
        if reset_blinks_btn:
            ctx.video_processor.detector.reset()
            ctx.video_processor.current_blink_count = 0
            ctx.video_processor.show_alert = False
            ctx.video_processor.speak_triggered = False
            ctx.video_processor.text_to_speak = ""

        # Handle manual speak button
        if manual_speak_btn:
            if ctx.video_processor.sentence:
                speak_this_frame = " ".join(ctx.video_processor.sentence)
                ctx.video_processor.sentence = []
                ctx.video_processor.captured_this_sign = False
                ctx.video_processor.current_prediction_text = ""
            else:
                ctx.video_processor.show_alert = True
                speak_this_frame = "No sentence to speak. Please make a sign first."

        # Get text to speak from blink trigger
        if not speak_this_frame and ctx.video_processor.text_to_speak:
            speak_this_frame = ctx.video_processor.text_to_speak
            ctx.video_processor.text_to_speak = ""
            ctx.video_processor.detector.reset()
            ctx.video_processor.current_blink_count = 0
            ctx.video_processor.speak_triggered = False
            ctx.video_processor.show_alert = False

        current_sentence = ctx.video_processor.sentence.copy()
        current_blinks = ctx.video_processor.current_blink_count
        show_alert = ctx.video_processor.show_alert

    # Audio playback with gTTS
    if speak_this_frame:
        try:
            st.toast(f"🔊 Speaking: {speak_this_frame}", icon="🗣️")
            if gTTS:
                tts = gTTS(text=speak_this_frame, lang='en', slow=False)
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                st.audio(audio_buffer, format="audio/mp3", autoplay=True)
        except Exception as tts_err:
            st.error(f"Audio Error: {tts_err}")

    # Render Visual Blink Progress
    blink_dots = ["⚪", "⚪", "⚪"]
    for i in range(min(current_blinks, 3)):
        blink_dots[i] = "🟢" if current_blinks >= 3 else ("🟠" if i == 1 else "🟡")
    dots_str = " ".join(blink_dots)

    blink_status = f"{current_blinks}/3 Blinks Completed"
    if current_blinks >= 3:
        blink_status = "🎉 3 Blinks Reached — Speaking Sentence!"

    blink_placeholder.markdown(f"""
    <div class="blink-card">
        <div style="font-size: 13px; color: #9aa2c2; text-transform: uppercase; letter-spacing: 1px;">Speech Trigger</div>
        <div class="blink-dots">{dots_str}</div>
        <div style="font-weight: 700; color: {'#00E676' if current_blinks >= 3 else '#f1f5f9'}; font-size: 15px;">{blink_status}</div>
    </div>
    """, unsafe_allow_html=True)

    # Render Sentence Box
    if current_sentence:
        sentence_html = f'<div class="sentence-display">{" ".join(current_sentence)}</div>'
        word_count = len(current_sentence)
    else:
        sentence_html = '<div class="sentence-display sentence-placeholder">Make a sign to start building sentence...</div>'
        word_count = 0

    sentence_placeholder.markdown(f"""
    <div class="sentence-card">
        <div class="sentence-title">Constructed Sentence ({word_count} words)</div>
        {sentence_html}
    </div>
    """, unsafe_allow_html=True)

    if show_alert:
        alert_placeholder.markdown(
            '<div class="alert-box">⚠️ No sentence to speak! Please make a sign first.</div>',
            unsafe_allow_html=True
        )
    else:
        alert_placeholder.empty()

    time.sleep(0.1)
    st.rerun()
else:
    blink_placeholder.markdown("""
    <div class="blink-card">
        <div class="blink-dots">⏸️</div>
        <div style="color: #64748b; font-size: 14px;">Camera is currently paused</div>
    </div>
    """, unsafe_allow_html=True)
    sentence_placeholder.markdown("""
    <div class="sentence-card">
        <div class="sentence-title">Constructed Sentence</div>
        <div class="sentence-display sentence-placeholder">Click START above to allow camera access and begin.</div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 13px; padding: 10px 0;">
    🤟 <strong>ISL Sign Language Recognition System</strong> • UFOS Training Center Bhavnagar • AI Internship 2026<br>
    Built with MediaPipe, scikit-learn, Streamlit WebRTC & Twilio
</div>
""", unsafe_allow_html=True)
