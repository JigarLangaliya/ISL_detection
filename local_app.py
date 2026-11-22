import os
import streamlit as st
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
except ImportError:
    webrtc_streamer = None
    class VideoProcessorBase: pass

import av
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

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
import platform
try:
    from blink_Detector import BlinkDetector
except ImportError:
    from blink_detector import BlinkDetector

st.set_page_config(page_title="ISL Live Prediction", layout="centered")

# Custom CSS
st.markdown("""
<style>
    .blink-counter {
        font-size: 48px;
        font-weight: bold;
        color: #ff4b4b;
        text-align: center;
        padding: 20px;
        background: #1e1e1e;
        border-radius: 15px;
        margin: 10px 0;
    }
    .sentence-text {
        font-size: 24px;
        color: #4CAF50;
        text-align: center;
        padding: 15px;
        background: #1e1e1e;
        border-radius: 10px;
        min-height: 60px;
    }
    .status-speaking {
        font-size: 20px;
        color: #FFA500;
        text-align: center;
        padding: 10px;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.5; }
        50% { opacity: 1; }
        100% { opacity: 0.5; }
    }
    .alert-box {
        font-size: 20px;
        color: #ff4444;
        text-align: center;
        padding: 15px;
        background: #2d0000;
        border-radius: 10px;
        border: 2px solid #ff4444;
        animation: shake 0.5s;
    }
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-10px); }
        75% { transform: translateX(10px); }
    }
</style>
""", unsafe_allow_html=True)

st.title("🤟 ISL — Live Sign Language Prediction")
st.caption("Make signs → Blink 3 times to speak → Repeat")


# --- Load Model ---
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    for candidate in ['isl_master_model.pkl', 'isl_final_fully_model.pkl']:
        p = os.path.join(BASE_DIR, candidate)
        if os.path.exists(p):
            return joblib.load(p)
    return joblib.load('isl_master_model.pkl')


model = load_model()

SIGN_LABELS = {
    'open_palm': 'Hello',
    'thumbs_up': 'Good',
    'thumbs_down': 'No',
    'pointing': 'Look',
    'fist': 'Stop',
    'thank_you': 'Thank You',
    'sorry': 'Sorry',
    'please': 'Please',
    'unhappy': 'Sad',
    'water': 'Water',
    'eat': 'Eat',
    'ily': 'I Love You',
    'cool': 'OK',
    'ok': 'OK',
}


def translate(label):
    return SIGN_LABELS.get(label, label.replace('_', ' ').title())


# --- Text-to-Speech Engine ---
def speak_sentence_threaded(text, processor):
    try:
        if platform.system() == "Windows":
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except ImportError:
                pass

        if pyttsx3:
            engine = pyttsx3.init()
            engine.setProperty('rate', 130)
            engine.setProperty('volume', 1.0)
            engine.say(text)
            engine.runAndWait()
    except Exception:
        pass
    finally:
        if platform.system() == "Windows":
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass
        with processor.lock:
            processor.is_speaking = False
            processor.detector.reset()
            processor.current_blink_count = 0


def speak_alert_threaded(text, processor):
    """Separate function for alert messages (no sentence)"""
    try:
        if platform.system() == "Windows":
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except ImportError:
                pass

        if pyttsx3:
            engine = pyttsx3.init()
            engine.setProperty('rate', 130)
            engine.setProperty('volume', 1.0)
            engine.say(text)
            engine.runAndWait()
    except Exception:
        pass
    finally:
        if platform.system() == "Windows":
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass
        with processor.lock:
            processor.is_speaking = False
            processor.show_alert = False
            processor.detector.reset()
            processor.current_blink_count = 0


# --- Mediapipe Config ---
if mp:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
else:
    mp_hands = None
    mp_drawing = None

STABLE_THRESHOLD = 3
CONFIDENCE_HIGH = 70.0


class SignProcessor(VideoProcessorBase):
    def __init__(self):
        if mp_hands:
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5
            )
        else:
            self.hands = None

        self.last_capture_time = 0
        self.COOLDOWN = 1.5

        self.detector = BlinkDetector()
        self.BLINKS_NEEDED = 3

        self.sentence = []
        self.last_prediction = None
        self.stable_count = 0
        self.captured_this_sign = False
        self.current_prediction_text = ""

        self.current_blink_count = 0

        self.lock = threading.Lock()
        self.is_speaking = False
        self.show_alert = False
        self.session_start_time = time.time()
        self.sign_history = []
        self.spoken_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # ── Reuse RGB Image ──
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ── Blink detection ──
        blink_count = self.detector.check(img, rgb_frame=rgb_img)
        if blink_count is None:
            blink_count = 0

        # ── Hand detection ──
        raw_prediction = None
        confidence = 0.0

        if self.hands:
            rgb_img.flags.writeable = False
            results = self.hands.process(rgb_img)
            rgb_img.flags.writeable = True

            if results.multi_hand_landmarks and model is not None:
                for hand_landmarks in results.multi_hand_landmarks:
                    if mp_drawing:
                        mp_drawing.draw_landmarks(
                            img, hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )

                    lm = hand_landmarks.landmark
                    wrist_x = lm[0].x
                    wrist_y = lm[0].y

                    coordinates = []
                    for i in range(21):
                        rel_x = lm[i].x - wrist_x
                        rel_y = lm[i].y - wrist_y
                        coordinates.extend([rel_x, rel_y])

                    def landmark_distance(lm, idx1, idx2):
                        x1, y1 = lm[idx1].x, lm[idx1].y
                        x2, y2 = lm[idx2].x, lm[idx2].y
                        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                    distances = [
                        landmark_distance(lm, 4, 8),
                        landmark_distance(lm, 8, 12),
                        landmark_distance(lm, 0, 8)
                    ]

                    features = coordinates + distances
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

            # Sign prediction stability
            if raw_prediction is not None:
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

            # Capture sign
            if (self.stable_count >= STABLE_THRESHOLD and raw_prediction is not None):
                current_time = time.time()
                human_text = translate(raw_prediction)
                self.current_prediction_text = human_text

                if (confidence >= CONFIDENCE_HIGH
                        and not self.captured_this_sign
                        and current_time - self.last_capture_time >= self.COOLDOWN):
                    self.sentence.append(human_text)
                    self.sign_history.append(human_text)
                    self.captured_this_sign = True
                    self.last_capture_time = current_time

            # ── Speech Trigger ──
            if self.current_blink_count >= self.BLINKS_NEEDED:
                if self.sentence and not self.is_speaking:
                    # HAVE SENTENCE - Speak it!
                    text_to_speak = " ".join(self.sentence)
                    self.is_speaking = True
                    self.sentence = []
                    self.spoken_count += 1
                    self.captured_this_sign = False
                    self.current_prediction_text = ""

                    speech_thread = threading.Thread(
                        target=speak_sentence_threaded,
                        args=(text_to_speak, self),
                        daemon=True
                    )
                    speech_thread.start()

                elif not self.sentence and not self.is_speaking:
                    # NO SENTENCE - Show alert and speak warning!
                    self.is_speaking = True
                    self.show_alert = True

                    alert_message = "No sentence to speak. Please make a sign first."

                    alert_thread = threading.Thread(
                        target=speak_alert_threaded,
                        args=(alert_message, self),
                        daemon=True
                    )
                    alert_thread.start()

        # ── DISPLAY ON VIDEO ──
        # Display Blinks counter (top-right)
        blink_text = f"Blinks: {self.current_blink_count}/3"
        text_size = cv2.getTextSize(blink_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(img,
                      (img.shape[1] - text_size[0] - 30, 10),
                      (img.shape[1] - 10, 10 + text_size[1] + 20),
                      (0, 0, 0), -1)
        cv2.rectangle(img,
                      (img.shape[1] - text_size[0] - 30, 10),
                      (img.shape[1] - 10, 10 + text_size[1] + 20),
                      (0, 255, 0), 2)
        cv2.putText(img, blink_text,
                    (img.shape[1] - text_size[0] - 20, 10 + text_size[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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


# --- Streamlit UI ---
if webrtc_streamer:
    ctx = webrtc_streamer(
        key="isl-predict",
        video_processor_factory=SignProcessor,
        media_stream_constraints={
            "video": {"width": 640, "height": 480},
            "audio": False
        },
        async_processing=True
    )
else:
    ctx = None

st.divider()

# Main UI Layout
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    st.markdown("### 👁️ Blinks")
    blink_placeholder = st.empty()

st.markdown("### 📝 Your Sentence")
sentence_placeholder = st.empty()

# Alert placeholder
alert_placeholder = st.empty()

# Control buttons
col1, col2, col3 = st.columns(3)
with col1:
    backspace_btn = st.button("⌫ Backspace", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear All", use_container_width=True)
with col3:
    reset_blinks_btn = st.button("🔄 Reset Blinks", use_container_width=True)

# ── Live UI Updater (Non-blocking, Reactive) ──
if ctx and ctx.state.playing and ctx.video_processor:
    with ctx.video_processor.lock:
        if backspace_btn and ctx.video_processor.sentence:
            ctx.video_processor.sentence.pop()

        if clear_btn:
            ctx.video_processor.sentence = []

        if reset_blinks_btn:
            ctx.video_processor.detector.reset()
            ctx.video_processor.current_blink_count = 0
            ctx.video_processor.show_alert = False

        current_sentence = ctx.video_processor.sentence.copy()
        is_speaking = ctx.video_processor.is_speaking
        current_blinks = ctx.video_processor.current_blink_count
        show_alert = ctx.video_processor.show_alert

    # Update blink counter
    blink_color = "#ff4b4b" if current_blinks > 0 else "#666666"
    blink_bg = "#2d0000" if current_blinks > 0 else "#1e1e1e"

    blink_placeholder.markdown(f"""
        <div style="
            font-size: 60px;
            font-weight: bold;
            color: {blink_color};
            text-align: center;
            padding: 20px;
            background: {blink_bg};
            border-radius: 20px;
            border: 3px solid {blink_color};
            box-shadow: 0 0 20px {blink_color}40;
        ">
            {current_blinks}/3
        </div>
    """, unsafe_allow_html=True)

    # Show alert if no sentence
    if show_alert:
        alert_placeholder.markdown("""
            <div class="alert-box">
                ⚠️ No sentence to speak! Please make a sign first.
            </div>
        """, unsafe_allow_html=True)
    else:
        alert_placeholder.empty()

    # Update sentence display
    if is_speaking:
        sentence_placeholder.markdown("""
            <div style="
                font-size: 24px;
                color: #FFA500;
                text-align: center;
                padding: 15px;
                background: #1e1e1e;
                border-radius: 10px;
                animation: pulse 1s infinite;
            ">
                🔊 Speaking...
            </div>
        """, unsafe_allow_html=True)
    else:
        display_text = " ".join(current_sentence) if current_sentence else "Sign something..."
        sentence_placeholder.markdown(f"""
            <div style="
                font-size: 24px;
                color: #4CAF50;
                text-align: center;
                padding: 15px;
                background: #1e1e1e;
                border-radius: 10px;
                min-height: 60px;
            ">
                {display_text}
            </div>
        """, unsafe_allow_html=True)

    time.sleep(0.1)
    st.rerun()
else:
    blink_placeholder.markdown("""
        <div style="
            font-size: 40px;
            color: #666666;
            text-align: center;
            padding: 20px;
        ">
            ⏸️
        </div>
    """, unsafe_allow_html=True)
    sentence_placeholder.info("Click START above to begin.")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #888888;">
    <p>💡 <strong>How to use:</strong> Make signs → Blink 3 times → Sentence is spoken</p>
    <p>⚠️ If no sentence: You'll hear "No sentence to speak. Please make a sign first."</p>
    <p>👁️ Blink counter persists until you complete 3 blinks or manually reset</p>
</div>
""", unsafe_allow_html=True)
