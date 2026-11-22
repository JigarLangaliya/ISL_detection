import os
import time
import math
import threading
import cv2
try:
    import mediapipe as mp
except ImportError:
    mp = None

import joblib
import numpy as np
import av
import streamlit as st
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
except ImportError:
    webrtc_streamer = None
    class VideoProcessorBase: pass

st.title("ISL — Live Prediction In Browser")


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
    'open_palm'   : 'Hello',
    'thumbs_up'   : 'Good',
    'thumbs_down' : 'No',
    'pointing'    : 'Look',
    'fist'        : 'Stop',
    'thank_you'   : 'Thank You',
    'sorry'       : 'Sorry',
    'please'      : 'Please',
    'unhappy'     : 'Sad',
    'water'       : 'Water',
    'eat'         : 'Eat',
    'ily'         : 'I Love You',
    'cool'        : 'OK',
    'ok'          : 'OK',
}


def translate(label):
    return SIGN_LABELS.get(label, label.replace('_', ' ').title())


if mp:
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
else:
    mp_hands = None
    mp_drawing = None

STABLE_THRESHOLD = 5
CONFIDENCE_HIGH  = 80.0
COOLDOWN_TIME    = 1.5


class SignProcessor(VideoProcessorBase):

    def __init__(self):
        if mp_hands:
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
        else:
            self.hands = None
        self.sentence            = []
        self.last_prediction     = None
        self.stable_count        = 0
        self.captured_this_sign  = False
        self.last_capture_time   = 0.0
        self.lock                = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        raw_prediction = None
        confidence     = 0.0

        if self.hands:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

                    lm      = hand_landmarks.landmark
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
                        return math.sqrt((x2-x1)**2 + (y2-y1)**2)

                    distances = [
                        landmark_distance(lm, 4, 8),
                        landmark_distance(lm, 8, 12),
                        landmark_distance(lm, 0, 8)
                    ]

                    features      = coordinates + distances
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

        # ── Use the LOCK before touching shared memory ──
        with self.lock:
            if raw_prediction is not None:
                if raw_prediction == self.last_prediction:
                    self.stable_count += 1
                else:
                    self.stable_count       = 0
                    self.last_prediction    = raw_prediction
                    self.captured_this_sign = False
            else:
                self.stable_count       = 0
                self.last_prediction    = None
                self.captured_this_sign = False

            if (self.stable_count >= STABLE_THRESHOLD and raw_prediction is not None):
                display_text = f"{translate(raw_prediction)} ({confidence:.0f}%)"
                cv2.putText(img, display_text,
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 3)

                current_time = time.time()
                if (confidence >= CONFIDENCE_HIGH
                        and not self.captured_this_sign
                        and (current_time - self.last_capture_time >= COOLDOWN_TIME)):

                    human_text = translate(raw_prediction)
                    self.sentence.append(human_text)
                    self.captured_this_sign = True
                    self.last_capture_time = current_time

        return av.VideoFrame.from_ndarray(img, format="bgr24")


if webrtc_streamer:
    ctx = webrtc_streamer(
        key="isl-predict",
        video_processor_factory=SignProcessor
    )
else:
    ctx = None

st.divider()
st.subheader("Your Sentence")

if ctx and ctx.video_processor:
    with ctx.video_processor.lock:
        current_sentence = ctx.video_processor.sentence.copy()

    st.write(" ".join(current_sentence) if current_sentence else "Sign something...")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Backspace"):
            with ctx.video_processor.lock:
                if ctx.video_processor.sentence:
                    ctx.video_processor.sentence.pop()

    with col2:
        if st.button("Clear"):
            with ctx.video_processor.lock:
                ctx.video_processor.sentence = []
else:
    st.write("Click START above to begin.")
