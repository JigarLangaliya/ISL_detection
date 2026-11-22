import cv2
try:
    import mediapipe as mp
except ImportError:
    mp = None

import math
import time


class BlinkDetector:

    def __init__(self):
        if mp:
            self.mp_face = mp.solutions.face_mesh
            self.face_mesh = self.mp_face.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.mp_face = None
            self.face_mesh = None

        # Left eye landmark indices
        self.L_TOP = 159
        self.L_BOTTOM = 145
        self.L_LEFT = 33
        self.L_RIGHT = 133

        # Right eye landmark indices
        self.R_TOP = 386
        self.R_BOTTOM = 374
        self.R_LEFT = 362
        self.R_RIGHT = 263

        # Thresholds
        self.CLOSE_THRESHOLD = 0.20
        self.OPEN_THRESHOLD = 0.25

        self.eye_closed = False
        self.blink_count = 0
        self.current_ear = 0.0

        # Timing
        self.last_blink_time = 0.0
        self.BLINK_COOLDOWN = 0.5  # Half second between blinks to prevent double-counting
        self.last_face_seen_time = time.time()

    def _distance(self, p1, p2, width=1.0, height=1.0):
        # Scale by width and height to remove aspect ratio distortion
        return math.sqrt(
            ((p2.x - p1.x) * width) ** 2 +
            ((p2.y - p1.y) * height) ** 2
        )

    def _ear(self, top, bottom, left, right, width=1.0, height=1.0):
        vertical = self._distance(top, bottom, width, height)
        horizontal = self._distance(left, right, width, height)

        if horizontal == 0:
            return 0.0

        return vertical / horizontal

    def check(self, frame, rgb_frame=None):
        if not self.face_mesh:
            return self.blink_count

        current_time = time.time()
        h, w = frame.shape[:2]

        if rgb_frame is not None:
            rgb = rgb_frame
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            # Face lost - reset eye closure state to avoid false blink on recovery
            self.eye_closed = False
            # Wait 5 seconds before resetting blink count if face completely left
            if current_time - self.last_face_seen_time > 5.0:
                self.blink_count = 0
                self.last_blink_time = 0.0
            return self.blink_count

        # Face is visible - update timer
        self.last_face_seen_time = current_time

        face = results.multi_face_landmarks[0]
        lm = face.landmark

        # Calculate EAR for both eyes with aspect-ratio correction
        left_ear = self._ear(
            lm[self.L_TOP], lm[self.L_BOTTOM],
            lm[self.L_LEFT], lm[self.L_RIGHT],
            w, h
        )
        right_ear = self._ear(
            lm[self.R_TOP], lm[self.R_BOTTOM],
            lm[self.R_LEFT], lm[self.R_RIGHT],
            w, h
        )

        self.current_ear = (left_ear + right_ear) / 2.0

        # ── BLINK STATE MACHINE (NO AUTO-RESET) ──
        if not self.eye_closed:
            # Eye is OPEN - check if it just closed
            if self.current_ear < self.CLOSE_THRESHOLD:
                self.eye_closed = True
        else:
            # Eye is CLOSED - check if it just opened
            if self.current_ear > self.OPEN_THRESHOLD:
                self.eye_closed = False

                # Check cooldown to prevent double-counting
                time_since_last_blink = current_time - self.last_blink_time

                if time_since_last_blink > self.BLINK_COOLDOWN:
                    # Valid blink detected
                    self.blink_count += 1
                    self.last_blink_time = current_time

                    # Cap at 3 to prevent overflow
                    if self.blink_count > 3:
                        self.blink_count = 3

        return self.blink_count

    def reset(self):
        """Reset blink counter - called after speech or manually."""
        self.blink_count = 0
        self.eye_closed = False
        self.last_blink_time = 0.0
