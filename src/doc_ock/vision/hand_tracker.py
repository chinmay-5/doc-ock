
import time
from pathlib import Path

import cv2
import mediapipe as mp


class HandTracker:
    def __init__(
        self,
        model_path: Path,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
    ):
        self.model_path = Path(model_path)
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.landmarker = None
        self._start_time = None
        self._last_timestamp_ms = -1

    def __enter__(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Hand landmark model not found: {self.model_path}"
            )
        
        print("[startup] Opening webcam...", flush=True)
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            raise RuntimeError(
                f"Could not open webcam index {self.camera_index}"
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        print("[startup] Webcam opened.", flush=True)

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        try:
            print("[startup] Loading MediaPipe model...", flush=True)
            self.landmarker = (
                mp.tasks.vision.HandLandmarker.create_from_options(options)
            )
            print("[startup] MediaPipe model loaded.", flush=True)
        except Exception:
            self.cap.release()
            self.cap = None
            raise

        self._start_time = time.perf_counter()
        return self

    def read(self):
        """Return (frame, landmarks); landmarks is None when no hand is found.

        Each landmark is a normalized (x, y, z) tuple.
        The returned frame is mirrored for intuitive webcam interaction.
        """
        if self.cap is None or self.landmarker is None:
            raise RuntimeError("Use HandTracker inside a with statement")

        ok, frame = self.cap.read()
        if not ok:
            return None, None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        elapsed_ms = int(
            (time.perf_counter() - self._start_time) * 1000
        )
        timestamp_ms = max(elapsed_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(image, timestamp_ms)

        if not result.hand_landmarks:
            return frame, None

        landmarks = [
            (point.x, point.y, point.z)
            for point in result.hand_landmarks[0]
        ]
        return frame, landmarks

    def __exit__(self, exc_type, exc_value, traceback):
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        return False
