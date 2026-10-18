
import time
from pathlib import Path

import cv2
import mediapipe as mp


MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Download hand_landmarker.task into the models folder."
        )

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Try camera index 1."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    connections = (
        mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
    )

    previous_time = time.perf_counter()
    start_time = time.perf_counter()
    frame_count = 0

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                # VIDEO mode requires monotonically increasing timestamps.
                timestamp_ms = int(
                    (time.perf_counter() - start_time) * 1000
                )

                result = landmarker.detect_for_video(
                    image, timestamp_ms
                )

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]

                    # Draw all 21 landmarks and their connections.
                    h, w = frame.shape[:2]
                    points = []
                    for landmark in landmarks:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        points.append((x, y))
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                    for connection in connections:
                        start = points[connection.start]
                        end = points[connection.end]
                        cv2.line(frame, start, end, (255, 180, 0), 2)

                    wrist = landmarks[0]
                    thumb_tip = landmarks[4]
                    index_tip = landmarks[8]

                    pinch_distance = (
                        (thumb_tip.x - index_tip.x) ** 2
                        + (thumb_tip.y - index_tip.y) ** 2
                        + (thumb_tip.z - index_tip.z) ** 2
                    ) ** 0.5

                    cv2.putText(
                        frame,
                        f"Wrist: {wrist.x:.2f}, {wrist.y:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0), 2,
                    )
                    cv2.putText(
                        frame,
                        f"Pinch distance: {pinch_distance:.3f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0), 2,
                    )
                    cv2.putText(
                        frame, "HAND TRACKED", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0), 2,
                    )
                else:
                    cv2.putText(
                        frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2,
                    )

                now = time.perf_counter()
                dt = now - previous_time
                previous_time = now
                fps = 1.0 / dt if dt > 0 else 0.0

                cv2.putText(
                    frame, f"FPS: {fps:.1f}",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2,
                )

                cv2.imshow("dr-octopus | Hand Tracking", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
