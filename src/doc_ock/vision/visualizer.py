
import cv2
import mediapipe as mp


class HandVisualizer:
    def __init__(self):
        self.connections = (
            mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        )

    def draw(
        self,
        frame,
        landmarks,
        pinch_distance,
        gripper_closed,
        control_active,
        fps,
    ):
        if landmarks is not None:
            height, width = frame.shape[:2]
            points = [
                (int(x * width), int(y * height))
                for x, y, _ in landmarks
            ]

            for connection in self.connections:
                cv2.line(
                    frame,
                    points[connection.start],
                    points[connection.end],
                    (255, 180, 0),
                    2,
                )

            for point in points:
                cv2.circle(frame, point, 4, (0, 255, 0), -1)

            wrist = landmarks[0]
            cv2.putText(
                frame,
                f"Wrist: {wrist[0]:.2f}, {wrist[1]:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Pinch: {pinch_distance:.3f} | "
                f"Gripper: {'CLOSED' if gripper_closed else 'OPEN'}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "No hand detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        status = "CONTROL ACTIVE" if control_active else "CONTROL PAUSED"
        cv2.putText(
            frame,
            f"{status} | SPACE: toggle | R: re-anchor | Q: quit",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        return frame
