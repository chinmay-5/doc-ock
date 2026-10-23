
import math


class GestureDetector:
    def __init__(
        self,
        close_threshold: float = 0.05,
        open_threshold: float = 0.07,
    ):
        if close_threshold >= open_threshold:
            raise ValueError(
                "close_threshold must be less than open_threshold"
            )

        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.gripper_closed = False

    @staticmethod
    def pinch_distance(landmarks):
        """3D normalized distance between thumb tip (4) and index tip (8)."""
        thumb = landmarks[4]
        index = landmarks[8]
        return math.sqrt(sum(
            (thumb[i] - index[i]) ** 2 for i in range(3)
        ))

    def update(self, landmarks):
        """Return the current (pinch_distance, gripper_closed) state."""
        distance = self.pinch_distance(landmarks)

        if distance < self.close_threshold:
            self.gripper_closed = True
        elif distance > self.open_threshold:
            self.gripper_closed = False

        return distance, self.gripper_closed

    def reset(self):
        self.gripper_closed = False
