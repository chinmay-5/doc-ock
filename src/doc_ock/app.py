
import time
from pathlib import Path

import cv2
import rclpy

from doc_ock.vision.hand_tracker import HandTracker
from doc_ock.vision.gestures import GestureDetector
from doc_ock.vision.visualizer import HandVisualizer
from doc_ock.vision.retargeter import Retargeter
from doc_ock.ros.teleop_node import TeleopNode


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
STARTUP_TIME = time.perf_counter()

def startup_log(message):
    elapsed = time.perf_counter() - STARTUP_TIME
    print(f"[{elapsed:6.2f}s] {message}", flush=True)

def main():
    startup_log("Initializing ROS 2...")
    rclpy.init()
    node = TeleopNode()
    startup_log("ROS 2 node created")

    tracker = HandTracker(MODEL_PATH)
    startup_log("Hand tracker object created")

    gestures = GestureDetector()
    retargeter = Retargeter()
    visualizer = HandVisualizer()
    startup_log("Other components created")
    
    teleop_enabled = False
    previous_time = time.perf_counter()

    try:
        with tracker:
            while rclpy.ok():
                frame, landmarks = tracker.read()
                if frame is None:
                    node.get_logger().warning("Camera frame unavailable")
                    break

                rclpy.spin_once(node, timeout_sec=0.0)

                now = time.perf_counter()
                dt = now - previous_time
                previous_time = now
                fps = 1.0 / dt if dt > 0 else 0.0

                if landmarks is not None:
                    pinch_distance, gripper_closed = gestures.update(
                        landmarks
                    )
                else:
                    pinch_distance = 0.0
                    gestures.reset()
                    gripper_closed = False

                active = teleop_enabled and landmarks is not None

                if active:
                    offset = retargeter.update(landmarks)
                    if offset is not None:
                        node.publish_offset(offset)
                else:
                    retargeter.reset()

                node.publish_state(
                    active,
                    gripper_closed if active else False,
                )

                frame = visualizer.draw(
                    frame=frame,
                    landmarks=landmarks,
                    pinch_distance=pinch_distance,
                    gripper_closed=gripper_closed if active else False,
                    control_active=active,
                    fps=fps,
                )

                cv2.imshow("dr-octopus | Hand Teleoperation", frame)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

                if key == ord(" "):
                    teleop_enabled = not teleop_enabled
                    retargeter.reset()
                    node.get_logger().info(
                        "Teleoperation "
                        + ("enabled" if teleop_enabled else "paused")
                    )

                elif key in (ord("r"), ord("R")):
                    retargeter.reset()
                    node.get_logger().info("Reference position reset")

    except KeyboardInterrupt:
        pass
    finally:
        # Attempt to publish the inactive state before shutdown.
        if rclpy.ok():
            node.publish_state(False, False)
            rclpy.spin_once(node, timeout_sec=0.0)
            node.destroy_node()
            rclpy.shutdown()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
