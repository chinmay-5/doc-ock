
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Bool


class TeleopNode(Node):
    def __init__(self):
        super().__init__("hand_teleop")

        self.movement_pub = self.create_publisher(
            Vector3Stamped,
            "/teleop/delta",
            10,
        )
        self.enabled_pub = self.create_publisher(
            Bool,
            "/teleop/enabled",
            10,
        )
        self.gripper_pub = self.create_publisher(
            Bool,
            "/teleop/gripper_closed",
            10,
        )

    def publish_state(self, enabled, gripper_closed):
        enabled_msg = Bool()
        enabled_msg.data = bool(enabled)
        self.enabled_pub.publish(enabled_msg)

        gripper_msg = Bool()
        gripper_msg.data = bool(gripper_closed)
        self.gripper_pub.publish(gripper_msg)

    def publish_offset(self, offset):
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "hand_camera"

        msg.vector.x = float(offset[0])
        msg.vector.y = float(offset[1])
        msg.vector.z = float(offset[2])

        self.movement_pub.publish(msg)
