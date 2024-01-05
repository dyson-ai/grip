#!/usr/bin/env python3
import sys
import unittest
import pytest
from unittest.mock import Mock
import grip.robot as robot

try:
    import rclpy
    import sensor_msgs.msg as sensor_msgs
    from grip.ros.io import xacro_to_urdf_path
    from grip.ros import ROSRobotArm
except ImportError:
    pass


class Subscriber:
    """
    Subscribes to a topic
    """

    def __init__(self, topic: str, msg_type, node):
        self._sub = node.create_subscription(
            msg_type,
            topic=topic,
            callback=self._callback,
            qos_profile=10,
        )

        self._msgs = list()

    @property
    def num_messages(self) -> int:
        """Get the number of messages received"""
        return len(self._msgs)

    def _callback(self, msg) -> None:
        """Subscriber callback method"""
        self._msgs.append(msg)


@pytest.mark.skipif("rclpy" not in sys.modules, reason="requires ROS")
class TestROSArm(unittest.TestCase):
    """Tests if the ROS robot is publishing as expected."""

    N_PUB = 5

    @classmethod
    def setUpClass(cls):
        cls.world = robot.BulletWorld(phys_opt="direct")

        rclpy.init()

        cls.node = rclpy.create_node("test_ros_arm")

        urdf_path = xacro_to_urdf_path(
            "franka_description",
            "panda_arm.urdf.xacro",
            "/tmp/panda_arm.urdf",
            pkg_subfolder="robots",
            xacro_args="hand:=true",
        )

        cls.arm = ROSRobotArm(
            parent_node=cls.node,
            urdf_file=urdf_path,
            phys_id=cls.world.id,
            ee_mount_link="panda_link8",
            tip_link="panda_hand_tcp",
            gripper_type="panda_hand",
        )

        cls.joint_state_subscriber = Subscriber(
            cls.arm.joint_state_pub.topic_name, sensor_msgs.JointState, cls.node
        )

    @classmethod
    def tearDownClass(cls):
        cls.world.disconnect()
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_joint_state_publishing(self):
        # publishes a few times
        for _ in range(self.N_PUB):
            self.arm.joint_state_pub.execute()

            rclpy.spin_once(self.node, timeout_sec=0.1)

        self.assertGreaterEqual(
            self.joint_state_subscriber.num_messages,
            1,
            f"Failed to publish joint state in topic {self.arm.joint_state_pub.topic_name}",
        )


if __name__ == "__main__":
    unittest.main()
