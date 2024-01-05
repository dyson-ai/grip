#!/usr/bin/python
import sys
import unittest
import pytest
import numpy as np
import grip
import grip.robot as robot


try:
    import rclpy
    import sensor_msgs.msg as sensor_msgs
    from grip.ros import ROSCamera
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
class TestROSRGBDCamera(unittest.TestCase):
    """Tests if the ROS camera is publishing as expected."""

    @classmethod
    def setUpClass(cls):
        cls.world = robot.BulletWorld(phys_opt="direct")

        rclpy.init()

        cls.node = rclpy.create_node("test_ros_camera")

        # creates a simulated camera by setting sim=True
        cls.sim_camera = ROSCamera(
            name="top_camera",
            cid=cls.world.id,
            position=np.array([0, 0, 1.5]),
            orientation=grip.math.rpy2quaternion([0.0, np.pi, 0.0]),
            parent_node=cls.node,
            sim=True,  # It's a simulated camera
        )

        cls.image_subscriber = Subscriber(
            cls.sim_camera.image_topic, sensor_msgs.Image, cls.node
        )
        cls.depth_subscriber = Subscriber(
            cls.sim_camera.depth_topic, sensor_msgs.Image, cls.node
        )
        cls.camera_info_subscriber = Subscriber(
            cls.sim_camera.image_info_topic, sensor_msgs.CameraInfo, cls.node
        )

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_acquire_observation(self):
        rgb, depth = self.sim_camera.obs()

        # Spins 3 times, so the 3 messages are published: rgb, depth and camera info
        for _ in range(3):
            rclpy.spin_once(self.node)

        self.assertEqual(
            self.image_subscriber.num_messages, 1, "Failed to publish RGB image"
        )
        self.assertEqual(
            self.depth_subscriber.num_messages, 1, "Failed to publish Depth image"
        )
        self.assertEqual(
            self.camera_info_subscriber.num_messages, 1, "Failed to publish Camera info"
        )
        self.assertTrue(rgb is not None, "Error: colour image is none")
        self.assertTrue(depth is not None, "Error: depth image is none")


if __name__ == "__main__":
    unittest.main()
