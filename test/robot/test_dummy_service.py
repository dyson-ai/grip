#!/usr/bin/env python3

import sys
import unittest
import pytest
from unittest.mock import Mock


try:
    import rclpy
    from rclpy.node import Node
    from std_srvs.srv import SetBool
    from rclpy.executors import SingleThreadedExecutor
except ImportError:
    Node = Mock


class DummyService(Node):
    def __init__(self):
        super().__init__("dummy_service")
        self.srv = self.create_service(
            SetBool, "activate_robot", self.callback_activate_robot
        )
        self.activated = False

    def callback_activate_robot(self, request, response):
        self.activated = request.data

        response.success = True

        if self.activated:
            response.message = "Robot is activated"
        else:
            response.message = "robot is deactivated"

        return response


class DummyClientAsync(Node):
    def __init__(self):
        super().__init__("dummy_client_async")
        self.cli = self.create_client(SetBool, "activate_robot")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")
        self.req = SetBool.Request()

    def send_request(self, actiavte):
        self.req.data = actiavte
        self.future = self.cli.call_async(self.req)
        return self.future


@pytest.mark.skipif("rclpy" not in sys.modules, reason="requires ROS")
class TestDummyService(unittest.TestCase):
    """Tests calling a dummy service"""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

        cls.service_node = DummyService()
        cls.service_client_node = DummyClientAsync()

        cls.executor = SingleThreadedExecutor()
        cls.executor.add_node(cls.service_client_node)
        cls.executor.add_node(cls.service_node)

    @classmethod
    def tearDownClass(cls):
        cls.service_node.destroy_node()
        cls.service_client_node.destroy_node()
        rclpy.shutdown()

    def test_service_call(self):
        # input flag
        activated = True

        # sends request
        future = self.service_client_node.send_request(activated)

        # executes until future is complete
        self.executor.spin_until_future_complete(future)

        # gets result
        response = future.result()
        print(f"Result: {response.message}")

        self.assertTrue(
            self.service_node.activated == activated,
            "Service call failed",
        )


if __name__ == "__main__":
    unittest.main()
