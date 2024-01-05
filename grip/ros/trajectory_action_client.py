import rclpy
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory


class TrajectoryActionClient:
    """A simple trajectory action client."""

    def __init__(
        self,
        parent_node: "rclpy.node.Node",
        action_name: str = "position_joint_trajectory_controller/follow_joint_trajectory",
    ):
        self.node = parent_node
        self._action_client = ActionClient(
            self.node,
            FollowJointTrajectory,
            action_name,
        )

    def send_trajectory_goal(self, goal_msg: FollowJointTrajectory.Goal):
        self._action_client.wait_for_server()

        self._action_client.send_goal_async(goal_msg)
