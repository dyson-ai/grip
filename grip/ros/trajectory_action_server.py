#! /usr/bin/env python
import rclpy
from rclpy.action import ActionServer
from control_msgs.action import FollowJointTrajectory
import grip
from ..motion import Path
import time


class TrajectoryActionServer:
    """Grip Bullet Sim trajectory action server
    Execute trajectory actions

    Args:
        parent_node (rclpy.node.Node): parent node
    """

    def __init__(self, parent_node: "rclpy.node.Node", robot: "grip.robot.BulletRobot"):
        self.node = parent_node
        self.robot = robot

        self.callback_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        self._action_server = ActionServer(
            self.node,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
            self.execute_callback,
            callback_group=self.callback_group,
        )

    def follow_joint_path(
        self, trajectory: FollowJointTrajectory.Goal, dt: float = 0.01
    ):
        if self.robot.sim:
            trajectory = Path.from_ros_trajectory_action_goal(trajectory)

            for wpt in trajectory:
                self.robot.set_angles(wpt, joint_names=trajectory.joint_names)
                time.sleep(dt)

    def execute_callback(
        self, goal_handle: rclpy.action.server.ServerGoalHandle
    ) -> FollowJointTrajectory.Result:
        """Executes trajectory action request

        Returns:
            FollowJointTrajectory.Result: result of trajectory execution
        """
        self.follow_joint_path(goal_handle.request, dt=0.001)
        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL

        return result

    def destroy(self) -> None:
        """Destroy the underlying action server handle."""

        # action server seems to require explicit destroy call (as per 14 March 2023)
        # PRs maybe related to this are currently in rolling distro (not in humble yet):
        # see:
        # https://github.com/ros2/rclpy/pull/1070
        # https://github.com/ros2/rclpy/issues/1034
        self._action_server.destroy()
        self.robot = None
        self.node = None
