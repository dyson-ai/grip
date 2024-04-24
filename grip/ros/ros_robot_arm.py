from typing import Union
from .joint_state_tools import JointStatePub, JointStateSub, LocalJointStatePub
from .trajectory_action_server import TrajectoryActionServer
from .trajectory_action_client import TrajectoryActionClient
from control_msgs.action import FollowJointTrajectory
import time

from ..robot import BulletRobotArm
from ..motion import Path
from .io import is_sim


class ROSRobotArm(BulletRobotArm):
    """A ROS robot that specialises a BulletRobotArm representing a robot arm which can optionally have a gripper as an end-effector. It exposes standard ROS interfaces mimicking and exposing the same basic topics and actions of a real robot. It is also able to also connect to a real robot.
    Represents a simulated or a real robot arm. It dynamically detects whether or not it should simulate or simply connect to a robot if a robot already exists.
    When connected to a real robot, the simulated robot will act as a slave, mirroing the state of the real robot.
    """

    def __init__(
        self, parent_node: "rclpy.node.Node", auto_sim=True, sim=True, **kwargs
    ):
        """ROS Robot Arm constructor

        Args:
            parent_node (rclpy.node.Node): associated parent node.
            gripper_type (str, optional): gripper type name from grip.robot.EndEffectorRegistry to be used as the robot end-effector. Defaults to None, meaning no gripper is created.
            gripper_max_force (float, optional): maximum gripper force. Defaults to 40N.
            urdf_file (str): full path to a robot URDF description
            id (int): if the robot entity already exists in the physics engine, then the id of the robot can be passed instead of its urdf_file
            parent (grip.robot.BulletRobot, optional): This robot may be a sub-part of an already existing parent robot (e.g. a robot hand attached to a robot arm)
            phys_id (int): physics engine identifier (the id of a BulletWorld)
            use_fixed_base (bool): Whether or not to make the object fixed in place. Default: True.
            position (numpy.ndarray): shapebase position of the robot
            orientation (numpy.ndarray): base orientation of the robot as en orientation quaterion following JPL format [x, y, z, w]
            ee_index (int, optional): end-effector link id. Defaults to None. Note: either ee_index or tip_link must be set, but not both.
            tip_link (str, optional): end-effector tip link name. Specifies the end-effector tip link, if ee_index is None, then ee_index = self.get_link_id(tip_link). Defaults to None.
            Note: either ee_index or tip_link must be set, but not both.
            ee_mount_link (str, optional): end-effector mount link name, used to specify up at which link the end-effector has been mounted on the robot.
            joint_names (str, optional): joint names that this robot instance should be controlling (can be a subset of the total number of joints of the robot)
            if not passed as a parameter, or None is given, then this BulletRobot will use all joint names as defined in the urdf description
            has_kinematics (bool): enable or disable forward and inverse kinematics with this robot (this should be usually true, but some robots may not need it)
            max_force (float): maximum generalised force
            sliders (bool): whether or not joint sliders should be added to the debug GUI (only used if grip.robot.BulletWorld was created with phys_opt="gui")
            ee_sliders (bool): whether or not cartesian end-effector sliders should be added to the debug GUI (only used if grip.robot.BulletWorld was created with phys_opt="gui")
            auto_sim (bool, optional): automatically detects if a real robot exists in the network and connects to it if true, this will yield self.sim=False the argument sim is ignored. Defaults True.
            sim (bool, optional): when auto_sim=False, choose whether this instance will be a simulation instance of a connection to the real robot which will mirror the robot state.
            enable_action_server (bool, optional): whether or not internal joint trajectory action server will be enabled.
        """
        super().__init__(**kwargs)

        self.node = parent_node

        self.sim = is_sim(self.node) if auto_sim else sim

        self._last_joint_command = None

        self.state_topic = kwargs.setdefault("state_topic", "/joint_states")
        self.command_topic = kwargs.setdefault("command_topic", "/joint_commands")
        self.enable_action_server = kwargs.setdefault("enable_action_server", True)
        self.enable_pub = kwargs.setdefault("enable_pub", True)

        if self.sim:
            # If we are in simulation we need to simulate the joint state publisher and the trajectory action server.

            if self.enable_pub:
                self.joint_state_pub = JointStatePub(
                    self.node,
                    self,
                    topic_name=self.state_topic,
                    command_topic_name=self.command_topic,
                )

                if self.gripper:
                    self.gripper_joint_state_pub = LocalJointStatePub(
                        self.node,
                        self.gripper,
                    )

                if self.enable_action_server:
                    self.trajectory_action_server = TrajectoryActionServer(
                        self.node, self
                    )

        else:
            self.joint_state_sub = JointStateSub(
                self.node,
                topic_name=self.state_topic,
                joint_update_callback=self.set_angles,
            )
            self.trajectory_client = TrajectoryActionClient(self.node)

    def start_async_ros_comm(self, rate: int = 10):
        self.joint_state_pub.setup_timer_publisher(rate)

        if self.gripper:
            self.gripper_joint_state_pub.setup_timer_publisher(rate)

    def follow_joint_path(
        self, trajectory: Union[Path, FollowJointTrajectory.Goal], dt: float = 0.01
    ):
        if self.sim:
            if isinstance(trajectory, FollowJointTrajectory.Goal):
                trajectory = Path.from_ros_trajectory_action_goal(trajectory)

            for wpt in trajectory:
                self.set_angles(wpt, joint_names=trajectory.joint_names)
                time.sleep(dt)
        else:
            if isinstance(trajectory, Path):
                trajectory = trajectory.as_ros_trajectory_action_goal()

            self.trajectory_client.send_trajectory_goal(trajectory)
