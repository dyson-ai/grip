import rclpy
from sensor_msgs.msg import JointState
from typing import Callable, List
import numpy as np
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup


class JointStatePub:
    """A joint state publisher class responsible for publishing all joint states of a grip.robot.BulletRobot."""

    def __init__(
        self,
        parent_node: "rclpy.node.Node",
        robot: "grip.robot.BulletRobot",
        topic_name: str = "/joint_states",
        command_topic_name: str = "/joint_commands",
        **kwargs,
    ):
        """Constructs a JointStatePub object

        Args:
            parent_node (rclpy.node.Node): parent node to which this joint state publisher is associated with.
            robot (grip.robot.BulletRobot): robot from which joint states should be published.
            topic_name (str, optional): the desired topic name that the joint states should be published to. Defaults to "/joint_states".
        """

        self.lock = kwargs.get("lock", None)
        self.topic_name = topic_name
        self.robot = robot
        self.node = parent_node

        # register this node in the network as a publisher in /joint_states topic
        self.pub_joint_states = self.node.create_publisher(JointState, topic_name, 10)

        self.joint_command_sub = self.node.create_subscription(
            JointState,
            command_topic_name,
            self.joint_command_callback,
            1,
        )

        self.joint_msg = JointState()
        self.joint_msg.name = [name for name, _ in self.robot.joint_dict.items()]
        self._timer_func = None
        self._last_joint_command = None

    def execute(self) -> None:
        """Single step execution: reads current joint states and publishes it."""

        if self.lock is not None:
            with self.lock:
                self._execute()
        else:
            self._execute()

    def _execute(self) -> None:
        """Single step execution: reads current joint states and publishes it.

        This function gets called from pybullet ros main update loop
        """
        # setup msg placeholder
        joint_msg = JointState()
        # get joint states
        for joint_name, joint_info in self.robot.joint_dict.items():
            # get joint state from pybullet

            joint_state = self.robot.joint_angle(joint_info["idx"])
            # fill msg
            joint_msg.name.append(joint_name)
            joint_msg.position.append(joint_state[0])
            joint_msg.velocity.append(joint_state[1])
            joint_msg.effort.append(joint_state[2])  # applied effort in last sim step

        # update msg time using ROS time api
        joint_msg.header.stamp = self.node.get_clock().now().to_msg()
        # publish joint states to ROS
        self.pub_joint_states.publish(joint_msg)

        self.command_update()

    def _publisher_callback(self) -> None:
        """Publisher callback, used for asynchronous publishing using timers."""
        try:
            self.execute()
        except Exception as e:
            self.node.get_logger().warning(
                f"Async joint state publisher failed to publish joint state. Message {e}",
                once=True,
            )

    def setup_timer_publisher(self, rate: int = 10) -> None:
        """Sets up a timer function that periodically publishes joint states at specified rate.

        Args:
            rate (int, optional): desired publishing rate. Defaults to 10.
        """
        self._timer_cb_group = MutuallyExclusiveCallbackGroup()
        self._timer_func = self.node.create_timer(
            1.0 / rate, self._publisher_callback, callback_group=self._timer_cb_group
        )

    def stop_timer_publisher(self) -> None:
        """Stops the timer function"""

        if self._timer_func is None:
            return

        self._timer_func.destroy()
        self._timer_func = None

    def joint_command_callback(self, msg: JointState) -> None:
        """Callback for the joint command subscriber."""
        # print("Received command: ", msg)
        self._last_joint_command = msg

    def command_update(self):
        if self._last_joint_command is not None and np.any(
            self._last_joint_command.position
        ):
            self.robot.set_angles(
                self._last_joint_command.position,
                joint_names=self._last_joint_command.name,
            )

    def destroy(self) -> None:
        """Destroy the underlying ROS handles."""

        self.stop_timer_publisher()
        self.pub_joint_states.destroy()
        self.pub_joint_states = None
        self.robot = None
        self.node = None


class LocalJointStatePub(JointStatePub):
    """Publishes subset of joint states as reported by robot.BulletRobot joint_names and get_joint_states"""

    def __init__(
        self,
        parent_node: "rclpy.node.Node",
        robot: "grip.robot.BulletRobot",
        topic_name: str = "/gripper/joint_states",
        command_topic_name: str = "/gripper/joint_commands",
        **kwargs,
    ):
        super().__init__(parent_node, robot, topic_name, command_topic_name, **kwargs)

        self.joint_msg = JointState()
        self.joint_msg.name = self.robot.joint_names
        self._timer_func = None
        self._last_joint_command = None

    def _execute(self) -> None:
        """Single step execution: reads current joint states and publishes it.

        This function gets called from pybullet ros main update loop
        """
        # setup msg placeholder
        joint_msg = JointState()

        joint_names = self.robot.joint_names
        position, velocity, _, effort = self.robot.get_joint_state()

        joint_msg.name = joint_names
        joint_msg.position = position.tolist()
        joint_msg.velocity = velocity.tolist()
        joint_msg.effort = effort.tolist()

        # update msg time using ROS time api
        joint_msg.header.stamp = self.node.get_clock().now().to_msg()
        # publish joint states to ROS
        self.pub_joint_states.publish(joint_msg)

        self.command_update()


class JointStateSub:
    """A joint state subscriber class responsible for subscribing to the joint state of an external robot and mirroring this state to a grip.robot.BulletRobot instance."""

    def __init__(
        self,
        parent_node: "rclpy.node.Node",
        topic_name: str = "/joint_states",
        joint_update_callback: Callable[
            [np.ndarray, List[str]], None
        ] = lambda _, __: None,
        **kwargs,
    ):
        """Constructs a JointStatePub object

        Args:
            parent_node (rclpy.node.Node): parent node to which this joint state subscriber is associated with.
            topic_name (str, optional): the desired joint state topic name to subscribe to. Defaults to "/joint_states".
            joint_update_callback (Callable[[np.ndarray, List[str]], None]): callback used for updating joint states. Defaults to empty lambda.
        """

        self.lock = kwargs.get("lock", None)
        self.topic_name = topic_name
        self.node = parent_node

        # register this node in the network as a publisher in /joint_states topic
        self.sub_joint_states = self.node.create_subscription(
            JointState, self.topic_name, self.on_joint_state, 1
        )

        self._positions = None
        self._joint_names = None
        self._joint_update_callback = joint_update_callback

    def _register_joint_update_callback(
        self, joint_update_callback: Callable[[np.ndarray, List[str]], None]
    ) -> None:
        self._joint_update_callback = joint_update_callback

    def on_joint_state(self, joint_state: JointState) -> None:
        """Single step execution: receives current joint states and mirroes it to simulated robot.

        Args:

            joint_state: incoming joint state.
        """

        self._positions = np.array(joint_state.position)
        self._joint_names = joint_state.name

        # Sets simulated robot to same joint states received from external robot
        self._joint_update_callback(self._positions, self._joint_names)

    @property
    def positions(self) -> np.ndarray:
        """
        Current joint positions

        Returns:
            (np.ndarray): most recently received joint angles
        """

        return self._positions

    @property
    def joint_names(self) -> List[str]:
        """
        Get received joint names

        Returns:
            (List[str]): most recently received joint names
        """

        return self._joint_names

    def destroy(self):
        """Destroy the underlying ROS handles."""
        self.sub_joint_states.destroy()
        self.sub_joint_states = None
