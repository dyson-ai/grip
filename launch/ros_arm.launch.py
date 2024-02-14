import os

from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue

from grip.ros.depth_proc_launch import depth_proc_launch
import pybullet_data as pd


def generate_launch_description():
    rviz_file = os.path.join(
        get_package_share_directory("grip"), "launch", "rviz/view_panda_d415.rviz"
    )

    franka_xacro_file = os.path.join(
        get_package_share_directory("franka_description"),
        "robots",
        "panda_arm.urdf.xacro",
    )

    robot_description = Command(
        [FindExecutable(name="xacro"), " ", franka_xacro_file, " hand:=true"]
    )

    launch_description = depth_proc_launch("hand_camera")

    launch_description.add_entity(
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[
                {"robot_description": ParameterValue(robot_description, value_type=str)}
            ],
        )
    )

    launch_description.add_entity(
        Node(
            package="grip",
            executable="ros_arm_example",
            name="ros_arm_example",
            output="screen",
        )
    )

    launch_description.add_entity(
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["--display-config", rviz_file],
        )
    )

    return launch_description
