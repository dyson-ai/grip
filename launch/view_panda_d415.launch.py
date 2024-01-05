import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


def generate_launch_description():
    rviz_file = os.path.join(
        get_package_share_directory("grip"), "launch", "rviz/view_panda_d415.rviz"
    )

    franka_xacro_file = os.path.join(
        get_package_share_directory("panda_panda_d415"), "urdf", "robot.urdf.xacro"
    )
    robot_description = Command([FindExecutable(name="xacro"), " ", franka_xacro_file])

    return LaunchDescription(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[
                    {
                        "robot_description": ParameterValue(
                            robot_description, value_type=str
                        )
                    }
                ],
                arguments=["--log-level", "debug"],
            ),
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                name="joint_state_publisher_gui",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["--display-config", rviz_file],
            ),
        ]
    )
