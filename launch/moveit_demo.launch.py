"""Launch file for the demo."""

from ament_index_python.packages import (
    get_package_share_path,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from grip.ros import depth_proc_launch

from launch_param_builder import ParameterBuilder

_PANDA_MOVEIT_CONFIG_RSC = "moveit_resources_panda_moveit_config"


def _octomap_launch_params(params: ParameterBuilder):
    params.parameter("octomap_frame", "panda_hand")
    params.parameter("octomap_resolution", 0.05)
    params.parameter("max_range", 5.0)
    return params.to_dict()


def generate_launch_description() -> LaunchDescription:
    """Generate launch description."""

    grip_shared_path = get_package_share_path("grip")

    sensor_path = grip_shared_path / "grip_assets/config" / "sensors3d.yaml"

    print("Path is: ", sensor_path)
    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(
            file_path=grip_shared_path / "grip_assets/urdf/robots" / "panda.urdf.xacro",
        )
        .robot_description_semantic(file_path="config/panda.srdf")
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        .planning_pipelines(
            pipelines=["ompl", "pilz_industrial_motion_planner"],
        )
        .sensors_3d(file_path=sensor_path)
        .to_moveit_configs()
    )

    # Pybullet grip
    grip_node = Node(
        package="grip",
        executable="ros_robot_moveit",
        parameters=[
            moveit_config.robot_description,
            # {"enable_gui": LaunchConfiguration("enable_gui")},
        ],
        output="screen",
    )

    _params_movegroup = ParameterBuilder(_PANDA_MOVEIT_CONFIG_RSC)

    # Start the actual move_group node/action server
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()]
        + [_octomap_launch_params(_params_movegroup)],
    )

    # RViz
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=[
            "-d",
            PathJoinSubstitution(
                [
                    FindPackageShare("grip"),
                    "launch",
                    "rviz/panda_moveit.rviz",
                ],
            ),
        ],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    # Static TF
    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "panda_link0"],
    )

    # Publish TF
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    # ros2_control
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,
            get_package_share_path("moveit_resources_panda_moveit_config")
            / "config"
            / "ros2_controllers.yaml",
        ],
        output="screen",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    panda_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["panda_arm_controller", "-c", "/controller_manager"],
    )

    panda_hand_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["panda_hand_controller", "-c", "/controller_manager"],
        output="screen",
    )

    launch_description = depth_proc_launch("hand_camera")

    return LaunchDescription(
        [
            grip_node,
            rviz_node,
            static_tf_node,
            robot_state_publisher,
            move_group_node,
            ros2_control_node,
            joint_state_broadcaster_spawner,
            panda_arm_controller_spawner,
            panda_hand_controller_spawner,
            launch_description,
            DeclareLaunchArgument(
                "enable_gui",
                default_value="true",
                description="Whether to start Pybullet GUI",
            ),
        ],
    )
