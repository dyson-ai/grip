import os

from ament_index_python.packages import get_package_share_directory

import launch_ros.actions
import launch_ros.descriptions

from grip.ros import depth_proc_launch


def generate_launch_description(camera_name="CameraSensor"):
    default_rviz = os.path.join(
        get_package_share_directory("grip"), "launch", "rviz/rgbd_camera_example.rviz"
    )

    # depth processing launch (converts rgb, depth and camera info into point cloud efficiently)
    launch_description = depth_proc_launch(camera_name)

    # example rgbd camera node
    launch_description.add_entity(
        launch_ros.actions.Node(
            package="grip", executable="rgbd_camera_example", name="rgbd_camera_example"
        )
    )

    # rviz
    launch_description.add_entity(
        launch_ros.actions.Node(
            package="rviz2",
            executable="rviz2",
            output="screen",
            arguments=["--display-config", default_rviz],
        )
    )

    return launch_description
