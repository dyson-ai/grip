# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Software License Agreement (BSD License 2.0)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of the Willow Garage nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from typing import Optional
from launch import LaunchDescription

import launch_ros.actions
import launch_ros.descriptions


def depth_proc_launch(
    camera_name: Optional[str] = "CameraSensor",
    camera_info_topic: Optional[str] = "/hand_camera/color/undistorted/camera_info",
    colour_image_topic: Optional[str] = "/hand_camera/color/undistorted/image_rect",
    depth_image_topic: Optional[
        str
    ] = "/hand_camera/depth_registered/undistorted/image_rect",
) -> LaunchDescription:
    """Creates the launch description for a depth processing node, which converts rgb, depth and camera info to a point cloud with xyzrgb fields.

    Args:
        camera_name (Optional[str], optional): camera name which is used as prefix for the topics this depth proccessing node will subscribe to. Defaults to "CameraSensor".

    Returns:
        LaunchDescription: the launch description for a depth processing node
    """

    return LaunchDescription(
        [
            # launch plugin through rclcpp_components container
            launch_ros.actions.ComposableNodeContainer(
                name="container",
                namespace="",
                package="rclcpp_components",
                executable="component_container",
                composable_node_descriptions=[
                    # Driver itself
                    launch_ros.descriptions.ComposableNode(
                        package="depth_image_proc",
                        plugin="depth_image_proc::PointCloudXyzrgbNode",
                        name="point_cloud_xyzrgb_node",
                        remappings=[
                            ("rgb/camera_info", camera_info_topic),
                            (
                                "rgb/image_rect_color",
                                colour_image_topic,
                            ),
                            (
                                "depth_registered/image_rect",
                                depth_image_topic,
                            ),
                            ("points", f"/{camera_name}/depth_registered/points"),
                        ],
                    ),
                ],
                output="screen",
            ),
        ]
    )


def generate_launch_description(camera_name="CameraSensor"):
    return depth_proc_launch(camera_name)
