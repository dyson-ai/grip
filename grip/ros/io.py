import os
from typing import Optional, List, Tuple
from ament_index_python.packages import get_package_share_directory
from ..io import pushd, get_file_path, log
import subprocess
import shlex
from .ros2_future import wait_for_message


def xacro_to_urdf_path(
    ros_pkg: str,
    xacro_filename_in: str,
    urdf_filename_out: str,
    xacro_args: Optional[str] = "",
    pkg_subfolder: Optional[str] = "",
    dependencies=[],
) -> str:
    """
    Args:
        ros_pkg: ros package name where the xacro file is located.
        xacro_file_in: xacro filename to look for.
        urdf_filename_out: desired urdf filename output.
        pkg_subfolder: optional package subfolder to inform where to look for xacro_file_in.
    Returns:
        str: full path to output urdf file

    """

    pkg_path = os.path.join(get_package_share_directory(ros_pkg), pkg_subfolder)

    log.debug(f"ROS package path: {pkg_path}")

    xacro_file_path = get_file_path(xacro_filename_in, pkg_path)

    xacro_file_directory = os.path.dirname(xacro_file_path)

    log.debug(f"Xacro file directory: {xacro_file_directory}")

    log.debug(f"Xacro file path: {xacro_file_path}")

    urdf_file_path = os.path.join(xacro_file_directory, urdf_filename_out)

    log.debug(f"Output urdf file path: {xacro_file_path}")

    file_path_out = ""
    with pushd(xacro_file_directory):
        os.system(f"rm {urdf_filename_out} || true 2>&1")
        os.system(
            f"ros2 run xacro xacro {xacro_filename_in} {xacro_args} > {urdf_filename_out}"
        )

        if urdf_filename_out.startswith("/"):
            dir_path = os.path.dirname(urdf_filename_out)
            tmp_pkg_path = f"{dir_path}/{ros_pkg}"
            os.system(f"rm -rf {tmp_pkg_path} || true 2>&1")

            os.system(f"mkdir -p {tmp_pkg_path} || true 2>&1")
            os.system(f"cp -r {pkg_path}/../* {tmp_pkg_path}/")
            os.system(f"cp -r {urdf_filename_out} {tmp_pkg_path}/{pkg_subfolder}")
            _, filename = os.path.split(urdf_filename_out)
            file_path_out = f"{tmp_pkg_path}/{pkg_subfolder}/{filename}"

            for dep in dependencies:
                dp = get_package_share_directory(dep)
                os.system(f"cp -r {dp} {tmp_pkg_path}/{pkg_subfolder}")
        else:
            file_path_out = urdf_file_path

    return file_path_out


def xacro_to_urdf_string(
    ros_pkg: str,
    xacro_filename_in: str,
    xacro_args: Optional[str] = "",
    pkg_subfolder: Optional[str] = "",
) -> str:
    """
    Args:
        ros_pkg: ros package name where the xacro file is located.
        xacro_file_in: xacro filename to look for.
        pkg_subfolder: optional package subfolder to inform where to look for xacro_file_in.
    Returns:
        str: urdf string

    """

    pkg_path = os.path.join(get_package_share_directory(ros_pkg), pkg_subfolder)

    log.debug(f"ROS package path: {pkg_path}")

    xacro_file_path = get_file_path(xacro_filename_in, pkg_path)

    xacro_file_directory = os.path.dirname(xacro_file_path)

    log.debug(f"Xacro file directory: {xacro_file_directory}")

    log.debug(f"Xacro file path: {xacro_file_path}")

    args = shlex.split(f"ros2 run xacro xacro {xacro_filename_in} {xacro_args}")

    sp = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=xacro_file_directory
    )

    output, error = sp.communicate()

    assert len(error) == 0, "Could not convert xacro file to URDF"

    return output.decode("utf-8")


def list_topics(parent_node) -> List[Tuple[str, List[str]]]:
    """Lists existing ros2 topics and corresponding type names

    Returns:
        parent_node: associated parent node
        List[str, List[str]]: list of topic names and associated type names
    """
    from ros2topic.api import get_topic_names_and_types

    topic_names_and_types = get_topic_names_and_types(
        node=parent_node, include_hidden_topics=True
    )

    return topic_names_and_types


def topic_exists(parent_node: "rclpy.node.Node", topic_name: str) -> bool:
    """checks if a given topic exists with topic_name

    Args:
        parent_node: associated parent node
        topic_name (str): the name of to topic to check its existence

    Returns:
        bool: whether or not a topic with given topic name exists
    """

    topic_dict = dict(list_topics(parent_node))

    return topic_name in topic_dict


def is_sim(
    parent_node: "rclpy.node.Node",
    query_topic_test: str = "/joint_states",
    time_to_wait: float = 1,
) -> bool:
    """Checks if we are running in sim or if a real robot already exists in the network

    Naive approach (TODO: implement a less naive approach)
    Assumption: if a robot is already running, then there should already be a /joint_states publishing topic.
    Solution: checks if the topic /joint_states is being published.

    Args:
        parent_node: associated parent node
        query_topic_test: query topic name to be used as reference check
        time_to_wait: time to wait for topic message
    Returns:
        bool: whether or not we are running
    """
    from sensor_msgs.msg import JointState

    received = False

    joint_state_topic_exists = topic_exists(parent_node, query_topic_test)

    if joint_state_topic_exists:
        received, _ = wait_for_message(
            JointState, parent_node, query_topic_test, time_to_wait
        )

    return not (joint_state_topic_exists and received)


def is_real_robot_connected(parent_node: "rclpy.node.Node") -> bool:
    """Checks if we are running in sim or if a real robot already exists in the network

    Naive approach (TODO: implement a less naive approach)
    Assumption: if a robot is already running, then there should already be a /diagnostics topic.
    Solution: checks if the topic /diagnostics is being published.

    Args:
        parent_node: associated parent node
    Returns:
        bool: whether or not we are running
    """

    query_topic_test = "/dynamic_joint_states"
    return topic_exists(parent_node, query_topic_test)
