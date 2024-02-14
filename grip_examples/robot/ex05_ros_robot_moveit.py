#!/usr/bin/env python3
import os
import numpy as np

try:
    import rclpy

except ImportError:
    raise RuntimeError("Unable to run example without ros2 installed.")

import pybullet_data as pd
import grip
from grip.ros import ROSRobotArm, ROSCamera

from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from grip.ros.urdf_utils import normalise_urdf_paths


class ROSArmMoveitExample(rclpy.node.Node):
    """This node example will simulate a robot arm with a wrist camera."""

    def __init__(self, camera_fps: int = 10, joint_pub_rate: int = 150):
        super().__init__("robot_arm_moveit_example")

        self.declare_parameter(
            "robot_description",
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )
        self.declare_parameter(
            "enable_gui",
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_BOOL),
        )

        # Creates bullet world with GUI
        self.world = grip.robot.BulletWorld(phys_opt="gui")

        # Creates ground plane object
        self.plane = grip.robot.BulletObject(
            urdf_file=os.path.join(pd.getDataPath(), "plane.urdf"), position=[0, 0, 0]
        )

        self.plane.set_colour([0.0, 0.7, 0.7, 1.0])
        self.duck = grip.robot.BulletObject(
            urdf_file=os.path.join(pd.getDataPath(), "duck_vhacd.urdf"),
            position=[0.7, 0.0, 0.0],
            scale=2.0,
            fixed=False,
        )

        urdf_path = normalise_urdf_paths(self.get_parameter("robot_description").value)

        # Creates robot arm
        self.arm = ROSRobotArm(
            parent_node=self,
            urdf_file=urdf_path,
            phys_id=self.world.id,
            ee_mount_link="panda_link8",
            tip_link="panda_hand",
            gripper_type="panda_hand",
            sliders=True,
            auto_sim=False,
            sim=True,
            state_topic="arm/joint_states",
            command_topic="arm/joint_commands",
        )

        print("Links: ", self.arm.get_link_names())

        # Creates ROSCamera object
        self.camera = ROSCamera(
            name="hand_camera",
            cid=self.world.id,
            position=np.array([0, 0, 1.5]),
            orientation=grip.math.rpy2quaternion([0.0, np.pi, 0.0]),
            parent_node=self,
            anchor_robot=self.arm,
            anchor_link="hand_camera_optical_frame",
            base_frame_id="panda_link0",
            sim=self.arm.sim,
        )

        self.send_goal_button = grip.robot.gui.BulletButton(
            self.world.id, "Execute Joint Goal"
        )

        if self.arm.sim:
            # If we are in sim, then we create timer publishers for camera and joint states
            # This is not needed if data is to be consumed synchrnously.
            # However, it is needed if one wants to simulate asynchronous publisers just like the real robot and cameras would be behave.

            # Start camera timer publisher
            self.camera.setup_timer_publisher(fps=camera_fps)
            self.arm.start_async_ros_comm(rate=joint_pub_rate)

            grip.robot.add_text([0, 0, 1], "mode: robot is simulated")

        else:
            grip.robot.add_text([0, 0, 1], "mode: robot is mirroring real robot state")

        self.timer = self.create_timer(1 / camera_fps, self.on_update)

    def on_update(self):
        if self.send_goal_button.was_pressed():
            gripper_goal = self.arm.gripper.joint_sliders.read()
            self.arm.gripper.set_angles(gripper_goal)


def main(args=None):
    rclpy.init(args=args)

    ros_arm_example = ROSArmMoveitExample()

    rclpy.spin(ros_arm_example)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    ros_arm_example.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
