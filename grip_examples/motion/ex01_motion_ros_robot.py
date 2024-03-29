#!/usr/bin/env python3


import os
import numpy as np

try:
    import rclpy

except ImportError:
    raise RuntimeError("Unable to run example without ros2 installed.")

import pybullet_data as pd
from grip.ros import ROSRobotArm, ROSCamera
import grip
from grip.ros import xacro_to_urdf_path
from grip.motion import PBBasicPlanner


class ROSMotionArmExample(rclpy.node.Node):
    """This node example will simulate a robot arm with a wrist camera and allow for simple motion planning and execution."""

    def __init__(self, camera_fps: int = 10, joint_pub_rate: int = 150):
        super().__init__("motion_robot_arm_example_node")

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

        urdf_path = os.path.join(pd.getDataPath(), "franka_panda/panda.urdf")

        # urdf_path = xacro_to_urdf_path(
        #     "franka_description",
        #     "panda_arm.urdf.xacro",
        #     "/tmp/panda_arm.urdf",
        #     pkg_subfolder="robots",
        #     xacro_args="hand:=true",
        # )

        # Creates robot arm
        self.arm = ROSRobotArm(
            parent_node=self,
            urdf_file=urdf_path,
            phys_id=self.world.id,
            ee_mount_link="panda_link8",
            tip_link="panda_grasptarget",  # "panda_hand_tcp",
            gripper_type="panda_hand",
            sliders=True,
            auto_sim=False,
            sim=True,
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
            anchor_link="panda_grasptarget",  # "panda_hand_tcp",
            base_frame_id="panda_link0",
            sim=self.arm.sim,
        )

        # If we are in sim, then we create timer publishers for camera and joint states
        # This is not needed if data is to be consumed synchrnously.
        # However, it is needed if one wants to simulate asynchronous publisers just like the real robot and cameras would be behave.

        # Start camera timer publisher
        self.camera.setup_timer_publisher(fps=camera_fps)
        self.arm.joint_state_pub.setup_timer_publisher(rate=joint_pub_rate)

        self.timer = self.create_timer(1 / camera_fps, self.on_update)

        grip.robot.add_text([0, 0, 1], "mode: robot is simulated")

        self.planner = PBBasicPlanner(self.arm)

        self.send_goal_button = grip.robot.gui.BulletButton(
            self.world.id, "Execute Joint Goal"
        )

    def on_update(self):
        arm_positions = self.arm.joint_sliders.read()
        gripper_positions = self.arm.gripper.joint_sliders.read()

        goal_pose = self.arm.forward_kinematics(arm_positions)
        grip.robot.add_debug_frame(*goal_pose)

        if self.send_goal_button.was_pressed():
            path = self.planner.plan_cartesian_goal(
                goal_pose, obstacles=[], disabled_collisions=[], full_ret=True
            )

            self.arm.follow_joint_path(path)

        if self.arm.gripper.joint_sliders.has_changed():
            self.arm.gripper.set_angles(gripper_positions)


def main(args=None):
    rclpy.init(args=args)

    motion_ros_arm_example = ROSMotionArmExample()

    rclpy.spin(motion_ros_arm_example)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    motion_ros_arm_example.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
