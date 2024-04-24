#!/usr/bin/env python

import os
import time
import math
import numpy as np
import pybullet as p

from grip.io import get_data_path
from grip.robot.bullet_robot import BulletRobot
from grip.math.quaternion import position_quaternion2matrix, matrix2position_quaternion
from .end_effector_registry import EndEffectorRegistry, EndEffectorType


base_dir = get_data_path()
PANDA_WITH_ROBOTIQ_2F_85_GRIPPER_URDF = os.path.join(
    base_dir, "urdf/franka_panda/panda_with_robtiq_2f_85_gripper.urdf"
)
PANDA_WITH_ROBOTIQ_2F_140_GRIPPER_URDF = os.path.join(
    base_dir, "urdf/franka_panda/panda_with_robotiq_2f_140_gripper.urdf"
)


class BulletRobotiq2FGripper(BulletRobot, EndEffectorType):
    # Gripper type name for real gripper class (in arm farm layer)
    HW_INTERFACE = "robot_interface.gripper.robotiq_2f_gripper:Robotiq2fGripper"

    def __init__(self, **kwargs):
        kwargs["urdf_file"] = kwargs.get("urdf_file", None)
        kwargs["ee_index"] = kwargs.get("ee_index", -1)
        kwargs["tip_link"] = kwargs.get("tip_link", "tool0")

        print("Tip link: ", kwargs["tip_link"])

        kwargs["joint_names"] = kwargs.get(
            "joint_names",
            [
                "finger_joint",
                "left_inner_finger_joint",
                "left_inner_knuckle_joint",
                "right_outer_knuckle_joint",
                "right_inner_finger_joint",
                "right_inner_knuckle_joint",
            ],
        )

        kwargs["has_kinematics"] = False

        # kwargs["max_force"] = kwargs.get("max_force", 15)

        super().__init__(**kwargs)

        self.T_ee2ik = None
        self.mimic_parent_joint_name = "finger_joint"
        self.ref_joint_name = "finger_joint"
        self.ref_joint_id = self.joint_dict[self.ref_joint_name]["idx"]
        min_angle = self.joint_dict[self.ref_joint_name]["lower_limit"]
        self.max_w = self.angle2width(min_angle)
        max_angle = self.joint_dict[self.ref_joint_name]["upper_limit"]
        self.min_w = self.angle2width(max_angle)

        self.close_angles = [0.700, 0.725, -0.725, -0.725, 0.725, -0.725]
        self.open_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    @property
    def true_ee_pose(self):
        lft_pos, _ = self.get_link_pose(self.link_dict["left_inner_finger_tip"])
        rft_pos, _ = self.get_link_pose(self.link_dict["right_inner_finger_tip"])
        ee_payload_pos = (lft_pos + rft_pos) / 2

        self.get_link_id()
        _, ee_payload_quat = self.get_link_pose(self.tip_link)
        return ee_payload_pos, ee_payload_quat

    @property
    def ee_height(self):
        return self.true_ee_pose[0][2]

    def angle2width(self, angle):
        """
        conversion utility function for robotiq-2f-85
        """
        width = 0.1143 * math.sin(0.715 - angle) + 0.01
        return width

    def width2angle(self, width):
        """
        conversion utility function for robotiq-2f-85
        """
        angle = 0.715 - math.asin((width - 0.01) / 0.1143)
        return angle

    @property
    def width(self):
        """
        Current gripper opening width
        """
        joint_state = p.getJointState(
            self.id, self.ref_joint_id, physicsClientId=self.phys_id
        )
        width = self.angle2width(joint_state[0])
        return width

    def move_to_width(self, width):
        """
        move gripper to the target width

        :param float width: the target width to reach
        """
        target_angle = self.width2angle(width)
        p.setJointMotorControl2(
            self.id,
            self.ref_joint_id,
            p.POSITION_CONTROL,
            targetPosition=target_angle,
            force=self.joint_dict[self.ref_joint_name]["max_force"],
            maxVelocity=self.joint_dict[self.ref_joint_name]["max_vel"],
            physicsClientId=self.phys_id,
        )
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.phys_id)
            time.sleep(1 / 240.0)

    def ee2ik_frame_pose(self, pos, quat, update_T=False):
        """
        conversion function when gripper was fully opened
        """
        if (self.T_ee2ik is None) or update_T:
            ik_frame_pose = position_quaternion2matrix(self.ee_pose[0], self.ee_pose[1])
            ee_frame_pose = position_quaternion2matrix(
                self.true_ee_pose[0], self.true_ee_pose[1]
            )
            self.T_ee2ik = np.matmul(np.linalg.inv(ee_frame_pose), ik_frame_pose)
        pose = position_quaternion2matrix(pos, quat)
        transformed_pose = np.matmul(pose, self.T_ee2ik)
        transformed_pos, transformed_quat = matrix2position_quaternion(transformed_pose)
        return transformed_pos, transformed_quat

    def apply_gripper_delta_action(self, action):
        """
        control gripper with a delta finger distance action.

        :param float action: delta joint angle action for all joints of the phase 4 gripper
        """
        curr_width = self.width
        target_width = min(self.max_w, max(self.min_w, curr_width + action))
        self.move_to_width(target_width)

    def control_fingers(self, mode="close"):
        """
        Opening/closing finger

        :param str mode: 'open' or 'close' mode
        """

        if mode == "close":
            angles = self.close_angles
        else:
            angles = self.open_angles
        for _ in range(5):
            self.set_angles(angles)
            p.stepSimulation(physicsClientId=self.phys_id)
            time.sleep(1 / 240.0)

    def validate_grasp(self, tol: float = 1e-3):
        diff = self.width - 0.012

        if diff < tol:
            has_object = False
        else:
            has_object = True

        return has_object

    def is_ready(self) -> bool:
        """Simulated gripper is always ready

        Returns:
            (bool): whether or not this gripper is ready (in sim it always is)
        """
        return True


class BulletRobotiq2F85Gripper(BulletRobotiq2FGripper, EndEffectorType):
    HW_INTERFACE = "robot_interface.gripper.robotiq_2f_gripper:Robotiq2f85Gripper"

    def __init__(self, **kwargs):
        kwargs["urdf_file"] = kwargs.get(
            "urdf_file", PANDA_WITH_ROBOTIQ_2F_85_GRIPPER_URDF
        )

        super().__init__(**kwargs)

        self.mimic_parent_joint_name = "finger_joint"
        self.mimic_children = {
            "right_outer_knuckle_joint": 1,
            "left_inner_knuckle_joint": 1,
            "right_inner_knuckle_joint": 1,
            "left_inner_finger_joint": -1,
            "right_inner_finger_joint": -1,
        }
        self.setup_mimic_joints(self.mimic_parent_joint_name, self.mimic_children)

        self.close_angles = [0.700, -0.725, 0.725, 0.725, -0.725, 0.725]
        self.open_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.physical_links = [8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20]
        self.home_positions = self.open_angles
        self.reset_angles(self.home_positions)


@EndEffectorRegistry.register("robotiq_2f140")
class BulletRobotiq2F140Gripper(BulletRobotiq2FGripper, EndEffectorType):
    HW_INTERFACE = "robot_interface.gripper.robotiq_2f_gripper:Robotiq2f85Gripper"

    def __init__(self, **kwargs):
        kwargs["urdf_file"] = kwargs.get(
            "urdf_file", PANDA_WITH_ROBOTIQ_2F_140_GRIPPER_URDF
        )

        super().__init__(**kwargs)

        self.mimic_children = {
            "left_inner_finger_joint": 1,
            "left_inner_knuckle_joint": -1,
            "right_outer_knuckle_joint": -1,
            "right_inner_finger_joint": 1,
            "right_inner_knuckle_joint": -1,
        }
        # self.setup_mimic_joints(self.mimic_parent_joint_name, self.mimic_children)

        self.close_angles = [0.700, 0.725, -0.725, -0.725, 0.725, -0.725]
        self.open_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.physical_links = [8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20]
        self.home_positions = self.open_angles
        self.reset_angles(self.home_positions)
