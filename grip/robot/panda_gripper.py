import time
import pybullet as p
import os, copy
import pybullet_data as pd
from .bullet_robot import BulletRobot
from .gripper_interface import GripperInterface
from .end_effector_registry import EndEffectorRegistry

base_dir = pd.getDataPath()
PANDA_WITH_DEFAULT_GRIPPER_URDF = os.path.join(base_dir, "franka_panda/panda.urdf")


@EndEffectorRegistry.register("panda_hand")
class BulletPandaGripper(BulletRobot, GripperInterface):
    """
    The gripper interface for default panda gripper, which is a two-finger parallel gripper
    with two controllable joints.

    :param str urdf_file: urdf file for robot arm with default gripper
    :param int ee_index: reference link index for end effector
    :param [str] joint_names: list of joint names
    :param bool has_kinematics: to/not to extend arm with gripper kinematic chain
    :param int max_force: maximum finger force to apply
    """

    # Gripper type name for real gripper class (in arm farm layer)
    HW_INTERFACE = "arm_farm_interface.gripper_interface:DefaultFrankaGripper"

    def __init__(self, **kwargs):
        kwargs["urdf_file"] = kwargs.get("urdf_file", PANDA_WITH_DEFAULT_GRIPPER_URDF)
        kwargs["ee_index"] = kwargs.get("ee_index", -1)
        kwargs["tip_link"] = kwargs.get("tip_link", "panda_hand")

        kwargs["joint_names"] = ["panda_finger_joint1", "panda_finger_joint2"]

        kwargs["max_force"] = kwargs.get("max_force", 15)

        super().__init__(has_kinematics=False, **kwargs)

        self.physical_links = [8, 9, 10]
        self.home_positions = copy.deepcopy(self._ul)

        self.close_angles, self.open_angles = self.joint_limits

        self.reset_angles(self.home_positions)

    @property
    def ee_height(self):
        return self.ee_pose[0][2]

    def ee2ik_frame_pose(self, pos, quat):
        return pos, quat

    def apply_gripper_delta_action(self, action):
        """
        control gripper with a delta finger distance action.

        :param float action: delta joint angle action for all joints of the phase 4 gripper
        """
        positions = self.angles

        for i in range(len(self.joint_ids)):
            positions += action

            positions[i] = min(self._ul[i], max(self._ll[i], positions[i]))

            self.set_angles(positions, positionGain=1)

    def control_fingers(self, mode="close"):
        """
        A hard-coded opening/closing finger process with constant speed +/-0.05 per step time

        :param str mode: 'open' or 'close' mode
        """

        delta_action = 0.005
        n_update = 10

        if mode == "close":
            action = -delta_action
        else:
            action = delta_action

        for _ in range(n_update):
            self.apply_gripper_delta_action(action)
            p.stepSimulation(physicsClientId=self.phys_id)
            if mode == "close":
                time.sleep(3 / 240.0)

    def is_ready(self) -> bool:
        """Simulated gripper is always ready

        Returns:
            (bool): whether or not this gripper is ready (in sim it always is)
        """
        return True

    def validate_grasp(self) -> bool:
        """Returns true if the gripper is holding an object

        Returns:
            bool: The gripper has a valid grasp on the object
        """

        self.control_fingers(mode="close")
        return self.angles[0] > 0.01
