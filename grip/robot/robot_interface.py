import abc
from typing import Protocol


class RobotInterface(abc.ABC):
    """
    Command methods with side effects
    """

    @abc.abstractmethod
    def exec_cartesian_cmd(self, cmd_p, cmd_q):
        """
        Not implemented
        """
        raise NotImplementedError("exec_cartesian_cmd: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def exec_cartesian_delta_cmd(self, cmd_dp, cmd_dq):
        """
        Not implemented
        """
        raise NotImplementedError(
            "exec_cartesian_delta_cmd: NO EFFECT, NOT IMPLEMENTED"
        )

    @abc.abstractmethod
    def exec_position_cmd(self, cmd):
        """
        Not implemented
        """

        raise NotImplementedError("exec_position_cmd: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def exec_position_cmd_delta(self, cmd):
        """
        Not implemented
        """
        raise NotImplementedError("exec_position_cmd_delta: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def exec_velocity_cmd(self, cmd):
        """
        Not implemented
        """
        raise NotImplementedError("exec_velocity_cmd: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def exec_torque_cmd(self, cmd):
        """
        Not implemented
        """
        raise NotImplementedError("exec_torque_cmd: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def move_to_home(self):
        """
        Not implemented
        """
        raise NotImplementedError("move_to_home: NO EFFECT, NOT IMPLEMENTED")

    """
    Get methods
    """

    @property
    @abc.abstractmethod
    def q_mean(self):
        """
        Not implemented, should return the mean between joint limits
        """
        raise NotImplementedError("q_mean: NO EFFECT, NOT IMPLEMENTED")

    @property
    @abc.abstractmethod
    def n_joints(self):
        """
        Not implemented, should return the number of joints of this robot, may differ to n_cmd()
        """
        raise NotImplementedError("n_joints: NO EFFECT, NOT IMPLEMENTED")

    @property
    @abc.abstractmethod
    def angles(self):
        """
        Returns current joint angles measured by encoders
        """
        raise NotImplementedError("angles: NO EFFECT, NOT IMPLEMENTED")

    @property
    @abc.abstractmethod
    def joint_limits(self):
        """
        Returns joint limits
        """
        raise NotImplementedError("joint_limits: NO EFFECT, NOT IMPLEMENTED")

    @property
    @abc.abstractmethod
    def joint_names(self):
        """
        Returns list of joint names for this robot
        """
        raise NotImplementedError("joint_names: NO EFFECT, NOT IMPLEMENTED")

    @property
    @abc.abstractmethod
    def joint_velocities(self):
        """
        Returns joint velocities measurements values for this robot
        """
        raise NotImplementedError("joint_velocities: NO EFFECT, NOT IMPLEMENTED")

    @property
    @abc.abstractmethod
    def joint_efforts(self):
        """
        Returns joint effort measurements values for this robot
        """
        raise NotImplementedError("joint_efforts: NO EFFECT, NOT IMPLEMENTED")

    @property
    @abc.abstractmethod
    def ee_pose(self):
        """
        Not implemented, should return end-effector pose
        """
        raise NotImplementedError("ee_pose: NO EFFECT, NOT IMPLEMENTED")

    @property
    @abc.abstractmethod
    def ee_velocity(self):
        """
        Not implemented, should return end-effector velocity
        """
        raise NotImplementedError("ee_velocity: NO EFFECT, NOT IMPLEMENTED")

    @property
    @abc.abstractmethod
    def base_pose(self):
        """
        Not implemented, should return robot base pose
        """
        raise NotImplementedError("base_pose: NO EFFECT, NOT IMPLEMENTED")

    @property
    @abc.abstractmethod
    def base_velocity(self):
        """
        Not implemented, should return robot spatial velocity
        """
        raise NotImplementedError("base_velocity: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def forward_kinematics(self, joint_angles=None):
        """
        Not implemented, should return the forward kinematics over joint_angles
        """
        raise NotImplementedError("forward_kinematics: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def inverse_kinematics(self, position, orientation=None):
        """
        Not implemented, should attempt to solve inverse kinematics
        for the given position and orientation
        """
        raise NotImplementedError("inverse_kinematics: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def jacobian(self, joint_angles=None):
        """
        Not implemented, should return the jacobian J(q) for the given joint angles
        """
        raise NotImplementedError("jacobian: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def inertia(self, joint_angles=None):
        """
        Not implemented, should return the inertia tensor for the given joint angles M(q)
        """
        raise NotImplementedError("inertia: NO EFFECT, NOT IMPLEMENTED")
