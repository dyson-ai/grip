import pybullet as p
import copy
import numpy as np
from ..io import *
from ..math import quat_multiply, rpy2quaternion, multiply_transform, invert_transform

from .robot_interface import RobotInterface
from .entity import Entity
from .world import BulletWorld
from .gui import BulletSliders
from .types import BulletDynamicsInfo, BulletJointInfo, VisualShapeData

from typing import Tuple, List, Dict, Optional


class BulletRobot(RobotInterface, Entity):
    """
    The BulletRobot represents an interface and connection to a robot
    An object of this class is exposes all generic functionalities to control a robot


    Args:
        urdf_file (str): full path to a robot URDF description
        id (int): if the robot entity already exists in the physics engine, then the id of the robot can be passed instead of its urdf_file
        parent (grip.robot.BulletRobot, optional): This robot may be a sub-part of an already existing parent robot (e.g. a robot hand attached to a robot arm)
        phys_id (int): physics engine identifier (the id of a BulletWorld)
        use_fixed_base (bool): Whether or not to make the object fixed in place. Default: True.
        position (numpy.ndarray): shapebase position of the robot
        orientation (numpy.ndarray): base orientation of the robot as en orientation quaterion following JPL format [x, y, z, w]
        ee_index (int, optional): end-effector link id. Defaults to None. Note: either ee_index or tip_link must be set, but not both.
        tip_link (str, optional): end-effector tip link name. Specifies the end-effector tip link, if ee_index is None, then ee_index = self.get_link_id(tip_link). Defaults to None.
        Note: either ee_index or tip_link must be set, but not both.
        ee_mount_link (str, optional): end-effector mount link name, used to specify up at which link the end-effector has been mounted on the robot.
        joint_names (str, optional): joint names that this robot instance should be controlling (can be a subset of the total number of joints of the robot)
            if not passed as a parameter, or None is given, then this BulletRobot will use all joint names as defined in the urdf description
        has_kinematics (bool): enable or disable forward and inverse kinematics with this robot (this should be usually true, but some robots may not need it)
        max_force (float): maximum generalised force
        sliders (bool): whether or not joint sliders should be added to the debug GUI (only used if grip.robot.BulletWorld was created with phys_opt="gui")
        ee_sliders (bool): whether or not cartesian end-effector sliders should be added to the debug GUI (only used if grip.robot.BulletWorld was created with phys_opt="gui")
    """

    def __init__(self, **kwargs):
        self.urdf_file = kwargs.get("urdf_file", None)
        self.data_path = kwargs.get("data_path", get_data_path())

        log.debug("Data path: {}".format(self.data_path))
        p.setAdditionalSearchPath(self.data_path)

        self.phys_id = kwargs.get("phys_id", None)
        self._useFixedBase = kwargs.get("use_fixed_base", True)
        self._position = np.array(kwargs.get("position", [0.0, 0.0, 0.0]))
        self._orientation = np.array(kwargs.get("orientation", [0, 0, 0, 1]))

        self.max_force = kwargs.get(
            "max_force", 500
        )  # max force when a joint in pos/vel controlled
        self.joint_friction = kwargs.get(
            "joint_friction", 0.01
        )  # friction force when a joint is in full torque mode

        self._sliders = kwargs.get("sliders", False)
        self._ee_sliders = kwargs.get("ee_sliders", False)

        # flags = p.URDF_USE_SELF_COLLISION#|p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        # flags = p.URDF_USE_SELF_COLLISION #p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        flags = 0
        self.parent = kwargs.get("parent", None)

        self.id = kwargs.get("id", None)

        self.texture_randomiser = kwargs.get("texture_randomiser", None)

        # assert self.parent is None or self.urdf_file is None and (not (self.parent is None and self.urdf_file is None)), \
        #     "Invalid instance. Provide either parent or URDF file, not both!"

        self.joint_idx0 = 0

        self.reset_home_positions = True
        if self.parent is None and self.id is None:
            self.id = p.loadURDF(
                self.urdf_file,
                self._position,
                self._orientation,
                useFixedBase=self._useFixedBase,
                physicsClientId=self.phys_id,
                flags=flags,
            )
        elif self.parent is not None and self.id is None:
            ## Else, this class is being attached to an existing body (e.g. gripper attached to arm)
            self.id = self.parent.id
            self.urdf_file = self.parent.urdf_file
            self.phys_id = self.parent.phys_id
            self.joint_idx0 = self.parent.n_joints
        else:
            # Else this is being attached to a robot existing somewhere else in the network
            self.reset_home_positions = False

        self._joint_names = kwargs.get("joint_names", None)

        self.ee_mount_link = kwargs.get("ee_mount_link", None)

        if self.ee_mount_link is not None and self._joint_names is None:
            _, self._joint_names = self.get_movable_joints(
                upto_link_name=self.ee_mount_link
            )

        self.joint_dict, self.link_dict = self.get_joint_dict()
        self.actuated_joint_dict = self.joint_dict.copy()
        self.link_dict["-1base"] = -1

        self.reverse_link_dict = dict([(v, k) for k, v in self.link_dict.items()])

        if self._joint_names is None:
            self._joint_names = list(self.joint_dict.keys())

        self.actuated_joint_names = self._joint_names.copy()
        self._setup_limits(**kwargs)

        if self.reset_home_positions:
            self.reset_angles(self.home_positions)

        # Kinematics setup
        self._kin = None
        has_kinematics_ = kwargs.get("has_kinematics", True)
        if has_kinematics_:
            self.setup_kinematics()

        # self.enable_torque_sensor(True)

        self.joint_sliders = None
        self.ee_sliders = None
        self.setup_gui()

        self._base_cid = None

    def _setup_limits(self, **kwargs) -> None:
        self.joint_ids = [self.joint_dict[name]["idx"] for name in self.joint_names]
        self.actuated_joint_ids = self.joint_ids.copy()
        self._ll = [self.joint_dict[name]["lower_limit"] for name in self.joint_names]
        self._ul = [self.joint_dict[name]["upper_limit"] for name in self.joint_names]
        self._jr = [self.joint_dict[name]["joint_range"] for name in self.joint_names]
        self._rp = [self.joint_dict[name]["rest_posture"] for name in self.joint_names]
        self._q_mean = [self.joint_dict[name]["q_mean"] for name in self.joint_names]

        self._n_joints = len(self.joint_ids)

        self._ndof = len(list(self.joint_dict.keys()))

        self.home_positions = kwargs.get("home_positions", self._rp)

        self.tip_link = kwargs.get("tip_link", None)

        self.ee_index = kwargs.get("ee_index", None)

        if self.tip_link is None and self.ee_index is not None:
            self.tip_link = self.get_link_name(self.ee_index)
        elif self.tip_link is not None and self.ee_index is None:
            self.ee_index = self.get_link_id(self.tip_link)
        elif self.tip_link is None and self.ee_index is None:
            self.ee_index = -1
            self.tip_link = self.get_link_name(self.ee_index)
        elif self.tip_link is not None and self.ee_index is not None:
            if self.get_link_id(self.tip_link) != self.ee_index:
                log.warning(
                    f"Tip link {self.tip_link} id does not match self.ee_index, but it should. Current ee_index is set to link name {self.get_link_name(self.ee_index)}. Will set self.ee_index = self.get_link_id(self.tip_link)."
                )

                self.ee_index = self.get_link_id(self.tip_link)

        log.debug("Tip link: {}".format(self.tip_link))
        log.debug("Joint names: {}".format(self.joint_names))
        log.debug("Joint IDs: {}".format(self.joint_ids))

    def setup_kinematics(self) -> None:
        """
        Sets up robot kinematics solver. This enable calls to forward_kinematics and inverse_kinematics.
        """

        if self._kin:
            log.info(
                "Kinematics was already set-up. Clearing up previous instance and re-initialising it."
            )

            self._kin._release()
            self._kin = None

        self._kin = RobotKinematics(parent=self)

    def has_kinematics(self) -> bool:
        """
        Checkes if this robot has a kinematics solver setup. The kinematics solver is a damped least squares differential inverse kinematics solver native from Bullet.

        Returns:
            (bool): whether or not this robot has a kinematics solver setup.

        """

        return hasattr(self, "_kin") and self._kin is not None

    def release_kinematics(self) -> None:
        """
        Releases default bullet kinematics solver for this robot.
        This method has no effect if this robot has no default kinematics solver or if its solver has been already released.
        """
        if self.has_kinematics():
            self._kin._release()
            self._kin = None

    def __del__(self):
        self.release_kinematics()

    def setup_gui(self) -> None:
        """
        Sets up debug GUI, adding joint sliders and end-effector joint sliders (if they were set to be shown in the constructor)
        """

        if self.joint_sliders is not None:
            self.joint_sliders.remove()
        if self.ee_sliders is not None:
            self.ee_sliders.remove()

        if self._sliders:
            self.joint_sliders = BulletSliders(
                self.phys_id, self.joint_names, self._ll, self._ul, self.angles
            )

        if self._ee_sliders:
            ee_names = ["ee_x", "ee_y", "ee_z", "ee_roll", "ee_pitch", "ee_yaw"]
            ee_pos, ee_ori = self.ee_pose
            ee_rpy = np.array(
                p.getEulerFromQuaternion(ee_ori, physicsClientId=self.phys_id)
            )

            min_pos = [c - 1.0 for c in ee_pos]
            max_pos = [c + 1.0 for c in ee_pos]
            min_ori = [-np.pi * 2] * 3
            max_ori = [np.pi * 2] * 3

            ee_pose = np.hstack([ee_pos, ee_rpy]).tolist()
            max_vals = max_pos + max_ori
            min_vals = min_pos + min_ori

            map_func = lambda values: (
                np.array(values[:3]),
                np.array(rpy2quaternion(values[3:])),
            )

            self.ee_sliders = BulletSliders(
                self.phys_id, ee_names, min_vals, max_vals, ee_pose, map_func=map_func
            )

    ## Getters

    def get_joint_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Gets the current joint state of the robot, which includes joint positions, velocities, reaction forces and joint efforts.

        Returns:
            (Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]): a tuple of 4 arrays with shape-(ndof,), each respectively being joint positions, velocities, reaction foces and joint efforts

        """
        joint_angles = []
        joint_velocities = []
        joint_reaction_forces = []
        joint_efforts = []

        for idx in self.joint_ids:
            joint_state = p.getJointState(self.id, idx, physicsClientId=self.phys_id)

            joint_angles.append(joint_state[0])

            joint_velocities.append(joint_state[1])

            joint_reaction_forces.append(joint_state[2])

            joint_efforts.append(joint_state[3])

        return (
            np.array(joint_angles),
            np.array(joint_velocities),
            np.array(joint_reaction_forces),
            np.array(joint_efforts),
        )

    def get_joint_dict(self) -> Dict[str, dict]:
        """
        Gets a dictionary of dictionaries containing information about the joints of this robot.

        Returns:
            (dict): joint info dictionary in the format of ``dict: {'joint_name': dict { 'q_index': [str, str], 'lower_limit': float, 'upper_limit': float, 'rest_posture': float, 'q_mean': float, 'joint_range': float, 'idx': int }}``.
        """
        # attribute_list = ['jointIndex', 'jointName', 'jointType',
        #                   'qIndex', 'uIndex', 'flags',
        #                   'jointDamping', 'jointFriction', 'jointLowerLimit',
        #                   'jointUpperLimit', 'jointMaxForce', 'jointMaxVelocity', 'linkName']

        joint_dict = {}
        link_dict = {}

        for i in range(p.getNumJoints(self.id, physicsClientId=self.phys_id)):
            joint_info = p.getJointInfo(self.id, i, physicsClientId=self.phys_id)

            link_name = joint_info[12].decode("utf-8")
            link_dict[link_name] = i

            is_movable = joint_info[2] != p.JOINT_FIXED

            if is_movable:
                q_index = joint_info[3]
                joint_name = joint_info[1].decode("utf-8")
                joint_type = joint_info[2]
                joint_ll = joint_info[8]
                joint_ul = joint_info[9]
                joint_max_force = joint_info[10]
                joint_max_vel = joint_info[11]

                rest_posture = 0.5 * (joint_ll + joint_ul)
                joint_range = joint_ul - joint_ll

                joint_dict[joint_name] = {
                    "q_index": q_index,
                    "lower_limit": joint_ll,
                    "upper_limit": joint_ul,
                    "max_force": joint_max_force,
                    "max_vel": joint_max_vel,
                    "rest_posture": rest_posture,
                    "q_mean": copy.deepcopy(rest_posture),
                    "joint_range": joint_range,
                    "idx": i,
                    "link_name": link_name,
                }

        return joint_dict, link_dict

    def get_movable_joints(
        self, upto_link_name: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Gets the movable joint ids and list of joint names.
        Args:

        Returns:
            (Tuple[numpy.ndarray, List[str]]): tuple containing the array with shape-(ndof,) with movable joint IDs, and list of movable joint names.

        """
        movable_joints = []
        joint_names = []

        for i in range(p.getNumJoints(self.id, physicsClientId=self.phys_id)):
            joint_info = p.getJointInfo(self.id, i, physicsClientId=self.phys_id)
            q_index = joint_info[3]
            joint_name = joint_info[1]
            joint_type = joint_info[2]
            joint_ll = joint_info[8]
            jint_ul = joint_info[9]
            link_name = joint_info[12].decode("utf-8")

            if joint_type != p.JOINT_FIXED:
                movable_joints.append(i)

                log.debug("Qidx {} CurrIdx {}".format(q_index, i))
                joint_names.append(joint_name.decode("utf-8"))

            if link_name == upto_link_name:
                break

        return np.array(movable_joints), joint_names

    def get_joint_limits(
        self,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Gets a tuple containing the lower and upper limits, as well as joint rage (upper-lower) and rest posture ``(upper+lower)*0.5``

        Returns:
            (Tuple[List[float], List[float], List[float], List[float]]): lower and upper limits, joint range and rest posture.

        """
        lower_lim = np.zeros(self.n_joints)

        upper_lim = np.zeros(self.n_joints)

        rest_posture = np.zeros(self.n_joints)

        joint_range = np.zeros(self.n_joints)

        for k, idx in enumerate(self.joint_ids):
            lower_lim[k] = p.getJointInfo(self.id, idx, physicsClientId=self.phys_id)[8]

            upper_lim[k] = p.getJointInfo(self.id, idx, physicsClientId=self.phys_id)[9]

            rest_posture[k] = 0.5 * (lower_lim[k] + upper_lim[k])

            joint_range[k] = upper_lim[k] - lower_lim[k]

        # [dict([x]) for x in zip(['upper'] * a.n_cmd(), a._bullet_robot.get_joint_limits()['upper'])]
        # {'lower': lower_lim, 'upper': upper_lim, 'mean': rest_posture, 'range': joint_range}
        return (
            lower_lim.tolist(),
            upper_lim.tolist(),
            joint_range.tolist(),
            rest_posture.tolist(),
        )

    def get_joint_info(self, joint_id: int) -> BulletJointInfo:
        """
        Returns

        Args:
            joint_id (int): joint identifier.

        Returns:
           BulletJointInfo: bullet joint information of given joint_id

        """

        return BulletJointInfo(
            *p.getJointInfo(self.id, joint_id, physicsClientId=self.phys_id)
        )

    def get_base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets the current base pose of this robot, as a position and quaternion.

        Returns:
            (Tuple[numpy.ndarray, numpy.ndarray]): tuple containing a position shape-(3,) and unit quaternion orientation shape-(4,) representing the pose of this robot.
        """
        pos, ori = p.getBasePositionAndOrientation(
            self.id, physicsClientId=self.phys_id
        )

        return np.asarray(pos), np.asarray(ori)

    def get_base_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets the current base velocity of this robot, as a linear and angular velocity in cartesian world coordinates.

        Returns:
            (Tuple[numpy.ndarray, numpy.ndarray]): tuple containing a linear shape-(3,) and angular shape-(3,) velocities.
        """
        lin, ang = p.getBaseVelocity(self.id, physicsClientId=self.phys_id)

        return np.asarray(lin), np.asarray(ang)

    def get_link_ids(self, link_names: Optional[list[str]] = None) -> List[int]:
        """
        Gets the list of link IDs of this robot.

        Args:
            link_names (list[str], optional): optional list of link names. If not passed method returns all link_ids of this kinematic chain
        Returns:
            (List[int]): list of unique identifiers the links belonging to this robot, as loaded by the physics engine.
        """

        if link_names:
            # link ids for selected link names
            return [self.get_link_id(lname) for lname in link_names]

        return [lid for _, lid in self.link_dict.items()]

    def get_link_name(self, link_id: int) -> str:
        """
        Gets link name of given link_id of this robot.

        Returns:
            (str): the name associated with link_id
        """

        return self.reverse_link_dict[link_id]

    def get_link_names(self) -> List[str]:
        """
        Gets the list of link names of this robot.

        Returns:
            (List[str]): list of link names of this robot, as loaded by the physics engine.
        """

        return [lname for lname, _ in self.link_dict.items()]

    def get_link_pose(self, link_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets the current pose of a given link identified by its link_id

        Args:
            link_id (int): link identifier.
        Returns:
            (Tuple[numpy.ndarray, numpy.ndarray]): tuple containing a position shape-(3,) and unit quaternion orientation shape-(4,) representing the pose of the chosen link.
        """

        if link_id < 0:
            return self.get_base_pose()

        link_state = p.getLinkState(
            self.id,
            link_id,
            computeForwardKinematics=True,
            physicsClientId=self.phys_id,
        )

        pos = np.asarray(link_state[4])
        ori = np.asarray(link_state[5])  # jpl convention

        return pos, ori

    def get_link_pose_by_name(self, link_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets the current pose of a given link identified by its link name

        Args:
            link_id (str): link name.
        Returns:
            (Tuple[numpy.ndarray, numpy.ndarray]): tuple containing a position shape-(3,) and unit quaternion orientation shape-(4,) representing the pose of the chosen link.
        """

        link_id = self.link_dict[link_name]

        return self.get_link_pose(link_id)

    def get_dynamics_info(self, link_name) -> BulletDynamicsInfo:
        """
        Gets dynamics information of chosen link as per loaded from URDF definition

        Args:
            link_id (str): link name.
        Returns:
            (BulletDynamicsInfo): dynamics information for chose link
        """

        link_id = self.get_link_id(link_name)

        info = p.getDynamicsInfo(self.id, link_id, physicsClientId=self.phys_id)

        return BulletDynamicsInfo(*info)

    def get_link_id(self, link_name: str) -> int:
        """
        Gets the link ID from a given link name

        Args:
            link_id (str): link name.
        Returns:
            (int): link identifier
        """
        return self.link_dict[link_name]

    def get_link_velocity(self, link_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets the velocity of given link identifier.

        Args:
            link_id (int): link identifier
        Returns:
            (Tuple[numpy.ndarray, numpy.ndarray]): tuple containing two vectors of shape-(3,), respectively, linear and angular velocity vectors.
        """

        link_state = p.getLinkState(
            self.id, link_id, computeLinkVelocity=True, physicsClientId=self.phys_id
        )

        lin_vel = np.asarray(link_state[6])
        ang_vel = np.asarray(link_state[7])

        return lin_vel, ang_vel

    ## RobotInterface properties/getters

    @property
    def q_mean(self):
        """
        numpy.ndarray: mean joint angles, i.e. ``(upper+lower)*0.5``
        """
        return self._q_mean

    @property
    def n_joints(self):
        """
        int: total number of joints that this robot has.
        """
        return self._n_joints

    @property
    def angles(self):
        """
        numpy.ndarray: current joint angles/positions of this robot.
        """
        return self.get_joint_state()[0]

    def joint_angle(self, joint_idx: int) -> float:
        """
        Gets the current joint angle of a specified joint index.

        Args:
            joint_idx (int): joint index.
        Returns:
            (float): current joint angle/position for the specified joint index.
        """
        # [pos, vel, reac_forces, effort]
        joint_state = p.getJointState(self.id, joint_idx, physicsClientId=self.phys_id)

        state = [joint_state[0], joint_state[1], joint_state[3]]

        return state

    @property
    def joint_limits(self):
        """
        Tuple[List[float], List[float]]: pre-computed joint limits, as defined in URDF file.
        """
        return self._ll, self._ul

    @property
    def joint_names(self):
        """
        List[str]: list of joint names of this robot.
        """
        return self._joint_names

    @property
    def joint_velocities(self):
        """
        numpy.ndarray: current joint velocities of this robot.
        """
        return self.get_joint_state()[1]

    @property
    def joint_efforts(self):
        """
        numpy.ndarray: current joint efforts of this robot.
        """
        return self.get_joint_state()[3]

    @property
    def ee_pose(self):
        """
        Tuple[numpy.ndarray, numpy.ndarray]: position vector shape-(3,) and quaternion shape-(4,) representing the pose of this robot's end-effector.
        """
        return self.get_link_pose(self.ee_index)

    @property
    def ee_velocity(self):
        """
        Tuple[numpy.ndarray, numpy.ndarray]: tuple containing two vectors of shape-(3,), respectively, linear and angular velocity vectors of the end-effector.
        """
        return self.get_link_velocity(link_id=self.ee_index)

    @property
    def base_pose(self):
        """
        Tuple[numpy.ndarray, numpy.ndarray]: position vector shape-(3,) and quaternion shape-(4,) representing the pose of this robot base frame.
        """
        return self.get_base_pose()

    @property
    def base_velocity(self):
        """
        Tuple[numpy.ndarray, numpy.ndarray]: tuple containing two vectors of shape-(3,), respectively, linear and angular velocity vectors of the robot base.
        """
        return self.get_base_velocity()

    @property
    def kinematics(self) -> "grip.robot.RobotKinematics":
        """
        RobotKinematics: the default kinematics of this robot. May be None if has_kinematics=False at init.
        """
        return self._kin

    def state(self) -> Dict:
        """
        Returns the current state of this robot, which includes its joint names, positions, velocities, efforts, end-effector pose and a pointer to self.

        Returns:
            (dict): dictionary with the following format ``dict { 'joint_names': List[str], 'positions': numpy.ndarray, 'velocities': numpy.ndarray, 'efforts': numpy.ndarray, 'ee_pose': Tuple[numpy.ndarray, numpy.ndarray], 'entity': grip.robot.Entity}``
        """

        state = dict()

        state["joint_names"] = self.joint_names
        state["positions"] = self.angles
        state["velocities"] = self.joint_velocities
        state["efforts"] = self.joint_efforts
        state["ee_pose"] = self.ee_pose
        state["entity"] = self

        return state

    def reset(self) -> None:
        """
        Resets robot angles to its home position.
        """

        self.reset_angles(self.home_positions)
        self.set_angles(self.home_positions)

    def forward_kinematics(self, angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        If this robot has kinematics enabled, then this will return the forward kinematic solution for the given angles.

        Args:
            angles (numpy.ndarray): input joint angles.
        Returns:
            (Tuple[numpy.ndarray, numpy.ndarray]): tuple containing a position shape-(3,) and unit quaternion orientation shape-(4,) as the pose of the end-effector for this robot given the input angles.
        """
        assert self._kin, "Robot Kinematics has not been setup"

        position, orientation = self._kin.forward_kinematics(angles)

        return position, orientation

    def inverse_kinematics(self, position: np.ndarray, orientation: np.ndarray):
        """
        If this robot has kinematics enabled, inverse kinematic solution to reache the position and orientation given as input.

        Args:
            position (numpy.ndarray): desired position array shape(3,).
            orientation (numpy.ndarray): desired orientation quaternion array shape(4,).
        Returns:
            (numpy.ndarray): inverse kinematic joint positions solution for reaching the desired end-effector position and orientation
        """
        assert self._kin, "Robot Kinematics has not been setup"

        joint_angles = self._kin.inverse_kinematics(position, orientation)

        begin = self.joint_idx0
        end = self.joint_idx0 + self.n_joints
        return np.array(joint_angles)[begin:end]

    def accurate_inverse_kinematics(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        residual_threshold: float = 1e-4,
        max_iteration: int = 1000,
    ):
        """
        Solve ik to taret pose, return valid joint angles only when ik is accurate
        with residual value smaller than the specified threshold

        Args:
            position (numpy.ndarray): desired position array shape(3,).
            orientation (numpy.ndarray): desired orientation quaternion array shape(4,).
            residual_threshold (float): minimum translation residual value for ik to considered accurate
            max_iteration (int): maximum allowed iteration to the ik solving
        Returns:
            (numpy.ndarray or None): inverse kinematic joint positions solution for reaching the desired end-effector position and orientation. When no accurate solution is found, None is returned

        """
        n_iter = 0
        while n_iter < max_iteration:
            joint_angles = self.inverse_kinematics(position, orientation)
            ik_pos, ik_orn = self.forward_kinematics(joint_angles)
            tmp_residual = np.linalg.norm(position - ik_pos)
            close_enough = tmp_residual < residual_threshold
            if close_enough:
                return joint_angles

            n_iter += 1
        return None

    #### Methods with side effects below ###

    def reset_base_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        """
        Sets the pose of the base-link of this robot.

        Args:
            position (numpy.ndarray): desired position array shape(3,).
            orientation (numpy.ndarray): desired orientation quaternion array shape(4,).
        """
        if self._base_cid is not None:
            p.changeConstraint(
                self._base_cid, position, orientation, maxForce=self.max_force
            )
        else:
            p.resetBasePositionAndOrientation(
                self.id, position, orientation, physicsClientId=self.phys_id
            )

        if self._kin:
            self._kin.set_base_pose(position, orientation)

        self._position = position
        self._orientation = orientation

    def set_link_pose(
        self, link_name: str, position: np.ndarray, orientation: np.ndarray
    ) -> None:
        """
        Sets a corresponding base pose such that desired link pose matches position and orientation.

        Args:
            position (numpy.ndarray): desired position array shape(3,).
            orientation (numpy.ndarray): desired orientation quaternion array shape(4,).
        """

        transform_wb_out = self.get_link_ref_base_pose(
            "-1base", link_name, position, orientation
        )

        self.reset_base_pose(*transform_wb_out)

    def get_link_ref_base_pose(
        self,
        base_link_name: str,
        link_name: str,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets a corresponding base pose such that desired link pose matches position and orientation.

        Args:
            base_link_name (numpy.ndarray): base reference link name
            link_name (numpy.ndarray): desired link name
            position (numpy.ndarray): desired position array shape(3,).
            orientation (numpy.ndarray): desired orientation quaternion array shape(4,).
        Returns:
            (Tuple[np.ndarray, np.ndarray]): pose of link 'base_link_name' must have
                                            such that child link 'link_name' matches given position and orientation
        """

        # T_{base}^{world}
        transform_wb = self.get_link_pose_by_name(base_link_name)

        # T_{link}^{world}
        transform_wl = self.get_link_pose_by_name(link_name)

        # T_{world}^{link}
        transform_lw = invert_transform(*transform_wl)

        # T_{base}^{link}
        transform_lb = multiply_transform(
            transform_lw[0], transform_lw[1], transform_wb[0], transform_wb[1]
        )

        # Output: T_{world}^{base} (corresponding base pose such that link pose matches desired position and orientation)
        transform_wb_out = multiply_transform(
            position, orientation, transform_lb[0], transform_lb[1]
        )

        return transform_wb_out

    def reset_angles(self, angles: np.ndarray) -> None:
        """
        Resets the joint positions of this robot.

        Args:
            angles (numpy.ndarray): desired joint positions
        """
        for i, idx in enumerate(self.joint_ids):
            p.resetJointState(self.id, idx, angles[i], 0, physicsClientId=self.phys_id)

    def move_to_home(self) -> None:
        """
        Sends robot to its rest/home pose (rp)
        """
        self.set_angles(self.home_positions)

    def set_angles(
        self,
        angles: np.ndarray,
        joint_names: List[str] = None,
        positionGain: float = 0.03,
        velocityGain: float = 1,
    ) -> None:
        """
        Sends robot to desired joint angles using position control (internal Bullet PD Controller).

        Args:
            angles (numpy.ndarray): desired joint angles/positions shape-(N,), where N <= ndof. Typically you want to control all joints of the robot, so N=ndof.
            joint_names (List[str], optional): desired joint names to be controlled shape-(N,) (it could be a subset of the total number of joints (non mimic joints) of the robot). Important: has to match the shape of angles argument.
            positionGain (float): position gain (proportional gain of PD controller). Defaults to 0.03.
            velocityGain (float): derivative gain of the PD controller. Defaults to 1 (critically damped).
        """
        if len(angles) == 0:
            return

        joint_ids = self.actuated_joint_ids
        if joint_names is None:
            joint_names = self.actuated_joint_names
        else:
            joint_ids = [self.actuated_joint_dict[name]["idx"] for name in joint_names]

        n = len(joint_ids)

        # log.debug("Angle cmd: {}".format(angles))
        # log.debug("Joint IDs cmd: {}".format(joint_ids))
        # log.debug("Joint names cmd: {}".format(joint_names))
        # print("Angle cmd: {}".format(angles))
        # print("Joint IDs cmd: {}".format(joint_ids))
        # print("Joint names cmd: {}".format(joint_names))
        assert len(angles) == len(joint_names) and len(joint_ids) == len(
            angles
        ), "Number joint angle commands must match number of joint names"

        p.setJointMotorControlArray(
            bodyIndex=self.id,
            jointIndices=joint_ids,
            controlMode=p.POSITION_CONTROL,
            targetPositions=angles,
            targetVelocities=[0] * n,
            forces=[self.max_force] * n,  # 10000, 150
            positionGains=[positionGain] * n,  # 0.03,
            velocityGains=[velocityGain] * n,
            physicsClientId=self.phys_id,
        )

    def set_joint_velocities(self, cmd: np.ndarray, joint_names: List[str] = None):
        """
        Sends joint velocity control commands to the robot.

        Args:
            cmd (numpy.ndarray): desired joint velocities shape-(N,), where N <= ndof. Typically you want to control all joints of the robot, so N=ndof.
            joint_names (List[str], optional): desired joint names to be controlled shape-(N,) (it could be a subset of the total number of joints (non mimic joints) of the robot). Important: has to match the shape of cmd argument.
        """
        if len(cmd) == 0:
            return

        joint_ids = self.actuated_joint_ids
        n = len(joint_ids)

        if joint_names is None:
            joint_names = self.actuated_joint_names
        else:
            joint_ids = [self.actuated_joint_dict[name]["idx"] for name in joint_names]

        assert len(cmd) == len(joint_names) and len(joint_ids) == len(
            cmd
        ), "Number joint velocity commands must match number of joint names"

        p.setJointMotorControlArray(
            self.id,
            joint_ids,
            controlMode=p.VELOCITY_CONTROL,
            forces=[self.max_force] * n,
            targetVelocities=cmd,
            physicsClientId=self.phys_id,
        )

    def set_joint_torques(self, cmd: np.ndarray, joint_names: List[str] = None) -> None:
        """
        Sends joint torque control commands to the robot.

        Args:
            cmd (numpy.ndarray): desired joint velocities shape-(N,), where N <= ndof. Typically you want to control all joints of the robot, so N=ndof.
            joint_names (List[str], optional): desired joint names to be controlled shape-(N,) (it could be a subset of the total number of joints (non mimic joints) of the robot). Important: has to match the shape of cmd argument.
        """

        if len(cmd) == 0:
            return

        joint_ids = self.actuated_joint_ids
        if joint_names is None:
            joint_names = self.actuated_joint_names
        else:
            joint_ids = [self.actuated_joint_dict[name]["idx"] for name in joint_names]

        assert len(cmd) == len(joint_names) and len(joint_ids) == len(
            cmd
        ), "Number joint torque commands must match number of joint names given"

        p.setJointMotorControlArray(
            self.id,
            joint_ids,
            controlMode=p.TORQUE_CONTROL,
            forces=cmd,
            physicsClientId=self.phys_id,
        )

    def enable_torque_mode(
        self, joint_names: List[str] = None, enable: bool = True
    ) -> None:
        """
        By default, each revolute joint and prismatic joint is motorized using a velocity motor. You can disable those default motor by using a maximum force of 0. This will let you perform torque control.
        Note: If torque control is enabled, it is better to call stepsimulation in the control loop, and disable realtime sim of the pybullet world.

        Args:
            joint_names (List[str], optional): desired joint names to be controlled shape-(N,) (it could be a subset of the total number of joints of the robot). Important: has to match the shape of cmd argument.
            enable (bool): whether to enable or disable torque control for the specified joints.
        """
        if joint_names is None:
            joint_ids = self.actuated_joint_ids
        else:
            joint_ids = [self.actuated_joint_dict[name]["idx"] for name in joint_names]

        if enable:
            p.setJointMotorControlArray(
                self.id,
                joint_ids,
                controlMode=p.VELOCITY_CONTROL,
                forces=np.ones(len(joint_ids)) * self.joint_friction,
            )
        else:
            p.setJointMotorControlArray(
                self.id,
                joint_ids,
                controlMode=p.VELOCITY_CONTROL,
                forces=np.ones(len(joint_ids)) * self.max_force,
            )

    def set_ee_pose(self, pos: np.ndarray, quat: np.ndarray) -> None:
        """
        Cartesian position control. Sends robot's end-effector to desired cartesian pose (position, quaternion) using position control.

        Args:
            pos (numpy.ndarray): desired end-effector position.
            quat (numpy.ndarray): desired end-effector orientation.
        """
        joint_angles = self.inverse_kinematics(pos, quat)

        self.set_angles(joint_angles)

    def exec_cartesian_cmd(self, cmd_p: np.ndarray, cmd_q: np.ndarray) -> None:
        """
        Cartesian position control. Sends robot's end-effector to desired cartesian pose (position, quaternion) using position control.

        Args:
            cmd_p (numpy.ndarray): desired end-effector position.
            cmd_q (numpy.ndarray): desired end-effector orientation.
        """

        self.set_ee_pose(cmd_p, cmd_q)

    def exec_cartesian_delta_cmd(self, cmd_dp: np.ndarray, cmd_dq: np.ndarray) -> None:
        """
        Cartesian position control. Sends robot's end-effector to desired cartesian displacement pose.

        Args:
            cmd_dp (numpy.ndarray): desired end-effector position displacement.
            cmd_dq (numpy.ndarray): desired end-effector orientation displacement.
        """
        ee_p, ee_q = self.ee_pose

        cmd_p = np.add(ee_p, cmd_dp)
        cmd_q = quat_multiply(ee_q, cmd_dq)
        log.info("cmd_p: {} cmd_q {}".format(cmd_dp, cmd_dq))

        # log.info("Close {}".format(np.allclose(cmd_p,ee_p)))
        self.set_ee_pose(cmd_p, cmd_q)

    def exec_position_cmd(self, cmd: np.ndarray) -> None:
        """
        Sends robot to desired joint angles using position control (internal Bullet PD Controller).

        Args:
            cmd (numpy.ndarray): desired joint angles/positions shape-(ndof,).
        """
        self.set_angles(cmd)

    def exec_position_cmd_delta(self, cmd: np.ndarray) -> np.ndarray:
        """
        Sends robot to desired joint angles using position control displacement command (internal Bullet PD Controller).

        Args:
            cmd (numpy.ndarray): desired joint relative displacement shape-(ndof,).
        """
        angles = self.angles

        position_cmd = np.add(angles, cmd)

        self.set_angles(position_cmd)

    def exec_velocity_cmd(self, cmd: np.ndarray) -> None:
        """
        Sends robot to desired joint velocity commands to the robot.

        Args:
            cmd (numpy.ndarray): desired joint velocities shape-(ndof,).
        """
        self.set_joint_velocities(cmd)

    def exec_torque_cmd(self, cmd: np.ndarray) -> None:
        """
        Sends robot to desired joint torque commands to the robot.

        Args:
            cmd (numpy.ndarray): desired joint torques shape-(ndof,).
        """
        self.set_joint_torques(cmd)

    def jacobian(self, joint_angles: np.ndarray = None) -> np.ndarray:
        """
        Gets jacobian of the robot at specified joint_angles, or current if joint_angles=None.

        Args:
            joint_angles (numpy.ndarray, optional): if different than None, retrieves jacobian matrix at specified joint_angles array with shape-(ndof,).
        Returns:
            (numpy.ndarray): manipulator jacobian matrix shape(6, ndof).
        """

        begin = self.joint_idx0
        end = self.joint_idx0 + self.n_joints

        if joint_angles is None:
            joint_angles = self.angles
            # log.info("Jacobian::joint_angles {}".format(joint_angles))

        n = len(joint_angles)

        positions = joint_angles
        if n < self._ndof:
            positions = np.zeros(self._ndof)
            # log.info("Jacobian::positions {}".format(positions))
            # log.info("Jacobian::joint_ids: {}".format(self.joint_ids))
            positions[begin:end] = joint_angles

        lin_jac, ang_jac = p.calculateJacobian(
            bodyUniqueId=self.id,
            linkIndex=self.ee_index,
            localPosition=[0.0, 0.0, 0.0],
            objPositions=list(positions),
            objVelocities=[0] * self._ndof,
            objAccelerations=[0] * self._ndof,
            physicsClientId=self.phys_id,
        )

        lin_jac = np.array(lin_jac)
        ang_jac = np.array(ang_jac)

        jac = np.vstack([lin_jac, ang_jac])[:, begin:end]

        return jac

    def inertia(self, joint_angles: np.ndarray = None) -> np.ndarray:
        """
        Args:
            joint_angles (numpy.ndarray, optional): optional parameter, if None, then returned inertia is evaluated at current joint_angles. Otherwise, returned inertia tensor is evaluated at current joint angles.

        Return:
            (numpy.ndarray): Joint space inertia tensor
        """

        begin = self.joint_idx0
        end = self.joint_idx0 + self.n_joints

        if joint_angles is None:
            joint_angles = self.angles

        inertia_tensor = np.array(
            p.calculateMassMatrix(
                self.id, list(joint_angles), physicsClientId=self.phys_id
            )
        )

        return inertia_tensor[begin:end, begin:end]

    def random_texture(self) -> None:
        """
        If texture randomiser is present, randomises this robot's texture. Useful for domain randomisation applications.
        """
        if self.texture_randomiser is not None:
            self.texture_randomiser.randomise(self)

    def enable_torque_sensor(self, enable: bool, joint_id: int = None) -> None:
        """
        Enable or disable torque_sensor.
        """
        joint = self.actuated_joint_ids[-1] if joint_id is None else joint_id
        p.enableJointForceTorqueSensor(
            self.id, joint, enable, physicsClientId=self.phys_id
        )

    def get_contact_force(
        self, target_links: list, ignore_self_contact: bool = True
    ) -> np.ndarray:
        """
        Calculate contact forces applied onto given link(s)

        Args:
            target_links (int): list of target link indexes to inspect contact
            ignore_self_contact (bool): if self contact should be countered
        Returns:
            (numpy.ndarray): total contact force on target links
        """
        contact_force = np.array([0.0, 0.0, 0.0])
        all_contacts = self.contact_points()
        for contact in all_contacts:
            (
                unused_flag,
                body_a_id,
                body_b_id,
                link_a_id,
                unused_link_b_id,
                unused_pos_on_a,
                unused_pos_on_b,
                contact_normal_b_to_a,
                unused_distance,
                normal_force,
                friction_1,
                friction_direction_1,
                friction_2,
                friction_direction_2,
            ) = contact
            if (ignore_self_contact) and (body_b_id == body_a_id):
                continue  # Ignore self contacts
            if link_a_id in target_links:
                tmp_contact_force = np.array(contact_normal_b_to_a) * normal_force
                contact_force += tmp_contact_force
        return contact_force

    def get_friction_force(self, target_links: list) -> None:
        """
        Calculate lateral friction forces applied onto given link(s)

        Args:
            target_links (int): list of target link indexes to inspect lateral friction
        Returns:
            (numpy.ndarray): total lateral friction force on target_links
        """

        friction_force = np.array([0.0, 0.0, 0.0])
        all_contacts = self.contact_points()
        for contact in all_contacts:
            (
                unused_flag,
                body_a_id,
                body_b_id,
                link_a_id,
                unused_link_b_id,
                unused_pos_on_a,
                unused_pos_on_b,
                contact_normal_b_to_a,
                unused_distance,
                normal_force,
                friction_1,
                friction_direction_1,
                friction_2,
                friction_direction_2,
            ) = contact
            if body_b_id == body_a_id:
                continue  # Ignore self contacts
            elif link_a_id in target_links:
                tmp_friction_force = (
                    np.array(friction_direction_1) * friction_1
                    + np.array(friction_direction_2) * friction_2
                )
                friction_force += tmp_friction_force
        return friction_force

    def attach_object(self, link_name: str, entity: Entity) -> None:
        """
        Attaches entity to selected link of this robot.

        Args:
            link_name (str): link name of this robot to attach entity to.
            entity (grip.robot.Entity): entity to attach to specified link.
        """
        object_pos, object_ori = entity.pose
        link_pos, link_ori = self.get_link_pose_by_name(link_name)

        ilp, ilq = invert_transform(link_pos, link_ori)

        rp, rq = multiply_transform(ilp, ilq, object_pos, object_ori)

        entity.create_constraint(rp, rq, self, link_name)

    def detach_object(self, entity: Entity) -> None:
        """
        Detaches entity from this robot (assuming it has been attached before via self.attach_object method)
        """
        entity.remove_constraint()

    def disable_collision(self) -> None:
        """
        Disable collisions between this robot and everything else.
        """
        p.setCollisionFilterGroupMask(self.id, -1, 0, 0, physicsClientId=self.phys_id)
        link_ids = self.get_link_ids()

        for lid in link_ids:
            p.setCollisionFilterGroupMask(
                self.id, lid, 0, 0, physicsClientId=self.phys_id
            )

    def set_collision_filter(self, entity: Entity, enable: bool):
        link_ids = self.get_link_ids()

        for lid in link_ids:
            p.setCollisionFilterPair(
                self.id, entity.id, lid, -1, int(enable), physicsClientId=self.phys_id
            )

            if isinstance(entity, BulletRobot):
                entity_link_ids = entity.get_link_ids()
                for elid in entity_link_ids:
                    p.setCollisionFilterPair(
                        self.id,
                        entity.id,
                        lid,
                        elid,
                        int(enable),
                        physicsClientId=self.phys_id,
                    )

    def contact_points(self):
        """
        Returns contact points between this robot and anything that it may be contacting

        Returns:
            contact points: [bullet_contacts]
                returns list of bullet contact points
                see: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.cb0co8y2vuvc

        """

        contact_points = p.getContactPoints(self.id, physicsClientId=self.phys_id)

        return contact_points

    def aabb(
        self, link_id: int = -1, expansion: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets axis-aligned bounding box of a given link of this robot.
        Args:
            link_id (int, optional): link id of this multi-body. Defaults to -1, meaning base link.
        Returns:
            (Tuple[numpy.ndarray, numpy.ndarray]): a tuple of 3-d vectors (aabb_min, aabb_max) representing the minimum and maximum bounds in world frame.
        """
        assert link_id == -1 or link_id in self.get_link_ids()

        aabb = p.getAABB(self.id, link_id, physicsClientId=self.phys_id)

        aabb_min = aabb[0]
        aabb_max = aabb[1]

        return np.array(aabb_min) - expansion, np.array(aabb_max) + expansion

    def create_constraint(
        self,
        position: np.ndarray = np.zeros(3),
        orientation: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0]),
        max_force: float = 200,
    ) -> None:
        """
        Creates a rigid body constraint for the base frame of this body.

        Args:
            position (numpy.ndarray): the orientation of the joint frame relative to parent center of mass coordinate frame
            orientation (numpy.ndarray): the orientation of the joint frame relative to the world origin coordinate frame
        """
        if self._base_cid is not None:
            return

        self._position_des = position
        self._orientation_des = orientation
        self._base_cid = p.createConstraint(
            self.id,
            -1,
            -1,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            position,  # [0.500000, 0.300006, 0.700000],
            self._orientation_des,
            physicsClientId=self.phys_id,
        )

        # Note: the mysterious `erp` is of EXTREME importance
        p.changeConstraint(
            self._base_cid, maxForce=max_force, erp=0.8, physicsClientId=self.phys_id
        )

    def remove_constraint(self) -> None:
        """Removes constraint created with self.create_constraint"""
        if self._base_cid is None:
            return

        p.removeConstraint(self._base_cid, physicsClientId=self.phys_id)

        self._base_cid = None

    def setup_mimic_joints(
        self,
        mimic_parent_joint_name: str,
        mimic_children: List[str],
        maxForce: float = 100,
    ) -> None:
        """Sets up mimic joints by creating gear joint constraints between parent and children joints

        Args:
            mimic_parent_joint_name (str): parent joint name to be mimicked
            mimic_children (List[str]): joint names of child joints that will be mimicking the desired parent joint.
        """

        if mimic_parent_joint_name not in self.joint_dict:
            log.warning(
                f"Impossible to setup mimic joints - parent joint {mimic_parent_joint_name} not found!"
            )
            return
        self.mimic_parent_id = self.joint_dict[mimic_parent_joint_name]["idx"]
        self.id2mimic_multiplier = {}
        # Preliminary check: all mimic_children joints needs to be available to be coupled (no action must be taken bofore it)
        for joint_name in mimic_children:
            if joint_name not in self.actuated_joint_dict:
                log.warning(
                    f"Impossible to setup mimic joints - child joint {joint_name} not found or already coupled with another parent joint!"
                )
                return
        for joint_name in mimic_children:
            joint_id = self.actuated_joint_dict[joint_name]["idx"]
            mimic_multiplier = mimic_children[joint_name]
            self.id2mimic_multiplier[joint_id] = mimic_multiplier
            log.debug("Removing '{}' from actuated joint list".format(joint_name))
            self.actuated_joint_names.remove(joint_name)
            self.actuated_joint_ids.remove(joint_id)
            del self.actuated_joint_dict[joint_name]

        for joint_id, multiplier in self.id2mimic_multiplier.items():
            c = p.createConstraint(
                self.id,
                self.mimic_parent_id,
                self.id,
                joint_id,
                jointType=p.JOINT_GEAR,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
                physicsClientId=self.phys_id,
            )
            # Note: the mysterious `erp` is of EXTREME importance
            p.changeConstraint(
                c,
                gearRatio=-multiplier,
                maxForce=maxForce,
                erp=1.0,
                physicsClientId=self.phys_id,
            )

        log.debug(
            "Actuated joint names (updated): {}".format(self.actuated_joint_names)
        )
        log.debug("Actuated joint IDs (updated): {}".format(self.actuated_joint_ids))

    def set_colour(self, rgba_colour: np.ndarray) -> None:
        """
        Sets robot colour.

        Args:
            rgba_colour: color components for RED, GREEN, BLUE and ALPHA, each in range [0..1]. Alpha has to be 0 (invisible) or 1 (visible) at the moment. Note that TinyRenderer doesn't support transparancy, but the GUI/EGL OpenGL3 renderer does.
        """
        shape_data = [
            VisualShapeData(*shape)
            for shape in p.getVisualShapeData(self.id, physicsClientId=self.phys_id)
        ]

        for shape in shape_data:
            p.changeVisualShape(
                self.id,
                shape.link_id,
                rgbaColor=rgba_colour,
                physicsClientId=self.phys_id,
            )

    def set_link_colour(self, rgba_colour: np.ndarray, link_ids: list[int]) -> None:
        """
        Sets robot link colour.

        Args:
            rgba_colour: color components for RED, GREEN, BLUE and ALPHA, each in range [0..1]. Alpha has to be 0 (invisible) or 1 (visible) at the moment. Note that TinyRenderer doesn't support transparancy, but the GUI/EGL OpenGL3 renderer does.
            link_ids: link indices to set desired colour
        """

        shape_link_ids = [
            VisualShapeData(*shape).link_id
            for shape in p.getVisualShapeData(self.id, physicsClientId=self.phys_id)
        ]

        assert all(
            lid in shape_link_ids for lid in link_ids
        ), "Link ids must be valid and exist/belong to this robot."

        for lid in link_ids:
            p.changeVisualShape(
                self.id, lid, rgbaColor=rgba_colour, physicsClientId=self.phys_id
            )


class RobotKinematics(object):
    def __init__(self, **kwargs):
        self.world = BulletWorld(phys_opt="direct")

        self._parent = kwargs.get("parent", None)

        log.debug("Parent: {}".format(self._parent.__dict__))
        log.debug("kinematics URDF: {}".format(self._parent.urdf_file))

        self.robot_arm = BulletRobot(
            urdf_file=self._parent.urdf_file,
            ee_index=self._parent.ee_index,
            joint_names=self._parent.joint_names,
            phys_id=self.world.id,
            position=self._parent._position,
            orientation=self._parent._orientation,
            home_positions=self._parent.home_positions,
            has_kinematics=False,
        )

    def _release(self):
        if self.world is not None and self.world.is_connected():
            self.world.disconnect()

        self.world = None

    def __del__(self):
        self._release()

    def set_base_pose(self, position, orientation):
        self.robot_arm.reset_base_pose(position, orientation)

    def forward_kinematics(self, angles) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the forward kinematic solution for the given angles.

        Args:
            angles (numpy.ndarray): input joint angles.
        Returns:
            (Tuple[numpy.ndarray, numpy.ndarray]): tuple containing a position shape-(3,) and unit quaternion orientation shape-(4,) as the pose of the end-effector for this robot given the input angles.
        """
        self.robot_arm.reset_angles(angles)

        return self.robot_arm.ee_pose

    def inverse_kinematics(self, position, orientation, seed=None) -> np.ndarray:
        """
        Computes and returns inverse kinematic solution to reach the position and orientation given as input.

        Args:
            position (numpy.ndarray): desired position array shape(3,).
            orientation (numpy.ndarray): desired orientation quaternion array shape(4,).
            seed (numpy.ndarray): Optional null-space IK rest poses.
        Returns:
            (numpy.ndarray): inverse kinematic joint positions solution for reaching the desired end-effector position and orientation
        """
        self.robot_arm.reset_angles(self._parent.angles)

        if seed is None:
            seed = self.robot_arm.angles  # self.robot_arm._rp

        joint_angles = p.calculateInverseKinematics(
            self.robot_arm.id,
            self.robot_arm.ee_index,
            position,
            orientation,
            self.robot_arm._ll,
            self.robot_arm._ul,
            self.robot_arm._jr,
            seed,
            jointDamping=[0.001] * self.robot_arm._ndof,
            maxNumIterations=1000,
            residualThreshold=1e-9,
            physicsClientId=self.world.id,
        )

        return np.array(joint_angles)
