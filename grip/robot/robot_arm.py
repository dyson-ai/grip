from ..io import log
from grip.robot import BulletRobot
from .end_effector_registry import EndEffectorRegistry


class BulletRobotArm(BulletRobot):
    """A class that specialises a BulletRobot representing a robot arm which can optionally have a gripper as an end-effector"""

    def __init__(
        self, gripper_type: str = None, gripper_max_force: float = 40, **kwargs
    ):
        """BulletRobotArm constructor

        Args:
            gripper_type (str, optional): gripper type name from grip.robot.EndEffectorRegistry to be used as the robot end-effector. Defaults to None, meaning no gripper is created.
            gripper_max_force (float, optional): maximum gripper force. Defaults to 40N.
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
        super().__init__(**kwargs)

        self.gripper_type = gripper_type

        log.info(f"Gripper type: {self.gripper_type}")
        self.has_gripper = (
            self.gripper_type is not None
            and self.gripper_type != "none"
            and EndEffectorRegistry.is_registered(self.gripper_type)
        )

        log.info(
            f"Gripper exists: {EndEffectorRegistry.is_registered(self.gripper_type)}"
        )
        self.gripper = None

        if self.has_gripper:
            log.info(
                "Gripper class {}".format(EndEffectorRegistry.get(self.gripper_type))
            )

            self.gripper = EndEffectorRegistry.make(
                self.gripper_type,
                parent=self,
                max_force=gripper_max_force,
                tip_link=self.tip_link,
                urdf_file=None,
                sliders=self._sliders,
            )

            self.gripper.control_fingers(mode="open")

    def state(self) -> dict:
        """Returns the current state of this robot, which includes its joint names, positions, velocities, efforts, end-effector pose and a pointer to self."""

        state = BulletRobot.state(self)

        if self.gripper is not None:
            state["gripper_state"] = self.gripper.state()

        return state

    def reset(self) -> None:
        """Resets robot angles to its home position."""

        super().reset()

        self.gripper.reset()
