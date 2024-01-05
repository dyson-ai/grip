import abc
from typing import Protocol
from ..io import import_module


class GripperInterface(Protocol):
    # Type name for real hardware interface
    HW_INTERFACE = "path.to.package:ClassName"

    @classmethod
    def get_hw_interface(cls):
        module_path = cls.HW_INTERFACE.split(":")[0]
        class_name = cls.HW_INTERFACE.split(":")[1]

        module = import_module(module_path)

        assert (
            module is not None
        ), f"Importing module {module_path} has failed. Are you sure it exists?"

        return getattr(module, class_name)

    @abc.abstractmethod
    def control_fingers(self, mode: str) -> None:
        """
        Control fingers

        Args:
            mode (str): mode name, e.g. "close", "open"
        """
        raise NotImplementedError("control_fingers: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def apply_gripper_delta_action(self, action: float) -> None:
        """
        control gripper with a delta finger distance action.

        Args:
            action (float): delta joint angle action for all joints of the gripper
        """
        raise NotImplementedError("control_fingers: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractproperty
    def angles(self) -> "numpy.ndarray":
        """
        Returns current joint angles measured by encoders

        Returns:
            (numpy.ndarray): joint state of the gripper
        """
        raise NotImplementedError("control_fingers: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """

        Returns True (bool) when the gripper is ready for operation.

        Returns:
            (Bool): True for gripper state ready

        """
        raise NotImplementedError("is_ready: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def validate_grasp(self) -> bool:
        """Returns true if the gripper is holding an object

        Returns:
            bool: The gripper has a valid grasp on the object
        """
        raise NotImplementedError(
            "Add method that attempts to close the gripper and check if the gripper is closed"
        )
