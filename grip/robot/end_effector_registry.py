from typing import Callable, Type

from ..io import import_module
from .gripper_interface import GripperInterface


class EndEffectorType(GripperInterface):
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


class EndEffectorRegistry:
    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Class method to register gripper classes to the internal gripper registry.

        Args:
            name: The functor name
        """

        def inner_wrapper(gripper_class: EndEffectorType) -> Callable:
            if name in cls.registry:
                raise RuntimeError(f"Error: {name} already exists in {cls.__name__}.")
            cls.registry[name] = gripper_class

            return gripper_class

        return inner_wrapper

    @classmethod
    def make(cls, name: str, *args, **kwargs) -> EndEffectorType:
        """
        Class method used to construct a gripper registered with a given name

        Args:
            name: the name of the gripper type
        Returns:
            (EndEffectorType): returns an instance of the end-effector registered with the given name.
        """
        gripper_class = cls.get(name)

        return gripper_class(*args, **kwargs)

    @classmethod
    def get(cls, name: str) -> Type[EndEffectorType]:
        """
        Class method used to get the end-effector type registered with given name.

        Args:
            name: the name of the gripper type
        Returns:
            (Type[EndEffectorType]): returns the class type of the gripper registered with given name.
        """

        if name not in cls.registry:
            raise RuntimeError(f"Error: {name} not registered in {cls.__name__}.")

        gripper_class = cls.registry[name]

        return gripper_class

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Checks if a gripper type with given name is already registered in this registry.

        Args:
            name: the name of the gripper type
        Returns:
            (bool): returns whether or not the type name has been already registered.
        """

        return name in cls.registry


EndEffectorRegistry.registry["none"] = None
