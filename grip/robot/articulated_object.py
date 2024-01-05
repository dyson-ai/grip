import numpy as np

import pybullet as p
from grip.robot import BulletObject
from grip.io import log


class ArticulatedObject(BulletObject):
    """
    The ArticulatedObject represents an interface to a Bullet object with multiple movable links.
    An object of this class exposes several utilities for configuring the object's joints of the movable links.

    Args:
        category_name (str): name of the object category.
        instance_id (int): id of the object instance in its category.
        movable_link (dict): a dictionary of the movable link info loaded from the object_meta_info.json in the object dataset. Each key is a movable link name and the value is a dict containing {"link_id", "link_name", "joint_type"}.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._category_name = kwargs.get("category_name", None)
        self._instance_id = kwargs.get("instance_id", None)
        self._movable_link = kwargs.get("movable_link", None)

        self.joint_states = dict()
        self.links = dict()
        for joint_id in range(p.getNumJoints(self.id, self._phys_id)):
            joint_info = p.getJointInfo(self.id, joint_id, self._phys_id)

            link_name = joint_info[12].decode("gb2312")
            self.links[link_name] = joint_id
            if link_name not in self._movable_link:
                continue
            self.joint_states[joint_id] = {
                "min_val": joint_info[8],
                "max_val": joint_info[9],
                "init_val": p.getJointState(self.id, joint_id, self._phys_id)[0],
            }

    def set_joint(self, joint_id: int, value: float) -> None:
        """
        Sets the joint value of the specified joint.
        If the target state is out of the joint's movable range, this function will log warning.

        Args:
            joint_id (int): id of the selected joint.
            value (float): the target joint value

        """
        if joint_id not in self.joint_states.keys():
            log.warning("Joint {} does not exist.".format(joint_id))

        elif (
            value < self.joint_states[joint_id]["min_val"]
            or value > self.joint_states[joint_id]["max_val"]
        ):
            log.warning("Specified value out of joint operation range.")

        else:
            p.resetJointState(self.id, joint_id, value, self._phys_id)

    def set_joint_random(self, joint_id: int) -> None:
        """
        Sets the joint value of the specified joint to a random value within its range

        Args:
            joint_id (int): id of the selected joint.

        """
        if joint_id not in self.joint_states.keys():
            log.warning("Joint {} does not exist.".format(joint_id))

        else:
            value = np.random.uniform(
                self.joint_states[joint_id]["min_val"],
                self.joint_states[joint_id]["max_val"],
            )
            p.resetJointState(self.id, joint_id, value, self._phys_id)

    def reset(self) -> None:
        """
        Resets all movable joints to their initial values.

        """
        super().reset()
        for joint_id in self.joint_states.keys():
            self.set_joint(joint_id, self.joint_states[joint_id]["init_val"])

    def check_joint_boundary(self, joint_id: int) -> bool:
        """
        Checks if the selected joint has reached its boundary.

        Args:
            joint_id (int): id of the selected joint.

        """
        reach_boundary = False
        self.joint_states[joint_id]["cur_val"] = p.getJointState(
            self.id, joint_id, self._phys_id
        )[0]
        joint_state = self.joint_states[joint_id]
        joint_info = p.getJointInfo(self.id, joint_id, self._phys_id)
        threshold = 0.1 if joint_info[2] == p.JOINT_REVOLUTE else 0.05
        threshold = min(
            threshold, (joint_state["max_val"] - joint_state["min_val"]) / 3.5
        )

        if abs(joint_state["cur_val"] - joint_state["init_val"]) > threshold and (
            joint_state["cur_val"] < joint_state["min_val"] + threshold
            or joint_state["cur_val"] > joint_state["max_val"] - threshold
        ):
            reach_boundary = True

        return reach_boundary
