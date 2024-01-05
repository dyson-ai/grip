import os
from ..io import get_data_path, log, import_module, get_package_path
from grip_assets import get_data_path
from grip.robot import BulletRobot
from ..robot import (
    BulletPandaGripper,
)

gd = import_module("grip_data")

base_dir = get_data_path() if gd is None else gd.get_data_path()

PANDA_WITH_DEFAULT_GRIPPER_URDF = os.path.join(
    get_package_path("pybullet_data"), "franka_panda/panda.urdf"
)
PANDA_WITH_ROBOTIQ_2F_85_GRIPPER_URDF = os.path.join(
    base_dir, "urdf/franka_panda/panda_with_robotiq_2f_85_gripper.urdf"
)
PANDA_WITH_ROBOTIQ_2F_140_GRIPPER_URDF = os.path.join(
    base_dir, "urdf/franka_panda/panda_with_robotiq_2f_140_gripper.urdf"
)

gripper_type_dict = {
    "default": (PANDA_WITH_DEFAULT_GRIPPER_URDF, BulletPandaGripper, 11),
    "none": ("", None, -1),
}


class BulletPandaArmGripper(BulletRobot):
    def __init__(self, **kwargs):
        self.bullet_gripper_class = kwargs.get("bullet_gripper_class", None)
        self.gripper_type = kwargs.get("gripper_type", "default")

        default_ee_index = -1

        if self.bullet_gripper_class is None:
            if self.gripper_type not in gripper_type_dict.keys():
                log.info("Gripper type {} is None supported!".format(self.gripper_type))
                self.gripper_type = "default"
            gripper_class = gripper_type_dict[self.gripper_type][1]

            if gripper_class is not None:
                kwargs["urdf_file"] = kwargs.get(
                    "urdf_file", gripper_type_dict[self.gripper_type][0]
                )
                default_ee_index = gripper_type_dict[self.gripper_type][2]

        else:
            kwargs["urdf_file"] = kwargs.get("urdf_file", None)
            gripper_class = self.bullet_gripper_class

        self.has_gripper = gripper_class is not None

        kwargs["ee_index"] = kwargs.get("ee_index", default_ee_index)
        kwargs["tip_link"] = kwargs.get("tip_link", "panda_link8")

        kwargs["joint_names"] = kwargs.get(
            "joint_names", ["panda_joint{}".format(idx) for idx in range(1, 8)]
        )

        kwargs["gripper_max_force"] = kwargs.get("gripper_max_force", 12)

        super().__init__(**kwargs)

        self.gripper = None

        if self.has_gripper:
            log.debug("Gripper class {}".format(gripper_class))
            self.gripper = gripper_class(
                parent=self,
                max_force=kwargs["gripper_max_force"],
                ee_index=self.ee_index,
                urdf_file=None,
                sliders=self._sliders,
            )
            self.gripper.control_fingers(mode="open")

    def state(self):
        state = BulletRobot.state(self)

        if self.gripper is not None:
            state["gripper_state"] = self.gripper.state()

        return state

    def reset(self):
        super().reset()

        self.gripper.reset()
