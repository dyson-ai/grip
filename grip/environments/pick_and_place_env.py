#!/usr/bin/env python3
import os
import numpy as np
import pybullet as p

import grip
from .template_env import TemplateEnvironment
from .. import robot as g
from ..io import get_data_path


import pybullet_data as pd


base_dir = get_data_path()

TRAY_URDF = os.path.join(pd.getDataPath(), "tray/tray.urdf")
OBJECT_URDF = os.path.join(pd.getDataPath(), "duck_vhacd.urdf")


class PickAndPlaceEnvironment(TemplateEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.text = grip.robot.add_text(
            [-0.5, 0, 1.0],
            text="Py-Grip Package: Table Pick & Place Environment",
            cid=self.world.id,
        )

        self.world.register_keyhandler(ord("r"), self.reset)

    def init(self):
        super().init()

        # return
        # spawn reminder of the objects
        self.random_offset = np.random.randn(3) * 0
        tray_pos = [
            0.3 + self.random_offset[0],
            -0.7 + self.random_offset[1],
            0.0,
        ]
        tray_ori = [0.0, 0.0, 0.0, 1.0]
        self.tray = g.BulletObject(
            world=self.world,
            urdf_file=TRAY_URDF,
            position=tray_pos,
            orientation=tray_ori,
            fixed=True,
            force_concave=True,
            compliant_dyn=True,
        )

        object_rel_pos = [0.0, 0.0, 0.1]
        object_rel_ori = [0.0, 0.0, 0.0, 1.0]

        object_pos, object_ori = grip.math.multiply_transform(
            tray_pos, tray_ori, object_rel_pos, object_rel_ori
        )

        self.object = g.BulletObject(
            world=self.world,
            urdf_file=OBJECT_URDF,
            position=object_pos,
            orientation=object_ori,
            fixed=False,
            compliant_dyn=True,
        )

        self.add("tray", self.tray)
        self.add("object", self.object)


if __name__ == "__main__":
    env = PickAndPlaceEnvironment()

    while p.isConnected():
        env.step(None)
