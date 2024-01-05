#!/usr/bin/env python3

import time
import numpy as np

import grip
import os
import pybullet_data as pd


if __name__ == "__main__":
    world = grip.robot.BulletWorld(phys_opt="gui")

    plane = grip.robot.BulletObject(
        urdf_file=os.path.join(pd.getDataPath(), "plane.urdf"), position=[0, 0, 0]
    )

    duck = grip.robot.BulletObject(
        urdf_file=os.path.join(pd.getDataPath(), "duck_vhacd.urdf"),
        position=[0, 0, 0.5],
        scale=8.0,
        fixed=False,
    )

    camera = grip.sensors.RGBDCamera(
        cid=world.id,
        position=np.array([0, 0, 1.5]),
        orientation=grip.math.rpy2quaternion([0.0, np.pi, 0.0]),
    )

    camera.draw_frustum(0.0)

    while True:
        rgb, depth, seg = camera.obs()

        time.sleep(0.1)
