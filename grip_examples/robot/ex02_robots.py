#!/usr/bin/env python3
import os
import time
import grip
import pybullet_data as pd

if __name__ == "__main__":
    world = grip.robot.BulletWorld(phys_opt="gui")

    urdf_path = os.path.join(pd.getDataPath(), "franka_panda/panda.urdf")

    robot = grip.robot.BulletRobot(urdf_file=urdf_path, sliders=True, phys_id=world.id)

    plane = grip.robot.BulletObject(
        urdf_file=os.path.join(pd.getDataPath(), "plane.urdf"), position=[0, 0, 0]
    )

    while world.is_connected():
        positions = robot.joint_sliders.read()

        robot.set_angles(positions)

        time.sleep(0.1)
