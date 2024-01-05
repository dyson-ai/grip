#!/usr/bin/env python3

import time
import grip.robot as robot
from grip.robot import EndEffectorRegistry


if __name__ == "__main__":
    world = robot.BulletWorld(phys_opt="gui")

    gripper = EndEffectorRegistry.make("panda_hand", phys_id=world.id, sliders=True)

    while world.is_connected():
        positions = gripper.joint_sliders.read()

        gripper.set_angles(positions)

        time.sleep(0.1)
