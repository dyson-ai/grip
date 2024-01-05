#!/usr/bin/env python3

import time
import grip.robot as robot
from grip.robot import EndEffectorRegistry
from grip.ros import xacro_to_urdf_path

if __name__ == "__main__":
    world = robot.BulletWorld(phys_opt="gui")

    urdf_path = xacro_to_urdf_path(
        "robotiq_140_description",
        "robotiq_140_gripper_instance.urdf.xacro",
        "/tmp/robotiq_2f140.urdf",
        pkg_subfolder="urdf",
        xacro_args="hand:=true",
    )

    gripper = EndEffectorRegistry.make(
        "robotiq_2f140", urdf_file=urdf_path, phys_id=world.id, sliders=True
    )

    toggle_gripper = robot.BulletButton(world.id, "Toggle Gripper")
    is_closed = False
    while world.is_connected():
        if toggle_gripper.was_pressed():
            is_closed = not is_closed

            gripper.control_fingers("close" if is_closed else "open")

        time.sleep(0.1)
