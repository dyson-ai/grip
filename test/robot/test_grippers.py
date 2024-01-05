#!/usr/bin/env python3

import unittest
import numpy as np
import grip.robot as robot
from grip.robot import EndEffectorRegistry


class TestGrippers(unittest.TestCase):
    world = None
    physics_iterations = 150
    tol = 0.01
    grippers = ["panda_hand"]

    @classmethod
    def setUpClass(cls):
        cls.world = robot.BulletWorld(phys_opt="direct")

    def test_close_grippers(self):
        for gripper_name in self.grippers:
            print("testing gripper: ", gripper_name)

            gripper = EndEffectorRegistry.make(gripper_name, phys_id=self.world.id)

            target = gripper.close_angles
            gripper.set_angles(target)

            for _ in range(self.physics_iterations):
                self.world.step()

            joint_state = gripper.angles

            self.assertTrue(
                np.isclose(np.linalg.norm(joint_state - target), 0.0, atol=self.tol)
            )

            self.world.reset()


if __name__ == "__main__":
    unittest.main()
