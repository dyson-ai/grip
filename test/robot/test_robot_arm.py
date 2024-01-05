#!/usr/bin/env python3
import os
import unittest
import numpy as np
import grip.robot as robot
import pybullet_data as pd


class TestRobotArm(unittest.TestCase):
    world = None
    sim_steps = 250
    tol = 0.01

    @classmethod
    def setUpClass(cls):
        cls.world = robot.BulletWorld(phys_opt="direct")

        urdf_path = os.path.join(pd.getDataPath(), "franka_panda/panda.urdf")
        cls.robot = robot.BulletRobotArm(
            urdf_file=urdf_path, phys_id=cls.world.id, tip_link="panda_grasptarget"
        )

    def test_move_to_home(self):
        self.robot.move_to_home()

        for _ in range(self.sim_steps):
            self.world.step()

        diff = self.robot.home_positions - self.robot.angles
        error = np.linalg.norm(diff)
        self.assertTrue(
            np.isclose(error, 0.0, atol=self.tol),
            f"Failed to move to home. Error magnitude: {error}",
        )

    def test_move_to_goal(self):
        goal_angles = np.array(self.robot.home_positions)
        goal_angles[0] += 0.5

        self.robot.set_angles(goal_angles)

        for _ in range(self.sim_steps):
            self.world.step()

        diff = goal_angles - self.robot.angles
        error = np.linalg.norm(diff)
        self.assertTrue(
            np.isclose(error, 0.0, atol=self.tol),
            f"Failed to move to desired goal. Error magnitude: {error}",
        )


if __name__ == "__main__":
    unittest.main()
