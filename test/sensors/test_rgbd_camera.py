#!/usr/bin/python
import os
import unittest
import numpy as np
import grip.robot as robot
import grip.sensors as sensors
import grip.environments as environments
import grip.io as io
import grip.math as math
import pybullet_data as pd


class TestRGBDCamera(unittest.TestCase):
    world = None
    N_OBS = 5

    textures_paths = io.file_list(
        io.get_package_path("grip_assets", "texture"), extension_filter="jpg"
    )

    @classmethod
    def setUpClass(cls):
        cls.world = robot.BulletWorld(phys_opt="direct")

        # Creates light randomiser
        cls.light_randomiser = environments.LightingRandomiser()

        # Creates texture randomiser
        cls.texture_randomiser = environments.TextureRandomiser(
            cls.world, cls.textures_paths
        )

        # Creates camera (it looks down towards the ground plane)
        cls.camera = sensors.RGBDCamera(
            cid=cls.world.id,
            position=np.array([0, 0, 1.0]),
            orientation=math.rpy2quaternion([0.0, np.pi, 0.0]),
        )

        # Creates ground plane object
        cls.plane = robot.BulletObject(
            urdf_file=os.path.join(pd.getDataPath(), "plane.urdf"), position=[0, 0, 0]
        )

    def test_acquire_observation(self):
        """Acquires ground-truth observations"""

        rgb, depth, seg = self.camera.obs()

        self.assertTrue(rgb is not None, "Error: colour image is none")
        self.assertTrue(depth is not None, "Error: depth image is none")
        self.assertTrue(seg is not None, "Error: segmentation mask is none")

    def test_acquire_light_randomised_observation(self):
        """Acquires a number N_OBS of observations with lighting randomised parameters"""

        for _ in range(self.N_OBS):
            rgb, depth, seg = self.camera.obs(self.light_randomiser)

            self.assertTrue(rgb is not None, "Error: colour image is none")
            self.assertTrue(depth is not None, "Error: depth image is none")
            self.assertTrue(seg is not None, "Error: segmentation mask is none")

    def test_acquire_texture_randomised_observation(self):
        """Acquires a number N_OBS of observations after randomising the ground-plane's texture"""

        for _ in range(self.N_OBS):
            self.texture_randomiser.randomise(self.plane)

            rgb, depth, seg = self.camera.obs()

            self.assertTrue(rgb is not None, "Error: colour image is none")
            self.assertTrue(depth is not None, "Error: depth image is none")
            self.assertTrue(seg is not None, "Error: segmentation mask is none")


if __name__ == "__main__":
    unittest.main()
