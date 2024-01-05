#!/usr/bin/env python3

import os
import numpy as np

try:
    import rclpy
    from rclpy.node import Node

except ImportError:
    raise RuntimeError("Unable to run example without ros2 installed.")

import grip
import grip.ros
import pybullet_data as pd


class ROSCameraExample(Node):
    def __init__(self, fps=10):
        super().__init__("rgbd_camera_example_node")

        # Creates bullet world with GUI
        self.world = grip.robot.BulletWorld(phys_opt="gui")

        # Creates ground plane object
        self.plane = grip.robot.BulletObject(
            urdf_file=os.path.join(pd.getDataPath(), "plane.urdf"), position=[0, 0, 0]
        )

        # Creates ROSCamera object
        self.camera = grip.ros.ROSCamera(
            cid=self.world.id,
            position=np.array([0, 0, 1.5]),
            orientation=grip.math.rpy2quaternion([0.0, np.pi, 0.0]),
            parent_node=self,
        )

        self.idx = 0
        self.angs = np.linspace(np.pi, 1.5 * np.pi, 100)

        self.timer = self.create_timer(1 / fps, self.on_update)

    def on_update(self):
        rgb, depth = self.camera.obs()

        ang = self.angs[self.idx]

        self.camera.update_view_from_pose(
            np.array([0, 0, 1.5 + np.sin(ang * 0.1)]),
            grip.math.rpy2quaternion([0.0, ang, 0.0]),
        )

        self.camera.draw_frustum(
            0.0,
        )

        self.idx = (self.idx + 1) % len(self.angs)


def main(args=None):
    rclpy.init(args=args)

    ros_camera_example = ROSCameraExample()

    rclpy.spin(ros_camera_example)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    ros_camera_example.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
