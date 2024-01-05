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

import cv2


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
            image_topic="/hand_camera/color/image_raw",
            image_info_topic="/hand_camera/color/camera_info/",
            depth_topic="/hand_camera/aligned_depth_to_color/image_raw",
            depth_info_topic="hand_camera/aligned_depth_to_color/camera_info",
            base_frame_id="hand_camera_link",
            sim=False,
        )

        self.timer = self.create_timer(1 / fps, self.on_update)

    def on_update(self):
        if not self.camera.is_ready:
            grip.io.log.info("Camera is not ready.")
            return

        rgb, depth = self.camera.obs()

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        depth_colormap = cv2.applyColorMap(
            # convert to mm
            cv2.convertScaleAbs(depth * 1e3, alpha=0.03),
            cv2.COLORMAP_JET,
        )

        images = np.hstack([bgr, depth_colormap])
        cv2.imshow("RGB-Depth", images)

        cv2.waitKey(1)


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
