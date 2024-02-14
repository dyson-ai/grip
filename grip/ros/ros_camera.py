from __future__ import annotations
from grip.sensors.camera import RGBDCamera
from ..robot import BulletRobot
from ..io import log
import open3d as o3d
from ..math import position_quaternion2matrix
import numpy as np
from typing import Tuple, Optional
from threading import RLock

import rclpy

from .tf_tools import TFListener, TFPublisher
from .io import xacro_to_urdf_path
import rclpy.node as rclpy_node
import sensor_msgs.msg as sensor_msgs
import cv_bridge
from .ros2_future import wait_for_message
import message_filters
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


class ROSCamera(RGBDCamera):
    """
    ROSCamera sensor, inherits from RGBDCamera, exposing extra ROS utilities.
    Represents a simulated or a real RGBD camera.

    Args:
        sim (bool, optional): whether or not this camera should be a simulated camera or if it should subscribe to real camera topics. Defaults to True.
        name (str, optional): camera name. Defaults to "CameraSensor"
        position (numpy.ndarray): position of the optical centre of this camera
        orientation (numpy.ndarray): orientation quaternion of the optical centre of this camera
        anchor_robot (grip.robot.BulletRobot, optional): robot that this camera may be anchored/attached to. Defaults to None.
        anchor_link (str, optional): required if anchor_robot is not None. Link name of the anchor_robot that this camera should be attached to.
            Typically a camera link has been already defined in the URDF of the robot, anchor_link can be chosen to be this pre-defined URDF link (camera optical frame)
        cid (int, optional): physics engine unique identifier (e.g. world.id)
        proj_from_intrinsics (bool, optional): if False sets up intrinsics, sets up projection matrix using self.compute_projection_matrix
            otherwise sets projection matrix using self.build_projection_matrix. Defaults to True.
        base_frame_id (str, optional): name of the base frame id. Default: "base_link".
        broadcast_tf_on (bool, optional): Applicable if sim=True. If broadcast_tf_on is also set to True, then whenever self.obs() is called it then broadcasts the transform tree of this camera. Link names that do not contain this camera name as a substring are ignored (not published). This flag applicable in simulation only.
        parent_node (rclpy_node.Node, optional): parent ROS2 node to which this ROSCamera may belong to. If not passed, this ROSCamera will be its own node. Defaults to None meaning this camera will have its own ROS2 node.

    """

    def __init__(self, **kwargs):
        self.sim = kwargs.get("sim", True)
        self.name = kwargs.get("name", "camera")
        self.broadcast_tf_on = kwargs.get("broadcast_tf_on", False)
        self.base_frame_id = kwargs.get("base_frame_id", "base_link")
        self.node = kwargs.get("parent_node", rclpy_node.Node(f"{self.name}_node"))

        self.image_topic = kwargs.get(
            "image_topic", f"{self.name}/color/undistorted/image_rect"
        )
        self.image_info_topic = kwargs.get(
            "image_info_topic", f"{self.name}/color/undistorted/camera_info"
        )

        self.depth_topic = kwargs.get(
            "depth_topic", f"{self.name}/depth_registered/undistorted/image_rect"
        )

        self.depth_info_topic = kwargs.get(
            "depth_info_topic", f"{self.name}/depth_registered/undistorted/camera_info"
        )

        kwargs.setdefault("width", 424)
        kwargs.setdefault("height", 280)
        kwargs.setdefault("camera_far", 4.0)
        kwargs.setdefault("proj_from_intrinsics", False)

        self.camera_info_msg = sensor_msgs.CameraInfo()

        self._timer_cb_group = MutuallyExclusiveCallbackGroup()

        self.depth_msg = None
        self.image_msg = None

        super().__init__(**kwargs)

        self.is_ready = True
        self.image_info_ready = True

        if self.sim:
            self._setup_sim(**kwargs)
        else:
            self._setup_real()

        self.rgb_image = None
        self.depth_image = None
        self.seg_image = None

        self.image_bridge = cv_bridge.CvBridge()

        self.mutex = RLock()

        self._cb_lock = kwargs.get("lock", RLock())
        self._timer_func = None
        self._async_on = False

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """
        Shuts down this camera
        """

        if not hasattr(self, "_cb_lock"):
            return

        if self.sim:
            with self._cb_lock:
                self.stop_timer_publisher()
                for pub in [
                    self.pub_image,
                    self.pub_image_info,
                    self.pub_depth,
                    self.pub_depth_info,
                ]:
                    try:
                        pub.destroy()
                    except Exception as e:
                        log.warning(
                            f"Could not kill publisher {pub.topic_name}. Error: {e}"
                        )

    def destroy(self) -> None:
        """Destroy underlying ROS handles"""

        self.shutdown()
        self.node.destroy_node()
        self.anchor_robot = None

    def _init_camera_info(self) -> bool:
        """
        Initialises camera info for the real camera (intrinsics), if self.sim = False.

        Returns:
            bool: whether or not received camera info successfully
        """

        success = False
        try:
            success, self.camera_info_msg = wait_for_message(
                sensor_msgs.CameraInfo,
                self.node,
                self.depth_info_topic,
                time_to_wait=5,
            )

            if success:
                log.info("Camera info received successfully!")
            else:
                log.warning("Camera info has not been received.")
        except Exception as e:
            log.warning(
                f"Waiting for {self.name} info timeout. Is the camera on? Initialising camera info with empty camera info. Error: {e}"
            )

        return success

    def _setup_sim(self, **kwargs) -> None:
        self.pub_image = self.node.create_publisher(
            sensor_msgs.Image, self.image_topic, 10
        )

        self.pub_image_info = self.node.create_publisher(
            sensor_msgs.CameraInfo, self.image_info_topic, 10
        )

        self.pub_depth = self.node.create_publisher(
            sensor_msgs.Image, self.depth_topic, 10
        )

        self.pub_depth_info = self.node.create_publisher(
            sensor_msgs.CameraInfo, self.depth_info_topic, 10
        )

        self.image_msg = None
        self.depth_msg = None
        self.camera_info_msg = None

        self.image_msg = sensor_msgs.Image()
        self.depth_msg = sensor_msgs.Image()

        self.image_msg.width = self.width
        self.image_msg.height = self.height
        self.image_msg.encoding = kwargs.get("colour_encoding", "rgb8")
        self.image_msg.is_bigendian = 0
        self.image_msg.step = self.image_msg.width * 3 * 1

        self.depth_msg.width = self.image_msg.width
        self.depth_msg.height = self.image_msg.height
        self.depth_msg.encoding = kwargs.get("depth_encoding", "32FC1")
        self.depth_msg.is_bigendian = self.image_msg.is_bigendian
        self.depth_msg.step = self.depth_msg.width * 1 * 4  # Width*Channels*nBytes

        camera_info = self.get_camera_info()

        self.camera_info_msg = sensor_msgs.CameraInfo(**camera_info)

        self.image_msg.header.frame_id = self.frame_id
        self.depth_msg.header.frame_id = self.frame_id
        self.camera_info_msg.header.frame_id = self.frame_id

        self.tf_publisher = TFPublisher(self.node)

    def _setup_real(self) -> None:
        """
        Sets up real camera. Registers to depth and rgb topics.
        """

        # from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
        # qos_profile = QoSProfile(
        #     reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT#RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
        #     history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
        #     depth=1
        # )

        self.is_ready = False
        self.image_info_ready = self._init_camera_info()

        self.tf_listener = TFListener(self.node)

        log.debug(f"Creating filter for topic: {self.depth_topic}")
        self.depth_sub = message_filters.Subscriber(
            self.node,
            sensor_msgs.Image,
            self.depth_topic,
        )

        log.debug(f"Creating filter for topic: {self.image_topic}")
        self.rgb_sub = message_filters.Subscriber(
            self.node,
            sensor_msgs.Image,
            self.image_topic,
        )

        self.time_synchroniser = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.rgb_sub], queue_size=1, slop=1e9
        )

        self.time_synchroniser.registerCallback(self._subscriber_callback)

    # overriden
    def obs(
        self, ret_ext: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Retrieves an observation from this camera.

        Args:
            ret_ext (bool, optional): whether of not it should return the extrinsics parameters of the camera as well.
        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, Optional[numpy.ndarray]]: three ndarrays rgb image shape-(width, height), depth image shape-(width, height),
                the third array (optional return) being the extrinsics of the camera with shape-(4,4) if ret_ext=True.
        """
        if self.sim:
            return self._obs_sim(ret_ext=ret_ext)

        return self._obs_real(ret_ext=ret_ext)

    def get_cloud_obs(self, full_ret=False) -> Tuple[
        o3d.geometry.PointCloud,
        Optional[o3d.geometry.RGBDImage],
        Optional[o3d.camera.PinholeCameraIntrinsic],
        Optional[np.ndarray],
    ]:
        """
        Gets open3d cloud observation.

        Args:
            full_ret (bool, optional): If full_ret=False, returns only point cloud observartion. If full_ret=True returns pcd, rgbd image, intrinsics, extrinsics.

        Returns:
            Tuple[o3d.geometry.PointCloud, Optional[o3d.geometry.RGBDImage], Optional[o3d.camera.PinholeCameraIntrinsic], Optional[np.ndarray]]: Point cloud observation and optionally rgbd_image, camera intrinsics and extrinsics if full_ret=True
        """

        colour, depth, extrinsics = self.obs(ret_ext=True)

        if colour is None or depth is None or extrinsics is None:
            return None

        pose = np.linalg.inv(extrinsics)

        intrinsics = self.get_o3d_intrinsics()

        log.debug(f"Intrinsics: {intrinsics.intrinsic_matrix}")

        rgb_image = self.pcd_parser.as_o3d_image(colour)
        depth_image = self.pcd_parser.as_o3d_image(depth)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_image,
            depth_image,
            depth_scale=1,
            depth_trunc=self.pcd_parser.depth_truncate,
            convert_rgb_to_intensity=False,
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            intrinsic=intrinsics,
            extrinsic=extrinsics,
            project_valid_depth_only=False,
        )

        log.debug(f"Point cloud size: {len(pcd.points)}")
        pcd.estimate_normals()
        pcd.orient_normals_towards_camera_location(camera_location=pose[:3, 3])

        ret = pcd

        if full_ret:
            ret = (pcd, rgbd_image, intrinsics, extrinsics)

        return ret

    def _obs_real(
        self, ret_ext: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Retrieves an observation from this (real) camera.

        Args:
            ret_ext (bool, optional): whether of not it should return the extrinsics parameters of the camera as well.
        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, Optional[numpy.ndarray]]: three ndarrays rgb image shape-(width, height), depth image shape-(width, height),
                the third array (optional return) being the extrinsics of the camera with shape-(4,4) if ret_ext=True.
        """

        rgb, depth, extrinsics = None, None, None

        with self.mutex:
            if self.ready:
                rgb, depth = self.rgb_image, self.depth_image
                extrinsics = self.get_extrinsics(self.depth_msg.header.stamp)
            else:
                log.warning("Camera is not ready yet.")
                raise RuntimeWarning("Camera is not ready yet.")

            if ret_ext:
                return rgb, depth, extrinsics

            return rgb, depth

    def _obs_sim(
        self, ret_ext: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Retrieves an observation from this (simulated) camera.

        Args:
            ret_ext (bool, optional): whether of not it should return the extrinsics parameters of the camera as well.
        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, Optional[numpy.ndarray]]: three ndarrays rgb image shape-(width, height), depth image shape-(width, height),
                the third array (optional return) being the extrinsics of the camera with shape-(4,4) if ret_ext=True.
        """
        self.rgb_image, self.depth_image, self.seg_image = super().obs()

        self.image_msg.data = self.image_bridge.cv2_to_imgmsg(
            self.rgb_image, encoding=self.image_msg.encoding
        ).data

        self.depth_msg.data = self.image_bridge.cv2_to_imgmsg(
            self.depth_image, encoding=self.depth_msg.encoding
        ).data

        stamp = self.node.get_clock().now().to_msg()
        self.image_msg.header.stamp = stamp
        self.depth_msg.header.stamp = stamp
        self.camera_info_msg.header.stamp = stamp

        self.pub_image.publish(self.image_msg)
        self.pub_image_info.publish(self.camera_info_msg)

        self.pub_depth.publish(self.depth_msg)
        self.pub_depth_info.publish(self.camera_info_msg)

        if self.sim and self.broadcast_tf_on and self.anchor_robot:
            self._broadcast_anchor_tfs(stamp)

        self.tf_publisher.broadcast_frame(
            self.pose, self.base_frame_id, self.frame_id, stamp
        )

        if ret_ext:
            extrinsics = self.get_extrinsics(self.depth_msg.header.stamp)
            return self.rgb_image, self.depth_image, extrinsics

        return self.rgb_image, self.depth_image

    def _broadcast_anchor_tfs(self, stamp: rclpy.time.Time) -> None:
        """
        Broadcasts tf tree of anchored robot (kinematic chain)

        Args:
            stamp: time stamp.
        """

        for link_name in self.anchor_robot.get_link_names():
            if self.name in link_name:
                pose = self.anchor_robot.get_link_pose_by_name(link_name)
                self.tf_publisher.broadcast_frame(
                    pose, link_name, self.base_frame_id, stamp
                )

    def _publisher_callback(self) -> None:
        with self._cb_lock:
            try:
                self.obs()
            except Exception as e:
                log.warning(
                    f"Shutting down {self.name}. Failed to query camera observation. Perhaps disconnected from world. Message {e}"
                )

    def setup_timer_publisher(self, fps: int = 10) -> None:
        """
        Sets up and starts a timer callback. Image topics will be published in the background for asynchronous usage.
        """
        if self._async_on:
            return

        self._timer_func = self.node.create_timer(
            1.0 / fps, self._publisher_callback, callback_group=self._timer_cb_group
        )

        self._async_on = True

    def stop_timer_publisher(self) -> None:
        """
        Stops timer publisher. Disables asyncronous publishing.
        """

        with self._cb_lock:
            if not self._async_on:
                return

            self._timer_func.destroy()

        self._timer_func = None
        self._async_on = False

    def _get_intrinsics_real(self) -> Tuple[float, float, float, float, float]:
        info_msg = self.camera_info_msg

        fx = info_msg.k[0]
        fy = info_msg.k[4]

        cx = info_msg.k[2]
        cy = info_msg.k[5]

        width = info_msg.width
        height = info_msg.height

        return width, height, fx, fy, cx, cy

    # override
    def get_intrinsics(self) -> np.ndarray:
        """
        Gets the raw intrinsics parameters of this camera.

        Returns:
            Tuple[float, float, float, float, float, float]: intrinsics parameters width, height, fx, fy, cx, cy
        """

        if self.sim:
            return super().get_intrinsics()

        return self._get_intrinsics_real()

    def get_intrinsic_matrix(self) -> np.ndarray:
        """
        Gets the camera intrinsic parameters of this camera as a matrix.

        Returns:
            (numpy.ndarray): shape-(3,3) matrix of intrinsic parameters of this camera
        """

        _, _, fx, fy, cx, cy = self.get_intrinsics()

        intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        return intrinsic_matrix

    def get_segmentation(self) -> np.ndarray:
        """
        Gets the ground-truth segmentation image/mask.


        Returns:
            (numpy.ndarray): shape-(width, height) segmentation mask image (ground-truth segmentation).
        """

        _, _, self.seg_image = super().obs()

        return self.seg_image

    def _get_real_extrinsics(self, time: rclpy.time.Time = None) -> np.ndarray:
        position, orientation, time = self.tf_listener.get_transform(
            self.camera_info_msg.header.frame_id, self.base_frame_id, time
        )

        if position is None or orientation is None:
            camera_extrinsics = None
        else:
            camera_extrinsics = np.array(
                position_quaternion2matrix(position, orientation)
            )

        return camera_extrinsics

    # override
    def get_extrinsics(self, time: rclpy.time.Time = rclpy.time.Time()) -> np.ndarray:
        """
        Gets the exintrics matrix of this camera (the inverse of its pose).

        Args:
            time (rclpy.time.Time, optional): gets extrinsics at specified time. Defaults to rclpy.time.Time(), meaning most current measurement.
        Returns:
            (numpy.ndarray): camera extrinsics shape-(4,4)
        """

        ret = None
        if self.sim:
            extrinsics = super().get_extrinsics()

            if self.anchor_robot:
                base_frame = self.anchor_robot.get_link_pose_by_name(self.base_frame_id)
                base_frame = position_quaternion2matrix(*base_frame)

                ret = extrinsics @ base_frame
            else:
                ret = extrinsics
        else:
            ret = self._get_real_extrinsics(time)

        return ret

    def get_transform(self) -> np.ndarray:
        """
        Gets the transformation matrix representing the pose of this camera.
        Returns the same as self.pose, but in matrix form.

        Returns:
            (numpy.ndarray): camera pose transform shape-(4,4)
        """
        return position_quaternion2matrix(*self.pose)

    def _get_real_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns real camera pose.

        Returns:
            (Tuple[numpy.ndarray, numpy.ndarray]): camera pose, position and quaternion
        """
        position, orientation, _ = self.tf_listener.get_transform(
            self.base_frame_id, self.camera_info_msg.header.frame_id, rclpy.time.Time()
        )
        return position, orientation

    # override
    @property
    def pose(self):
        """
        (Tuple[numpy.ndarray, numpy.ndarray]): camera pose, position and quaternion
        """
        if self.sim:
            return self.position, self.orientation
        else:
            return self._get_real_pose()

    def _subscriber_callback(self, depth_msg, rgb_msg) -> None:
        with self.mutex:
            self.image_msg = rgb_msg
            self.depth_msg = depth_msg

            self.rgb_image = None
            try:
                self.rgb_image = self.image_bridge.imgmsg_to_cv2(rgb_msg, "rgb8")

                match self.depth_msg.encoding:
                    case "16UC1":
                        self.depth_image = np.frombuffer(
                            depth_msg.data, dtype=np.uint16
                        ).reshape(depth_msg.height, depth_msg.width)
                        self.depth_image = np.multiply(
                            self.depth_image.astype("float32"), 0.001
                        )
                    case _:
                        self.depth_image = self.image_bridge.imgmsg_to_cv2(
                            depth_msg, desired_encoding="32FC1"
                        )

                # this camera is ready if messages have been received successfully including image info
                self.is_ready = self.image_info_ready

                # import cv2
                # cv2.imshow("o3d image", self.rgb_image)

                # cv2.imshow("o3d depth", self.depth_image)

                # print("Depth: ", np.median(self.depth_image))
                # cv2.waitKey(1)

                log.debug("Received camera images")

            except cv_bridge.CvBridgeError as e:
                log.error(f"Error on converting image to rgb8 format {e}")

    def ready(self) -> bool:
        """
        Indicates whether or not this camera is ready.

        Returns:
            (bool): whether or not this camera is ready.
        """
        return self.is_ready

    @classmethod
    def make_d415(
        cls,
        camera_name: str,
        position: np.ndarray,
        orientation: np.ndarray,
        sim: bool,
        lock: RLock = None,
        auto_anchor: bool = False,
        phys_id: int = 0,
    ) -> ROSCamera:
        """

        Creates a grip.sensors.ROSCamera using the intel d415 realsense urdf description.

        Args:
            camera_name: the name of the camera.
            position: the position of the camera optical centre with respect to world frame.
            orientation: the orientation of the camera optical with respect to world frame.
            sim: whether or not this is a simulated camera or not. If sim=False it will try to connect to a real camera publishing in standard topics.
            lock (optional): camera creates an internal lock to deal with critical memory regions. If global thread coordination is needed a global lock can be passed so this camera can use it instead.
            auto_anchor (optional): whether or not this camera should create its own rigid body chain in pybullet and anchor itself to it or not.
            phys_id (optional): physics id where camera is going be inserted to.

        Returns:
            (ROSCamera): the created camera.



        """
        camera_optical_link_frame = None
        camera_chain = None

        if auto_anchor:
            camera_optical_link_frame = f"{camera_name}_color_optical_frame"

            urdf_file = xacro_to_urdf_path(
                "sensor_descriptions",
                "intel_ds415_camera_instance.urdf.xacro",
                "intel_ds415_camera.urdf",
                xacro_args=f"name:='{camera_name}'",
                pkg_subfolder="intel_ds415",
            )

            camera_chain = BulletRobot(
                urdf_file=urdf_file,
                use_fixed_base=True,
                has_kinematics=False,
                phys_id=phys_id,
            )

            camera_chain.disable_collision()
            camera_chain.set_link_pose(
                camera_optical_link_frame,
                position,
                orientation,
            )

        camera = cls(
            cid=phys_id,
            name=camera_name,
            position=position,
            orientation=orientation,
            anchor_robot=camera_chain,
            anchor_link=camera_optical_link_frame,
            sim=sim,
            broadcast_tf_on=True,
            lock=lock,
        )

        return camera
