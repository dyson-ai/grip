import pybullet as p
import numpy as np
from typing import Tuple, Optional
import open3d as o3d

from .sensor import Sensor
from ..perception import RGBD2PCDParser
from ..robot.gui import addDebugFrame, draw_frustum
from ..math import position_quaternion2matrix, matrix2position_quaternion
from ..environments import LightingRandomiser
import math

VISION_META = {
    "view_dist": 1.6,
    "view_pitch": -128.8,
    "view_yaw": -176.6,
    "view_target_pos": [0.14, 0.24, 0.37],
    "width": 512,  # 640,
    "height": 512,  # 480,
    "fx": 256,
    "fy": 256,
    "cx": 256,
    "cy": 256,
    "depth_scale": 1,
    "camera_vfov": 1,
    "camera_hfov": 1.211,
    "camera_near": 0.03,
    "camera_far": 2.0,
    "downsample_height": 64,
    "downsample_width": 64,
}


class RGBDCamera(Sensor):
    """
    RGBDCamera sensor. Represents a simulated RGBD camera.

    Args:
        name (str, optional): camera name. Defaults to "CameraSensor"
        position (numpy.ndarray): position of the optical centre of this camera
        orientation (numpy.ndarray): orientation quaternion of the optical centre of this camera
        anchor_robot (grip.robot.BulletRobot, optional): robot that this camera may be anchored/attached to. Defaults to None.
        anchor_link (str, optional): required if anchor_robot is not None. Link name of the anchor_robot that this camera should be attached to.
            Typically a camera link has been already defined in the URDF of the robot, anchor_link can be chosen to be this pre-defined URDF link (camera optical frame)
        cid (int, optional): physics engine unique identifier (e.g. world.id)
        proj_from_intrinsics (bool, optional): if False sets up intrinsics, sets up projection matrix using self.compute_projection_matrix
            otherwise sets projection matrix using self.build_projection_matrix. Defaults to True.
        frame_id (str, optional): frame name id of this camera.

    """

    def __init__(self, **kwargs):
        self.cid = kwargs.get("cid", None)

        self.params = dict()

        # Default parameters
        self.params.update(VISION_META)

        # Kwargs overwrites default parameters
        self.params.update(kwargs)

        self.name = kwargs.get("name", "CameraSensor")

        self.width = self.params["width"]
        self.height = self.params["height"]
        self.ratio = self.width / self.height

        self.near = self.params["camera_near"]
        self.far = self.params["camera_far"]
        self.params["fx"] = self.params["fx"]
        self.params["fy"] = self.params["fy"]
        self.params["cx"] = self.params["cx"]
        self.params["cy"] = self.params["cy"]
        self.params["depth_scale"] = self.params["depth_scale"]
        self.params["camera_vfov"] = self.params["camera_hfov"] * (
            self.height / self.width
        )
        self.params["camera_aspect"] = float(self.width) / self.height

        self.proj_from_intrinsics = kwargs.get("proj_from_intrinsics", True)

        self.position = kwargs.get("position", [0.86925251, -0.00720329, 1.66544737])
        self.orientation = kwargs.get(
            "orientation", [-0.68534092, -0.70721588, 0.12470087, 0.12084375]
        )
        self.lookat_vector = kwargs.get("lookat_vector", [0, 0, 1])
        self.up_vector = kwargs.get("up_vector", [0, 0, 1])

        self.anchor_robot = kwargs.get("anchor_robot", None)
        self.anchor_link = kwargs.get("anchor_link", None)
        self._anchored_camera = False
        self.frame_id = kwargs.get("frame_id", f"{self.name}_link")
        if self.anchor_robot is not None:
            self.frame_id = self.anchor_link
            pos, ori = self.anchor_robot.get_link_pose_by_name(self.anchor_link)
            self.update_view_from_pose(pos, ori)
            self._anchored_camera = True

        self.set_camera_params()

        if not self.proj_from_intrinsics:
            self.setup_instrinsics()

        self.pcd_parser = RGBD2PCDParser(
            width=self.params["width"],
            height=self.params["height"],
            fx=self.params["fx"],
            fy=self.params["fy"],
            cx=self.params["cx"],
            cy=self.params["cy"],
            camera_near=self.params["camera_near"],
            camera_far=self.params["camera_far"],
            depth_scale=self.params["depth_scale"],
        )

        self._frustum_items = None

    def anchor_camera(
        self, anchor_robot: "grip.robot.BulletRobot", anchor_link: str
    ) -> None:
        """Anchors this camera to a given existing link in the anchor_robot.
        The camera will become rigidly attached to that link.
        The camera will look towards the z-axis of the desired link pose.

        Args:

            anchor_robot (grip.robot.BulletRobot): robot to anchor this camera to.
            anchor_link (str): existing link in anchor_robot to attach this camera to.
        """

        self.anchor_robot = anchor_robot
        self.anchor_link = anchor_link
        self.frame_id = self.anchor_link

        anchor_link_pose = self.anchor_robot.get_link_pose_by_name(self.anchor_link)
        self.update_view_from_pose(*anchor_link_pose)
        self._anchored_camera = True

    def setup_instrinsics(self) -> None:
        """
        Sets up this camera intrinsics parameters.
        It internally sets the intrinsic parameters of this camera from the return of self.get_intrsincs().
        """
        width, height, fx, fy, cx, cy = self.get_intrinsics()
        self.params["width"] = width
        self.params["height"] = height
        self.params["fx"] = fx
        self.params["fy"] = fy
        self.params["cx"] = cx
        self.params["cy"] = cy

    def obs(
        self,
        light_randomiser: Optional[LightingRandomiser] = None,
        shadow: Optional[bool] = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves an observation from this camera.

        Args:
            light_randomiser (Optional[LightingRandomiser]): optinal light randomiser. Warning: lighting randomisation uses simple tiny renderer which is CPU based and slow.
            Optional[bool]: whether or not to enable rendering of shadows in rgb image.
        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): three ndarrays with shape-(width, height), respectively, rgb, depth and segmentation mask image (ground-truth segmentation).
        """

        lighting_params = LightingRandomiser.DEFAULT_PARAMS
        renderer = p.ER_BULLET_HARDWARE_OPENGL

        if light_randomiser:
            lighting_params = light_randomiser.randomise()
            renderer = p.ER_TINY_RENDERER

        if self._anchored_camera:
            pos, ori = self.anchor_robot.get_link_pose_by_name(self.anchor_link)
            self.update_view_from_pose(pos, ori)

        _, _, rgb_img, z_buffer, seg_img = p.getCameraImage(
            width=self.params["width"],
            height=self.params["height"],
            viewMatrix=self.params["camera_view_matrix"],
            projectionMatrix=self.params["camera_project_matrix"],
            shadow=shadow,
            lightDirection=lighting_params.light_direction,
            lightColor=lighting_params.light_colour,
            lightAmbientCoeff=lighting_params.ambient_coef,
            lightDiffuseCoeff=lighting_params.diffuse_coef,
            lightSpecularCoeff=lighting_params.specular_coef,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=renderer,
            physicsClientId=self.cid,
        )

        if isinstance(rgb_img, tuple):
            rgb_img = np.array(rgb_img).reshape(
                (self.params["height"], self.params["width"], 4)
            )
            z_buffer = np.array(z_buffer)

        rgb_image = rgb_img[:, :, :3].astype(np.uint8)

        # opengl z_buffer to metric space (actual depth)
        depth_image = (
            1.0
            * self.params["camera_far"]
            * self.params["camera_near"]
            / (
                self.params["camera_far"]
                - (self.params["camera_far"] - self.params["camera_near"]) * z_buffer
            )
        )
        depth_image = depth_image.astype(np.float32)

        # seg_image = seg_img.astype(np.uint8)
        seg_image = seg_img

        return rgb_image, depth_image, seg_image

    def rgb2bgr(self, rgb_image: np.ndarray):
        """
        Converts rgb image to bgr image format (e.g. opencv format)

        Returns:
            (numpy.ndarray): bgr image
        """
        bgr_image = rgb_image[..., ::-1].copy()

        # import cv2
        # cv2.imshow("image", bgr_image)
        # cv2.waitKey()

        return bgr_image.astype(np.uint8)

    def pcd_from_rgbd(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        seg_image: np.ndarray = None,
        object_id: list = [],
        vis: bool = False,
    ) -> o3d.geometry.PointCloud:
        """
        Returns open3d point cloud from depth image.

        Args:
            rgb_image (numpy.ndarray): rgb image
            depth_image (numpy.ndarray): depth image
            seg_image (numpy.ndarray, optional): segmentation mask image. Defaults to None.
            object_id (list[int], optional): list of object ids to be kept in the depth cloud.
                This can be used to extract point clouds of selected object ids.
            vis (bool, optional): visualisation for debugging. Default is False.
        Returns:
            (open3d.geometry.PointCloud): point cloud with colours
        """
        return self.pcd_parser.object_pcd_from_rgbd(
            rgb_image,
            depth_image,
            seg_image,
            object_id=object_id,
            vis=vis,
            extrinsics=np.linalg.inv(position_quaternion2matrix(*self.pose)),
        )

    def pcd_from_depth(
        self,
        depth_image: np.ndarray,
        seg_image: np.ndarray = None,
        object_id: list = [],
        vis: bool = False,
    ) -> o3d.geometry.PointCloud:
        """
        Returns open3d point cloud from depth image.

        Args:
            depth_image (numpy.ndarray): depth image
            seg_image (numpy.ndarray, optional): segmentation mask image. Defaults to None.
            object_id (list[int], optional): list of object ids to be kept in the depth cloud.
                This can be used to extract point clouds of selected object ids.
            vis (bool, optional): visualisation for debugging. Default is False.
        Returns:
            (open3d.geometry.PointCloud): point cloud without colours
        """
        return self.pcd_parser.object_pcd_from_depth(
            depth_image,
            seg_image,
            object_id=object_id,
            vis=vis,
            extrinsics=self.get_extrinsics(),
        )

    def stitch_pcd(
        self, xyz: np.ndarray, rgb: np.ndarray, vis: bool = False
    ) -> o3d.geometry.PointCloud:
        """
        Stitch pointclouds extracted from multiple views into a single pointcloud.

        Args:
            xyz (list[numpy.ndarray]): list of xyz values for pointclouds from multiple views
            rgb (list[numpy.ndarray]): list of rgb values for pointclouds from multiple views
            vis (bool, optional): visualize the stitched pointcloud
        Returns:
            (open3d.geometry.PointCloud): The stitched single point cloud
        """
        return self.pcd_parser.stitch_pcd(xyz, rgb, vis=vis)

    # Needs testing
    def update_view_from_pose(
        self, position: np.ndarray, quaternion: np.ndarray
    ) -> None:
        """
        Sets pose of this camera.

        Args:
            position (numpy.ndarray): position of the optical centre of this camera
            orientation (numpy.ndarray): orientation quaternion of the optical centre of this camera
        """
        self.position = position
        self.orientation = quaternion

        opengl_view_mat = position_quaternion2matrix(self.position, self.orientation)
        opengl_view_mat[:3, 1] *= -1
        opengl_view_mat[:3, 2] *= -1

        opengl_view_mat = np.linalg.inv(opengl_view_mat).T

        self.params["camera_view_matrix"] = opengl_view_mat.flatten()

        self.lookat_vector = opengl_view_mat[:3, 2]
        self.up_vector = opengl_view_mat[:3, 1]

    def set_lookat(
        self,
        position: np.ndarray,
        lookat_vector: np.ndarray,
        up_vector: np.ndarray,
        debug: bool = False,
    ) -> None:
        """
        Sets pose of this camera based on position, lookat and up vectors.

        Args:
            position (numpy.ndarray): position of the optical centre of this camera
            lookat_vector (numpy.ndarray): direction of gaze, or look at vector.
            up_vector (numpy.ndarray): up vector of this camera
        """
        self.position = position
        self.lookat_vector = lookat_vector
        self.up_vector = up_vector

        self.update_view_from_lookat(debug=debug)

    def update_view_from_lookat(self, debug: bool = False) -> None:
        """
        Update this camera view parameters based on look at parameters.

        Args:
            debug (bool, optional): whether or not debug mode is on. Default is False.
        """
        self.params["camera_view_matrix"] = p.computeViewMatrix(
            self.position, self.lookat_vector, self.up_vector
        )

        self.position, self.orientation = self.pcd_parser.update_extrinsic(
            self.params["camera_view_matrix"], self.position
        )

        if debug:
            addDebugFrame(self.position, self.orientation, duration=100)

    def update_from_debug_view(self):
        self.position, self.orientation = self.get_debug_view_pose()

        self.update_view_from_pose(self.position, self.orientation)

        # addDebugFrame(self.position, self.orientation, duration=5, lw=5)

    def update_intrinsics(self) -> None:
        """
        Update this camera intrinsics parameters.
        """
        # Option (1) update the projection matrix with known hfov and vfov from camera setting
        #            vfov and fhov = 90 degree for pybullet default visualiser
        # Option(2) update projection with known fx, fy, cx, cy which usually come from camera info / calibration result

        if self.proj_from_intrinsics:
            self.params["camera_project_matrix"] = self.build_projection_matrix(
                self.params["width"],
                self.params["height"],
                self.params["fx"],
                self.params["fy"],
                self.params["cx"],
                self.params["cy"],
                self.params["camera_near"],
                self.params["camera_far"],
            )
        else:
            self.params["camera_project_matrix"] = self.compute_projection_matrix()

    def compute_projection_matrix(self) -> np.ndarray:
        """
        Returns projection matrix using camera frustum parameters.

        Returns:
            (numpy.ndarray): camera projection matrix (opengl)

        """

        return p.computeProjectionMatrix(
            left=-math.tan(float(self.params["camera_hfov"]) / 2.0)
            * self.params["camera_near"],
            right=math.tan(float(self.params["camera_hfov"]) / 2.0)
            * self.params["camera_near"],
            bottom=-math.tan(float(self.params["camera_vfov"]) / 2.0)
            * self.params["camera_near"],
            top=math.tan(float(self.params["camera_vfov"]) / 2.0)
            * self.params["camera_near"],
            nearVal=self.params["camera_near"],
            farVal=self.params["camera_far"],
        )

    def build_projection_matrix(
        self,
        width: float,
        height: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        near: float,
        far: float,
    ) -> np.ndarray:
        """
        Returns projection matrix using camera intrinsic parameters.

        Args:
            width (float): image width
            height (float): image height
            fx (float): camera horizontal focal length
            fy (float): camera vertical focal length
            cx (float): camera image centre of projection coordinate x
            cy (float): camera image centre of projection coordinate y
            near (float): camera near clipping plane
            far (float): camera far clipping plane.

        Returns:
            (numpy.ndarray): camera projection matrix (opengl)

        """
        perspective = np.array(
            [
                [fx, 0.0, -cx, 0.0],
                [0.0, -fy, -cy, 0.0],
                [0.0, 0.0, near + far, near * far],
                [0.0, 0.0, -1.0, 0.0],
            ]
        )

        ortho = self.gl_ortho(0.0, width, height, 0.0, near, far)
        proj_matrix = np.matmul(ortho, perspective)
        proj_matrix = proj_matrix.flatten(order="F")
        return proj_matrix

    def gl_ortho(
        self,
        left: float,
        right: float,
        bottom: float,
        top: float,
        near: float,
        far: float,
    ) -> np.ndarray:
        """
        OpenGL orthogonal projection matrix.

        Returns:
            (numpy.ndarray): opengl orthogonal projection matrix
        """
        ortho = np.diag(
            [2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0]
        )
        ortho[0, 3] = -(right + left) / (right - left)
        ortho[1, 3] = -(top + bottom) / (top - bottom)
        ortho[2, 3] = -(far + near) / (far - near)
        return ortho

    def set_projection_matrix(self, projection_matrix: np.ndarray) -> None:
        """
        Sets opengl camera projection matrix for this camera

        Args:
            projection_matrix (numpy.ndarray): opengl projection matrix
        """
        self.params["camera_project_matrix"] = projection_matrix

    def update_camera_params(self) -> None:
        """
        Updates camera params using the update_view_from_lookat
        """
        self.update_view_from_lookat()

    def set_camera_params(self) -> None:
        """
        Sets camera params using the update_view_from_pose
        """
        # Update intrinsics
        self.update_intrinsics()

        self.update_view_from_pose(self.position, self.orientation)

    @property
    def pose(self):
        """
        (Tuple[numpy.ndarray, numpy.ndarray]): camera pose, position and quaternion
        """
        return self.position, self.orientation

    def get_debug_view_matrix(self):
        """
        Gets opengl view matrix from debug visualiser

        Returns:
            (numpy.ndarray): opengl view matrix
        """

        camera_info = p.getDebugVisualizerCamera(physicsClientId=self.cid)
        opengl_view_mat = np.array(camera_info[2]).reshape(4, 4)

        return opengl_view_mat

    def get_debug_view_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets pose of the debug visualiser camera.

        Returns:
            (Tuple[numpy.ndarray, numpy.ndarray]): camera pose, position and quaternion.
        """
        opengl_view_mat = self.get_debug_view_matrix()

        camera_pose_gl = np.linalg.inv(opengl_view_mat).transpose()
        camera_pose_gl[:3, 1] *= -1
        camera_pose_gl[:3, 2] *= -1

        camera_pos, camera_quat = matrix2position_quaternion(camera_pose_gl)

        return camera_pos, camera_quat

    def get_debug_camera_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets params of the debug visualiser camera.

        Returns:
            (float, float, float, Tuple[float]): yaw, pitch, dist, target.
            yaw: yaw angle of the camera, in Cartesian local space coordinates
            pitch: pitch angle of the camera, in Cartesian local space coordinates
            dist: distance between the camera and the camera target
            target: target of the camera, in Cartesian world space coordinates

        """

        camera_info = p.getDebugVisualizerCamera(physicsClientId=self.cid)

        return camera_info[8:]

    def get_debug_view_projection_matrix(self) -> np.ndarray:
        """
        Gets projection matrix of the debug visualiser camera.

        Returns:
            (numpy.ndarray): opengl projection matrix
        """

        camera_info = p.getDebugVisualizerCamera(physicsClientId=self.cid)
        projection_mat = np.array(camera_info[3]).reshape(4, 4)

        return projection_mat

    def get_debug_view_image_size(self) -> Tuple[float, float]:
        """
        Gets the width and height parameters of the debug visualiser camera

        Returns:
            (Tuple[float, float]): width and height
        """
        camera_info = p.getDebugVisualizerCamera(physicsClientId=self.cid)

        return camera_info[0], camera_info[1]

    def get_intrinsics(self) -> Tuple[float, float, float, float, float, float]:
        """
        Gets the raw intrinsics parameters of this camera.

        Returns:
            Tuple[float, float, float, float, float, float]: intrinsics parameters width, height, fx, fy, cx, cy
        """

        width = self.params["width"]
        height = self.params["height"]

        proj_mat = np.array(self.params["camera_project_matrix"]).reshape([4, 4])
        fx = proj_mat[0, 0] / 2.0 * width
        fy = proj_mat[1, 1] / 2.0 * height
        cx = (1.0 - proj_mat[2, 0]) / 2.0 * width
        cy = (1.0 + proj_mat[2, 1]) / 2.0 * height

        return width, height, fx, fy, cx, cy

    def get_o3d_intrinsics(self) -> o3d.camera.PinholeCameraIntrinsic:
        """
        Gets the intrinsic parameters of this camera as an open3d PinholeCameraIntrinsic object.

        Returns:
            (open3d.camera.PinholeCameraIntrinsic): intrinsics parameters of this camera
        """
        width, height, fx, fy, cx, cy = self.get_intrinsics()

        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy
        )

        return camera_intrinsic

    def get_extrinsics(self) -> np.ndarray:
        """
        Gets the exintrics matrix of this camera (the inverse of its pose).

        Returns:
            (numpy.ndarray): camera extrinsics shape-(4,4)
        """

        return np.linalg.inv(position_quaternion2matrix(*self.pose))

    def get_camera_info(self) -> dict:
        """
        Constructs pinhole camera model matrix from opengl normalised device coordinates (NDC) perspective projection matrix.
        Equations for fx, fy, cx and cy are given by reversing the equations from the opengl NDC perspective projection matrix
        An example is given here: https://blog.noctua-software.com/opencv-opengl-projection-matrix.html
        More complete source opengl transforms: http://www.songho.ca/opengl/gl_transform.html

        Returns:
            (dict): a dict with format
            ``dict: {'width': float, 'height': float, 'distortion_model': str, 'D': numpy.ndarray, 'K': numpy.ndarray, 'R': numpy.ndarray, 'P': numpy.ndarray }``
        """

        width, height, fx, fy, cx, cy = self.get_intrinsics()

        d_matrix = list(np.zeros(5))

        view_matrix = list(np.zeros(9))
        view_matrix[0] = fx
        view_matrix[1] = 0.0
        view_matrix[2] = cx
        view_matrix[3] = 0.0
        view_matrix[4] = fy
        view_matrix[5] = cy
        view_matrix[6] = 0.0
        view_matrix[7] = 0.0
        view_matrix[8] = 1.0

        r_matrix = list(np.zeros(9))
        r_matrix[0] = 1.0
        r_matrix[1] = 0.0
        r_matrix[2] = 0.0
        r_matrix[3] = 0.0
        r_matrix[4] = 1.0
        r_matrix[5] = 0.0
        r_matrix[6] = 0.0
        r_matrix[7] = 0.0
        r_matrix[8] = 1.0

        projection_matrix = list(np.zeros(12))
        projection_matrix[0] = fx
        projection_matrix[1] = 0.0
        projection_matrix[2] = cx
        projection_matrix[3] = 0.0
        projection_matrix[4] = 0.0
        projection_matrix[5] = fy
        projection_matrix[6] = cy
        projection_matrix[7] = 0.0
        projection_matrix[8] = 0.0
        projection_matrix[9] = 0.0
        projection_matrix[10] = 1.0
        projection_matrix[11] = 0.0

        camera_info_dict = {
            "width": width,
            "height": height,
            "distortion_model": "plumb_bob",
            "d": d_matrix,
            "k": view_matrix,
            "r": r_matrix,
            "p": projection_matrix,
        }

        return camera_info_dict

    def draw_frustum(self, duration: float = 0.2) -> None:
        """
        Draws the frustom of this camera in the debug visualiser.

        Args:
            duration (float, optional): lifetime duration of this frustum. Defaults to 0.2s.
        """
        self._frustum_items = draw_frustum(
            self.position,
            self.orientation,
            self.params["camera_hfov"],
            self.params["camera_vfov"],
            self.params["camera_near"],
            self.params["camera_far"],
            self.ratio,
            duration=duration,
            cid=self.cid,
            item_ids=self._frustum_items,
        )
