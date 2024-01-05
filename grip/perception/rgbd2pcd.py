#!/usr/bin/env python3

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R

from ..robot.gui import visualizePcd


class RGBD2PCDParser(object):
    """
    This is a class that extract point cloud and infer other required information from
    rgbd information and

    :param int width: image width
    :param int height: image height
    :param float fx: focal length in x
    :param float fy: focal length in y
    :param float cx: x coordinate for camera principle point
    :param float cy: y coordinate for camera principle point
    :param float camera_near: near plane distance
    :param float camera_far: far plane distance
    :param float depth_scale: scale for current depth value wrt to 1mm
    """

    def __init__(
        self, width, height, fx, fy, cx, cy, camera_near, camera_far, depth_scale
    ):
        self.camera_position = [0, 0, 0]
        self.near = camera_near
        self.far = camera_far
        self.depth_scale = depth_scale
        self.depth_truncate = self.near * 100 + 1
        self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy
        )

        self.camera_extrinsic = np.eye(4)

    def extract_mask(self, seg_image, obj_id):
        """
        Extract a mask on target object for RGBD images

        :param nd.ndarray seg_image: input segmentation image from camera image
                                     rendered at the same time with rgb image
        :param [int] obj_id: list of object for target object id in simulation

        :return: A mask as a (width, height) array of boolean value
        :rtype: np.ndarray
        """
        if obj_id == -1:
            mask = np.ones(seg_image.shape).astype(np.uint8)
        else:
            mask = np.zeros(seg_image.shape).astype(np.uint8)
            for idx in obj_id:
                mask += (seg_image == idx).astype(np.uint8)
        return mask

    def segment_rgb(self, rgb_image, seg_image, obj_id):
        """
        Mask the rgb image to filter it with rgb for only target object remained

        :param nd.ndarray rgb_image: rendered rgb image
        :param nd.ndarray seg_image: segmentation mask/groundtruth for rendered rgb image
        :param [int] obj_id: list of target object id in simulation

        :return: A masked rgb image with only target object remained
        :rtype: np.ndarray
        """
        rgb_mask = self.extract_mask(seg_image, obj_id)
        rgb_mask = np.expand_dims(rgb_mask, 2)
        rgb_mask = np.repeat(rgb_mask, repeats=3, axis=2)
        rgb_image *= rgb_mask
        rgb_image = o3d.geometry.Image(rgb_image)
        return rgb_image

    def as_o3d_image(self, rgb_image):
        """
        Converts a numpy array image to an open3d Image

        :param nd.ndarray rgb_image: rendered rgb image

        :return: An Open3D rgb image
        :rtype: np.ndarray
        """

        rgb_image = o3d.geometry.Image(rgb_image)
        return rgb_image

    def segment_depth(self, depth_image, seg_image, obj_id):
        """
        Mask the depth image to filter it with only depth for target object remained

        :param nd.ndarray rgb_image: input rgb image from camera image
        :param nd.ndarray seg_image: input segmentation image from camera image
                                     rendered at the same time with rgb image
        :param [int] obj_id: list of target object id in simulation

        :return: A masked rgb image with only target object remained
        :rtype: np.ndarray
        """
        depth_mask = self.extract_mask(seg_image, obj_id)
        depth_image *= depth_mask
        depth_image = o3d.geometry.Image(depth_image)
        return depth_image

    def crop_z(self, pcd, z_crop):
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        colours = np.asarray(pcd.colors)

        pcd_out = o3d.geometry.PointCloud()

        mask = points[:, 2] > z_crop

        points = points[mask, :]
        pcd_out.points = o3d.utility.Vector3dVector(points[:, :3])

        if pcd.has_normals():
            normals = normals[mask, :]
            pcd_out.normals = o3d.utility.Vector3dVector(normals[:, :3])

        if pcd.has_colors():
            colours = colours[mask, :]
            pcd_out.colors = o3d.utility.Vector3dVector(colours[:, :3])

        return pcd_out

    def update_extrinsic(self, view_matrix, eye):
        """
        Update camera extrinsic matrix according to view matrix and eye position

        :param [float] view_matrix:  a list of 16 floats in the format of pybullet view matrix
        :param nd.ndarray eye: 3 element numpy array (x, y, z) as the camera focus position

        :return: Camera position (x, y, z) and orientation in quaternion (x, y, z, w)
        :rtype: np.ndarray
        """
        gl_view_matrix = np.array(view_matrix).reshape(4, 4).T
        rot = R.from_matrix(np.linalg.inv(gl_view_matrix[:3, :3]))
        gl_view_matrix[1, :] *= -1
        gl_view_matrix[2, :] *= -1
        rot = R.from_matrix(np.linalg.inv(gl_view_matrix[:3, :3]))
        quat = rot.as_quat()

        camera_pose = np.eye(4)
        camera_pose[:3, :3] = rot.as_matrix()
        camera_pose[:3, 3] = eye

        self.camera_extrinsic = np.linalg.inv(camera_pose)

        return eye, quat

    def init_volume(self, length=4.0, n_voxel=512, sdf_trunc=0.04):
        """
        Initialize a ScalableTSDFVolume class for volume integration later

        :param float length: length for the TSDF cubic per dimension
        :param int n_voxel: number of voxels per dimension
        :param float sdf_trunc: truncation value for signed distance function (SDF)

        :return: A initialized ScalableTSDFVolume with specified resolution and truncation value
        :rtype: open3d.integration.ScalableTSDFVolume
        """
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=length / n_voxel,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        return volume

    def object_pcd_from_rgbd(
        self, raw_rgb, raw_depth, raw_seg, object_id, vis=False, **kwargs
    ):
        """
        Extract the pointcloud for target object(s) from the rendered rgb-d images and
        segmentation mask

        :param np.ndarray raw_rgb: rendered rgb image
        :param np.ndarray raw_depth: rendered depth image
        :param np.ndarray raw_seg: rendered segmentation groundtruth
        :param [int] object_id: list of target object id from simulation
        :param bool vis: visualization for the extracted pointcloud (for debugging)

        :return: extracted colored pointcloud for target object(s)
        :rtype: open3d.geometry.PointCloud
        """

        self.camera_extrinsic = kwargs.get("extrinsics", np.eye(4))

        if raw_seg is not None:
            rgb_image = self.segment_rgb(raw_rgb, raw_seg, object_id)
            depth_image = self.segment_depth(raw_depth, raw_seg, object_id)
        else:
            rgb_image = self.as_o3d_image(raw_rgb)
            depth_image = self.as_o3d_image(raw_depth)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_image,
            depth_image,
            depth_scale=self.depth_scale,
            depth_trunc=self.depth_truncate,
            convert_rgb_to_intensity=False,
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic=self.camera_intrinsic, extrinsic=self.camera_extrinsic
        )

        if vis:
            visualizePcd(pcd, ratio=0.01, duration=10)

            # import matplotlib.pyplot as plt
            # plt.subplot(1, 2, 1)
            # plt.title('rgb image')
            # plt.imshow(rgbd_image.color)
            # plt.subplot(1, 2, 2)
            # plt.title('depth image')
            # plt.imshow(rgbd_image.depth)
            # plt.show()

            # o3d.visualization.draw_geometries([pcd])

        return pcd

    def object_pcd_from_depth(self, raw_depth, raw_seg, object_id, vis=False, **kwargs):
        """
        Extract the pointcloud for target object(s) from the rendered depth image and
        segmentation mask

        :param np.ndarray raw_depth: rendered depth image
        :param np.ndarray raw_seg: rendered segmentation groundtruth
        :param [int] object_id: list of target object id from simulation
        :param bool vis: visualization for the extracted pointcloud (for debugging)

        :return: extracted colored pointcloud for target object(s)
        :rtype: open3d.geometry.PointCloud
        """

        self.camera_extrinsic = kwargs.get("extrinsics", np.eye(4))
        if raw_seg is not None:
            depth_image = self.segment_depth(raw_depth, raw_seg, object_id)
        else:
            depth_image = self.as_o3d_image(raw_depth)

        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image,
            intrinsic=self.camera_intrinsic,
            extrinsic=self.camera_extrinsic,
            depth_scale=self.depth_scale,
            depth_trunc=self.depth_truncate,
        )

        if vis:
            visualizePcd(pcd, ratio=0.01, duration=10)
            # o3d.visualization.draw_geometries([pcd])

        return pcd

    def compute_curvature(self, pcd, radius=0.5):
        """
        A slow way to compute the curvature of input pointcloud from
        https://github.com/intel-isl/Open3D/issues/2471

        :param open3d.geometry.PointCloud pcd: input pointcloud
        :param float radius: radius of local region

        :return: list of float value which captures curvature value at each
                 points in the pointcloud
        :rtype: [float]
        """
        points = np.asarray(pcd.points)
        tree = KDTree(points)
        curvature = [0] * points.shape[0]
        for index, point in enumerate(points):
            indices = tree.query_ball_point(point, radius)
            # local covariance
            M = np.array([points[i] for i in indices]).T
            M = np.cov(M)
            # eigen decomposition
            V, E = np.linalg.eig(M)
            # h3 < h2 < h1
            h1, h2, h3 = V
            curvature[index] = h3 / (h1 + h2 + h3)
        return curvature

    def pcd_to_xyzrgba(self, pcd):
        """
        Convert pointcloud from open3d.geometry.PointCloud format into np.ndarray.
        The converted array has the shape as (N, 7) and contain x,y,z, rgba information

        Args:
            pcd (open3d.geometry.PointCloud): input pointcloud

        Returns:
            np.ndarray: (N, 7) array with x, y, z and rgba information of the input pointcloud
        """

        xyz = np.array(pcd.points, dtype=np.float32)
        rgb = np.array(pcd.colors, dtype=np.float32)
        a = np.ones((rgb.shape[0], 1), dtype=rgb.dtype)
        xyzrgba = np.concatenate([xyz, rgb, a], 1)
        return xyzrgba

    def pcd_to_xyzrgbn(self, pcd):
        """
        Convert pointcloud from open3d.geometry.PointCloud format into np.ndarray.
        The converted array has the shape as (N, 7) and contain x,y,z, rgb and normal information

        Args:
            pcd (open3d.geometry.PointCloud): input pointcloud

        Returns:
            np.ndarray: (N, 7) array with x, y, z, rgb, normal information of the input pointcloud
        """
        xyz = np.array(pcd.points, dtype=np.float32)
        rgb = np.array(pcd.colors, dtype=np.float32)
        if not pcd.has_normals():
            pcd.estimate_normals()
        normals = np.array(pcd.normals, dtype=np.float32)
        xyzrgbn = np.concatenate([xyz, rgb, normals], 1)
        return xyzrgbn

    def extract_from_pcd(self, pcd, normal=True, curvature=False):
        """
        Extract target field information (xyz, rgb, normal, curvature) or infer if required from input pointcloud according to specification

        :param open3d.geometry.PointCloud pcd: input pointcloud

        :return: (N, 6/7/8) array with x, y, z, rgb, (optional)normal, (optional) curvature
                 information of the input pointcloud
        :rtype: np.ndarray
        """
        # TODO: make xyz, rgb optional as well?
        xyz = np.array(pcd.points, dtype=np.float32)
        rgb = np.array(pcd.colors, dtype=np.float32)
        if normal:
            if not pcd.has_normals():
                pcd.estimate_normals()
            normals = np.array(pcd.normals, dtype=np.float32)
        else:
            normals = np.zeros(xyz.shape)
        if curvature:
            curvatures = np.array(self.compute_curvature(pcd), dtype=np.float32)
            curvatures = curvatures.reshape(-1, 1)
        else:
            curvatures = np.zeros((xyz.shape[0], 1))
        data = np.concatenate([xyz, rgb], 1)
        if normal:
            data = np.concatenate([data, normals], 1)
        if curvature:
            data = np.concatenate([data, curvatures], 1)
        return data

    # TODO add downsampling script
    def stitch_pcd(self, xyz, rgb, vis=False):
        """
        Stitch pointclouds extracted from multiple views into a single pointcloud.

        :param [np.ndarray] xyz, list of xyz values for pointclouds from multiple views
        :param [np.ndarray] rgb, list of rgb values for pointclouds from multiple views
        :param bool vis: visualize the stitched pointcloud

        :return: The stitched single pointcloud
        :rtype: open3d.geometry.PointCloud
        """
        pcd = o3d.geometry.PointCloud()
        xyz = np.vstack(xyz)
        rgb = np.vstack(rgb)
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        if vis:
            visualizePcd(pcd, ratio=0.001, duration=10)
            # o3d.visualization.draw_geometries([pcd])
        return pcd
