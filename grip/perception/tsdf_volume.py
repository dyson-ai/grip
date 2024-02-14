import numpy as np
import open3d as o3d
import trimesh


class TSDFVolume(object):
    """Integration of multiple depth images using a TSDF."""

    def __init__(self, size, resolution, origin, with_colour=False):
        self.size = size
        self.resolution = resolution
        self.voxel_size = self.size / self.resolution
        self.origin = origin
        self.sdf_trunc = 4 * self.voxel_size
        self.with_colour = with_colour

        self.reset()

    def reset(self):
        if o3d.__version__ >= "0.13.0":
            o3d_integration = o3d.pipelines.integration
        else:
            o3d_integration = o3d.integration

        self._volume = o3d_integration.UniformTSDFVolume(
            length=self.size,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=(
                o3d_integration.TSDFVolumeColorType.RGB8
                if self.with_colour
                else o3d_integration.TSDFVolumeColorType.NoColor
            ),
            origin=self.origin[:3, 3],
        )

    def integrate(self, depth_img, intrinsic, extrinsic, colour_img=None):
        """
        Args:
            depth_img: The depth image.
            intrinsic: o3d.camera.PinholeCameraIntrinsic
            extrinsics: numpy array
        """

        colour_image = np.empty_like(depth_img) if colour_img is None else colour_img
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(colour_image),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False,
        )

        pose = np.linalg.inv(extrinsic)
        pose[:3, :3] = np.matmul(pose[:3, :3], self.origin[:3, :3])
        pose[:3, 3] -= self.origin[:3, 3]
        extrinsic = np.linalg.inv(pose)

        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def get_grid(self):
        # TODO(mbreyer) very slow (~35 ms / 50 ms of the whole pipeline)
        shape = (1, self.resolution, self.resolution, self.resolution)
        tsdf_grid = np.zeros(shape, dtype=np.float32)
        voxels = self._volume.extract_voxel_grid().get_voxels()
        # vis_voxels = self._volume.extract_voxel_point_cloud()
        # o3d.visualization.draw_geometries([vis_voxels])
        for voxel in voxels:
            i, j, k = voxel.grid_index
            tsdf_grid[0, i, j, k] = voxel.color[0]
        return tsdf_grid

    def get_cloud(self):
        cloud = self._volume.extract_point_cloud()
        cloud = cloud.transform(self.origin)

        return cloud

    def get_mesh(self):
        mesh = self._volume.extract_triangle_mesh()

        mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=2000)
        mesh_smp.compute_vertex_normals(normalized=True)
        mesh_smp = mesh_smp.transform(self.origin)

        return mesh_smp

    def get_mesh_as_trimesh(self):
        mesh = self.get_mesh()

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.vertex_normals)

        # print(f"nvertices {vertices.shape} ntriangles {triangles.shape} nnormals {normals.shape}")
        tri_mesh = trimesh.Trimesh(vertices, triangles, vertex_normals=normals)

        return tri_mesh


class ScalableTSDFVolume(object):
    """Integration of multiple depth images using a TSDF."""

    def __init__(
        self, origin, depth_sampling_stride=4, resolution=9, with_colour=False
    ):
        self.origin = origin
        self.resolution = resolution
        scale = 2**self.resolution
        self.voxel_size = depth_sampling_stride / scale

        self.sdf_trunc = 0.04  # 4 * self.voxel_size

        self.with_colour = with_colour

    def reset(self):
        if o3d.__version__ >= "0.13.0":
            o3d_integration = o3d.pipelines.integration
        else:
            o3d_integration = o3d.integration

        self._volume = o3d_integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            volume_unit_resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=(
                o3d_integration.TSDFVolumeColorType.RGB8
                if self.with_colour
                else o3d_integration.TSDFVolumeColorType.NoColor
            ),
        )

    def integrate(self, depth_img, intrinsic, extrinsic, colour_img=None):
        """
        Args:
            depth_img: The depth image.
            intrinsic: o3d.camera.PinholeCameraIntrinsic
            extrinsics: numpy array
        """

        colour_image = np.empty_like(depth_img) if colour_img is None else colour_img
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(colour_image),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            depth_trunc=1.5,
            convert_rgb_to_intensity=False,
        )

        # pose = np.linalg.inv(extrinsic)
        # pose[:3, :3] = np.matmul(pose[:3, :3], self.origin[:3, :3])
        # pose[:3, 3] -= self.origin[:3, 3]
        # extrinsic = np.linalg.inv(pose)

        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def get_cloud(self):
        return self._volume.extract_point_cloud()

    def get_mesh(self):
        return self._volume.extract_triangle_mesh()

    def get_mesh_as_trimesh(self):
        mesh = self.get_mesh()

        mesh = mesh.merge_close_vertices(0.001)
        mesh = mesh.remove_duplicated_vertices()
        mesh = mesh.remove_degenerate_triangles()

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.vertex_normals)

        # print(f"nvertices {vertices.shape} ntriangles {triangles.shape} nnormals {normals.shape}")
        tri_mesh = trimesh.Trimesh(vertices, triangles, vertex_normals=normals)

        return tri_mesh
