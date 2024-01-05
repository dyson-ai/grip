"""
Serialization of sensor_msgs.PointCloud2 messages. Many of these are copied from sensor_msgs.point_cloud2.py in ros1.
"""

#! /usr/bin/python3
import ctypes
import math
import struct
import copy

import numpy as np
import open3d as o3d
from typing import List

# from nptyping import NDArray, Float32, Int32, Shape
from enum import Enum, unique

from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import PointCloud2, PointField


@unique
class PointCloudType(Enum):
    """
    Enum defining the type of point cloud, where type is determined by the data present.
    """

    XYZ = 0
    XYZ_RGB = 1
    XYZ_NORMAL = 2
    XYZ_NORMAL_RGB = 3
    XYZ_NORMAL_CURV_RGB = 4

    @classmethod
    def __missing__(cls, value):
        """
        If no value or invalid value is specified, defaults to XYZ
        """
        return cls.XYZ


_DATATYPES = {}
_DATATYPES[PointField.INT8] = ("b", 1)
_DATATYPES[PointField.UINT8] = ("B", 1)
_DATATYPES[PointField.INT16] = ("h", 2)
_DATATYPES[PointField.UINT16] = ("H", 2)
_DATATYPES[PointField.INT32] = ("i", 4)
_DATATYPES[PointField.UINT32] = ("I", 4)
_DATATYPES[PointField.FLOAT32] = ("f", 4)
_DATATYPES[PointField.FLOAT64] = ("d", 8)


FIELDS_ORDER = ["x", "y", "z", "normal_x", "normal_y", "normal_z", "rgb", "curvature"]


def read_points(cloud, field_names=None, skip_nans=True, uvs=None):
    """
    Read points from a L{sensor_msgs.PointCloud2} message.
    @param cloud: The point cloud to read from.
    @type  cloud: L{se_nsor_msgs.PointCloud2}
    @param field_names: The names of fields to read. If None, read all fields. [default: None]
    @type  field_names: iterable
    @param skip_nans: If True, then don't return any point with a NaN value.
    @type  skip_nans: bool [default: False]
    @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
    @type  uvs: iterable
    @return: Generator which yields a list of values for each point.
    @rtype:  generator
    """
    assert isinstance(cloud, PointCloud2), "cloud is not a sensor_msgs.msg.PointCloud2"
    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = (
        cloud.width,
        cloud.height,
        cloud.point_step,
        cloud.row_step,
        cloud.data,
        math.isnan,
    )
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step


def create_cloud(header, fields, points):
    """
    Create a L{sensor_msgs.msg.PointCloud2} message.
    @param header: The point cloud header.
    @type  header: L{std_msgs.msg.Header}
    @param fields: The point cloud fields.
    @type  fields: iterable of L{sensor_msgs.msg.PointField}
    @param points: The point cloud points.
    @type  points: list of iterables, i.e. one iterable for each point, with the
                   elements of each iterable being the values of the fields for
                   that point (in the same order as the fields parameter)
    @return: The point cloud.
    @rtype:  L{sensor_msgs.msg.PointCloud2}
    """

    cloud_struct = struct.Struct(_get_struct_fmt(False, fields))

    buff = ctypes.create_string_buffer(cloud_struct.size * len(points))

    point_step, pack_into = cloud_struct.size, cloud_struct.pack_into
    offset = 0
    i = 0
    has_colour = False
    while i < len(fields):
        if fields[i].name == "rgb":
            has_colour = True
            break
        i += 1

    for p in points:
        values = [v for v in p]

        if has_colour:
            values[i] = int(values[i])

        pack_into(buff, offset, *values)
        offset += point_step

    return PointCloud2(
        header=header,
        height=1,
        width=len(points),
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=cloud_struct.size,
        row_step=cloud_struct.size * len(points),
        data=buff.raw,
    )


def create_cloud_xyz32(header, points):
    """
    Create a L{sensor_msgs.msg.PointCloud2} message with 3 float32 fields (x, y, z).
    @param header: The point cloud header.
    @type  header: L{std_msgs.msg.Header}
    @param points: The point cloud points.
    @type  points: iterable
    @return: The point cloud.
    @rtype:  L{sensor_msgs.msg.PointCloud2}
    """
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    return create_cloud(header, fields, points)


def create_cloud_xyz32color(header, points):
    """
    Create a L{sensor_msgs.msg.PointCloud2} message with 3 float32 fields (x, y, z).
    @param header: The point cloud header.
    @type  header: L{std_msgs.msg.Header}
    @param points: The point cloud points.
    @type  points: iterable
    @return: The point cloud.
    @rtype:  L{sensor_msgs.msg.PointCloud2}
    """

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
    ]
    return create_cloud(header, fields, points)


def create_cloud_point_normal_color_curvature(header, points):
    """
    Create a L{sensor_msgs.msg.PointCloud2} message with 3 float32 fields (x, y, z).
    @param header: The point cloud header.
    @type  header: L{std_msgs.msg.Header}
    @param points: The point cloud points.
    @type  points: iterable
    @return: The point cloud.
    @rtype:  L{sensor_msgs.msg.PointCloud2}
    """

    # ['x', 'y', 'z', 'normal_x', 'normal_y', 'normal_z', 'rgb', 'curvature', 'principal_curvature_x', 'principal_curvature_y', 'principal_curvature_z', 'pc1', 'pc2']
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="normal_x", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="normal_y", offset=16, datatype=PointField.FLOAT32, count=1),
        PointField(name="normal_z", offset=20, datatype=PointField.FLOAT32, count=1),
        PointField(name="curvature", offset=24, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=28, datatype=PointField.UINT32, count=1),
        #   PointField('curvature', 28, PointField.FLOAT32, 1),
        #   PointField('principal_curvature_x', 32, PointField.FLOAT32, 1),
        #   PointField('principal_curvature_y', 36, PointField.FLOAT32, 1),
        #   PointField('principal_curvature_z', 40, PointField.FLOAT32, 1),
        #   PointField('pc1', 42, PointField.FLOAT32, 1),
        #   PointField('pc2', 46, PointField.FLOAT32, 1)
    ]
    return create_cloud(header, fields, points)


def create_cloud_point_color_normal(header, points):
    """
    Create a L{sensor_msgs.msg.PointCloud2} message with 3 float32 fields (x, y, z).
    @param header: The point cloud header.
    @type  header: L{std_msgs.msg.Header}
    @param points: The point cloud points.
    @type  points: iterable
    @return: The point cloud.
    @rtype:  L{sensor_msgs.msg.PointCloud2}
    """

    # ['x', 'y', 'z', 'normal_x', 'normal_y', 'normal_z', 'rgb']
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="normal_x", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="normal_y", offset=16, datatype=PointField.FLOAT32, count=1),
        PointField(name="normal_z", offset=20, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=24, datatype=PointField.UINT32, count=1),
    ]
    return create_cloud(header, fields, points)


def create_cloud_point_normal(header, points):
    """
    Create a L{sensor_msgs.msg.PointCloud2} message with 6 float32 fields (x, y, z, normal_x, normal_y, normal_z).
    @param header: The point cloud header.
    @type  header: L{std_msgs.msg.Header}
    @param points: The point cloud points.
    @type  points: iterable
    @return: The point cloud.
    @rtype:  L{sensor_msgs.msg.PointCloud2}
    """

    # ['x', 'y', 'z', 'normal_x', 'normal_y', 'normal_z']
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="normal_x", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="normal_y", offset=16, datatype=PointField.FLOAT32, count=1),
        PointField(name="normal_z", offset=20, datatype=PointField.FLOAT32, count=1),
    ]
    return create_cloud(header, fields, points)


def _get_struct_fmt(
    is_bigendian: bool, fields: List[PointField], field_names: List[str] = None
) -> str:
    """
    Creates struct format string used to create a new python struct object.

    :param is_bigendian: True if native byte order is big-endian
    :param fields: List of pointfields (e.g. x,y,z, rgb etc.)
    :param field_names: Optional list of field names
    """

    fmt = ">" if is_bigendian else "<"
    offset = 0
    for field in (
        f
        for f in sorted(fields, key=lambda f: f.offset)
        if field_names is None or f.name in field_names
    ):
        if offset < field.offset:
            fmt += "x" * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print("Skipping unknown PointField datatype [%d]" % field.datatype)
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            if field.name == "rgb":
                # For some reason the colour field is having a wrong data type. It should be unsigned int 32 bits. This is a hard-coded fix
                datatype_fmt, datatype_length = _DATATYPES[PointField.UINT32]
            fmt += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt


def msg_to_array(msg: PointCloud2) -> np.ndarray:
    """
    Converts data in PointCloud2 msg to numpy array

    :param msg: PointCloud2 msg
    :returns: Numpy array of point data
    """

    all_fields = [field.name for field in msg.fields]
    print("all fields: ", all_fields)
    point_data_generator = read_points(msg, field_names=all_fields, skip_nans=True)

    size = msg.width * msg.height
    dims = len(msg.fields)

    points = np.zeros((size, dims))
    colors = np.zeros((size, 1), dtype=np.int32)  # Seems to be unused??

    i = 0
    fields_order = [i for i in FIELDS_ORDER if i in all_fields]

    for p in point_data_generator:
        p_dict = dict(zip(fields_order, p))

        p_row = np.array([p_dict[k] for k in fields_order])

        if "rgb" in all_fields:
            colors[i] = p_dict["rgb"]
        points[i, :] = p_row

        i += 1

    return points


def o3d_point_cloud_to_msg(
    o3d_cloud: o3d.geometry.PointCloud, frame_id: str = "world"
) -> PointCloud2:
    """
    Converts open3D pointcloud to PointCloud2 message.

    :param o3d_cloud: Open3D point cloud
    :param frame_id: Cloud frame ID to be used in msg header.
    :returns: PointCloud2 message
    """
    empty_header = Header()

    points = np.asarray(o3d_cloud.points)
    size = len(points)

    if o3d_cloud.has_normals():
        normals = np.asarray(o3d_cloud.normals)
    if o3d_cloud.has_colors():
        colors = np.asarray(o3d_cloud.colors)

        def RGBTo32bitInt(r, g, b):
            return np.uint32(int("%02x%02x%02x" % (int(r), int(g), int(b)), 16))

        uint32_colors = np.zeros((size, 1), dtype=np.uint32)

        for i in range(size):
            r = colors[i, 0] * 255.0
            g = colors[i, 1] * 255.0
            b = colors[i, 2] * 255.0

            uint32_colors[i] = RGBTo32bitInt(r, g, b)

    msg = None

    if o3d_cloud.has_normals() and o3d_cloud.has_colors():
        data = np.hstack([points, normals, uint32_colors])
        msg = create_cloud_point_color_normal(empty_header, data)
    elif o3d_cloud.has_normals():
        data = np.hstack([points, normals])
        msg = create_cloud_point_normal(empty_header, data)
    elif o3d_cloud.has_colors():
        data = np.hstack([points, uint32_colors])
        msg = create_cloud_xyz32color(empty_header, data)
    else:
        data = points
        msg = create_cloud_xyz32(empty_header, data)

    msg.header.frame_id = frame_id
    return msg


def array_to_msg(
    data: List[np.ndarray],
    frame_id="world",
    pc_type: PointCloudType = PointCloudType.XYZ,
) -> PointCloud2:
    """
    Converts numpy array of xyz points to PointCloud2 msg.

    :param data: List of iterables, where each iterable corresponds to a point with the data in the same order as the fields (e.g. one point could be [x,y,z,n_x, n_y, n_z, rgb]).
    :param frame_id: Frame ID of point cloud
    :param pc_type: Type defining data present in cloud
    :returns: PointCloud2 message
    """

    empty_header = Header()
    msg = None
    if pc_type == PointCloudType.XYZ:
        msg = create_cloud_xyz32(empty_header, data)
    elif pc_type == PointCloudType.XYZ_NORMAL:
        msg = create_cloud_point_normal(empty_header, data)
    elif pc_type == PointCloudType.XYZ_RGB:
        msg = create_cloud_xyz32color(empty_header, data)
    elif pc_type == PointCloudType.XYZ_NORMAL_RGB:
        msg = create_cloud_point_color_normal(empty_header, data)
    elif pc_type == PointCloudType.XYZ_NORMAL_CURV_RGB:
        msg = create_cloud_point_normal_color_curvature(empty_header, data)

    msg.header.frame_id = frame_id
    return msg


def msg_to_o3d_point_cloud(
    msg: PointCloud2,
    indices: List[int] = None,
    normalise: bool = True,
    vis: bool = False,
) -> o3d.geometry.PointCloud:
    """
    Converts PointCloud2 to open3D point cloud and samples selected indices if specified. Optionally visualises the cloud.
    If colour is present in data, it assumes that RGB is represented by a single 32-bit int.

    NB This function replaces pydvil.utils.pointcloud2_to_o3d() with the same signature.

    :param msg: PointCloud2 message
    :param indices: Point cloud indices to retain. If None, use all.
    :param normalise: If True (default), normals are normalised.
    :param vis: If True, visualise the o3d point cloud.
    :returns: Open3D point cloud.
    """

    data = msg_to_array(msg)

    def toRGBReal(uint_rgb):
        blue = uint_rgb & 255
        green = (uint_rgb >> 8) & 255
        red = (uint_rgb >> 16) & 255
        return np.array([float(red) / 255.0, float(green) / 255.0, float(blue) / 255.0])

    if data.shape[0] > 0:
        pcd = o3d.geometry.PointCloud()
        xyz_points = data[:, :3]
        nan_rows = np.isnan(xyz_points).any(axis=1)
        pcd.points = o3d.utility.Vector3dVector(xyz_points)

        # if msg.fields.normal_x and msg.fields.normal_y and msg.fields.normal_z
        msg_fields = [msg.fields[i].name for i in range(len(msg.fields))]

        if (
            "normal_x" in msg_fields
            and "normal_y" in msg_fields
            and "normal_z" in msg_fields
        ):
            pcd.normals = o3d.utility.Vector3dVector(data[:, 3:6])
            if "rgb" in msg_fields:
                colors = np.zeros((data.shape[0], 3))
                for i in range(data.shape[0]):
                    color = int(data[i, 6])
                    colors[i] = toRGBReal(color)
                pcd.colors = o3d.utility.Vector3dVector(colors)
            if normalise:
                pcd = pcd.normalize_normals()
        elif "rgb" in msg_fields:
            colors = np.zeros((data.shape[0], 3))
            for i in range(data.shape[0]):
                color = int(data[i, 3])
                colors[i] = toRGBReal(color)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        if indices is not None:
            # remove indices corresponding to nan_rows
            indices = np.setdiff1d(np.asarray(indices), np.where(nan_rows))
            pcd_final = pcd.select_by_index(indices)
        else:
            # NaNs will need to be removed before calling segmentation functions
            pcd_final = pcd

        if vis:
            o3d.visualization.draw_geometries([pcd_final.remove_non_finite_points()])

        return pcd_final
    else:
        return None


def point_to_msg(p: List[int] | np.ndarray) -> Point:
    """
    Converts iterable point (xyz) to Point msg.

    :param p: 3x1 iterable representing x,y,z point.
    """

    return Point(x=p[0], y=p[1], z=p[2])


def quaternion_to_msg(q: List[int] | np.ndarray) -> Quaternion:
    """
    Converts iterable quaternion to Quaternion msg.

    :param q: 3x 1 iterable representing quaternion
    """
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


def position_quaternion_to_msg(
    p: List[int] | np.ndarray, q: List[int] | np.ndarray
) -> Pose:
    """
    Creates a Pose msg from a given xyz location 'p' and quaternion 'q'.

    :param p: 3x1 Iterable representing x,y,z point
    :param q: 3x1 Iterable representing quaternion
    :returns: Pose message
    """
    return Pose(position=point_to_msg(p), orientation=quaternion_to_msg(q))


# Bloc to upright a pointcloud.
def find_rotation_to_xy_plane(plane: np.ndarray) -> np.ndarray:
    """
    Computes the rotation matrix from a given plane (ie, its coefficients) to the XY plane (Normal along Z)

    Args:
        plane: vector4 plane coefficents (a, b, c, d)

    Returns:
        np.ndarray[3,3] rotation matrix to transform input plane to XY plane
    """
    a, b, c, _ = plane
    cos_phi = c / math.sqrt(a * a + b * b + c * c)
    sin_phi = math.sqrt((a * a + b * b) / (a * a + b * b + c * c))
    u1 = b / math.sqrt(a * a + b * b)
    u2 = -a / math.sqrt(a * a + b * b)
    out = np.array(
        [
            [cos_phi + u1 * u1 * (1 - cos_phi), u1 * u2 * (1 - cos_phi), u2 * sin_phi],
            [u1 * u2 * (1 - cos_phi), cos_phi + u2 * u2 * (1 - cos_phi), -u1 * sin_phi],
            [-u2 * sin_phi, u1 * sin_phi, cos_phi],
        ]
    )
    return out


def upright_cloud(scene_cloud, plane_coeffs, flip):
    """
    Transforms a cloud with a plance so that it is oriented so that the plane normal points to Z.
    Args:
        scene_cloud: Pointcloud with a planar surface to transform
        plane_coeffs: Coefficients of the plane on the scene
        flip: flag to invert orientation of the pointcloud
    Returns:
        scene_cloud oriented so that the plane normal is aligned with Z and origin on the centre of the plane.
    """
    scene_upright = copy.deepcopy(scene_cloud)
    scene_upright.translate((0, 0, plane_coeffs[3] / plane_coeffs[2]))
    R = find_rotation_to_xy_plane(plane_coeffs)
    scene_upright.rotate(R, center=(0, 0, 0))
    if flip:
        flip = scene_upright.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        scene_upright.rotate(flip, center=(0, 0, 0))
    return scene_upright
