import numpy as np
import cv2
from PIL import Image


def visualize_depth(depth_in, title, min_depth=None, max_depth=None, wait_time=0):
    if min_depth is None:
        min_depth = np.nanmin(depth_in)
    if max_depth is None:
        max_depth = np.nanmax(depth_in)
    depth_vis = np.clip(depth_in, min_depth, max_depth)
    depth_vis = depth_vis / max_depth
    # depth_vis = (depth_vis - min_depth) / (max_depth - min_depth)
    depth_vis *= 255.0
    depth_vis = depth_vis.astype(np.uint8)
    cv2.imshow(title, depth_vis)
    k = cv2.waitKey(wait_time)
    if wait_time == 0:
        cv2.destroyWindow(title)
    return k


def resize_numpy_image(image, size, resize_type="antialias"):
    assert resize_type in [
        "nearest",
        "antialias",
        "bicubic",
        "bilinear",
    ], "Resize type invalid"
    assert image.dtype == "uint8", "Input image array must be in np.uint8 type"
    image = image.astype(np.uint8)
    if resize_type == "nearest":  # for segmentation mask resize
        image = image.resize(size, Image.NEAREST)
    elif resize_type == "antialias":
        image = image.resize(size, Image.ANTIALIAS)
    elif resize_type == "bicubic":
        image = image.resize(size, Image.BICUBIC)
    elif resize_type == "bilinear":
        image = image.resize(size, Image.BILINEAR)
    image = np.array(image)
    return image


def pcd_to_image(obj_pcd, cam_intrinsic, cam_extrinsic=None):
    w = cam_intrinsic.width
    h = cam_intrinsic.height
    fx, fy = cam_intrinsic.get_focal_length()
    cx, cy = cam_intrinsic.get_principal_point()
    n_points = obj_pcd.shape[0]

    projected_image = np.zeros((h, w))
    projected_image[:] = np.NaN

    if n_points > 0:
        if cam_extrinsic is not None:
            cam_xyz = np.matmul(
                cam_extrinsic,
                np.expand_dims(np.c_[obj_pcd, np.ones(n_points)], axis=-1),
            )
            cam_xyz = cam_xyz.squeeze(-1)[:, :-1]
        else:
            cam_xyz = obj_pcd.copy()

        cam_xyz[:, 0] = np.round((cam_xyz[:, 0] * fx / cam_xyz[:, -1]) + cx)
        cam_xyz[:, 1] = np.round((cam_xyz[:, 1] * fy / cam_xyz[:, -1]) + cy)

        in_range_mask = np.logical_and(
            np.logical_and(
                np.logical_and(cam_xyz[:, 0] < w, cam_xyz[:, 1] < h), cam_xyz[:, 0] >= 0
            ),
            cam_xyz[:, 1] >= 0,
        )
        in_range_cam_xyz = cam_xyz[in_range_mask]
        projected_image[
            in_range_cam_xyz[:, 1].astype(int), in_range_cam_xyz[:, 0].astype(int)
        ] = in_range_cam_xyz[:, -1]

    return projected_image
