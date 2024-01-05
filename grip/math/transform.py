import pybullet as p
import numpy as np
from typing import Union, List, Tuple


def invert_transformation_matrix(transform_matrix: np.ndarray):
    """
    Inverts a 4x4 transformation matrix without using matrix inverse.
    Args: T (np.ndarray: shape-(4x4), a valid transformation matrix)

    """
    inverse_rot = transform_matrix[:3, :3].T
    inverse_pos = -inverse_rot @ transform_matrix[:3, 3]

    inverse = np.eye(4)
    inverse[:3, :3] = inverse_rot
    inverse[:3, 3] = inverse_pos

    return inverse


def invert_transform(
    position: Union[np.ndarray, List[float]], quaternion: Union[np.ndarray, List[float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the inverse of the given transform

    Args:
        position (np.ndarray): array-like shape-(3,) representing a position/translation
        quaternion (np.ndarray): array-like shape-(4,) unit quaternion
    Returns:
        (tuple[np.ndarray,np.ndarray]): position and orientation representing the inverse of the given transform
    """
    pos, quat = p.invertTransform(position, quaternion)
    return np.asarray(pos), np.asarray(quat)


def multiply_transform(
    pA: Union[np.ndarray, List[float]],
    qA: Union[np.ndarray, List[float]],
    pB: Union[np.ndarray, List[float]],
    qB: Union[np.ndarray, List[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multiply two transforms given by their respective positions and orientation quaternions

    Args:
        pA (np.ndarray): array-like shape-(3,) representing a position/translation
        qA (np.ndarray): array-like shape-(4,) unit quaternion
        pB (np.ndarray): array-like shape-(3,) representing a position/translation
        qB (np.ndarray): array-like shape-(4,) unit quaternion
    Returns:
        (tuple[np.ndarray,np.ndarray]): position and orientation representing the multiplied transforms
    """
    pos, quat = p.multiplyTransforms(pA, qA, pB, qB)
    return np.asarray(pos), np.asarray(quat)


def transform_inv(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transform points using the inverse of the given transform

    Args:
        points (np.ndarray): array-like shape-(M,3) representing a list of 3-dimensional points
        transform (np.ndarray): array-like shape-(4,4) transformation matrix
    Returns:
        (np.ndarray): array-like shape-(M,3) representing a list of 3-dimensional transformed points
    """
    R = transform[:3, :3].T
    t = transform[:3, 3]

    local_points = np.dot(R, (points - t).T).T

    return local_points


def transform(points, transform):
    """
    Transform points given a rigid body transformation matrix

    Args:
        points (np.ndarray): array-like shape-(M,3) representing a list of 3-dimensional points
        transform (np.ndarray): array-like shape-(4,4) transformation matrix
    Returns:
        (np.ndarray): array-like shape-(M,3) representing a list of 3-dimensional transformed points
    """
    R = transform[:3, :3]
    t = transform[:3, 3]

    transformed_points = np.dot(R, points.T).T + t.reshape(1, 3)

    return transformed_points
