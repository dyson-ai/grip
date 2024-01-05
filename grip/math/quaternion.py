import numpy as np
from typing import Union, List
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from transforms3d.euler import quat2euler

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def position_from_matrix(matrix):
    """
    Returns the translation part of a 4x4 homegeneous transform matrix

    Args:
        matrix (np.ndarray): A shape-(4,4) matrix

    Returns:
        (np.ndarray): A shape-(3,) translation part of the matrix
    """
    return matrix[:3, 3]


def quaternion_from_matrix(matrix: np.ndarray, isprecise=True) -> np.ndarray:
    """Return quaternion from rotation matrix.

    returned quaternion format (JPL): [x, y, z, w]

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / np.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, 0.0, 0.0, 0.0],
                [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)

    # Converting to [x, y, z, w] (JPL format)
    q = np.array([q[1], q[2], q[3], q[0]])
    q /= np.linalg.norm(q)
    return q


def quaternion_from_matrix2(R: np.ndarray) -> np.ndarray:
    """
    Converts a rotation matrix to an orientation quaternion.

    Args:
        R (numpy.ndarray): A shape-(4, 4) or shape-(3, 3) array
    Returns:
        (numpy.ndarray): A shape-(4,) array representing a unity quaternion
    """
    tc = np.trace(R[:3, :3])

    q = np.zeros((3, 1))
    if tc > 0:
        s = 2 * np.sqrt(1 + tc)
        qw = 0.25 * s  # s = 4 * qw

        q = np.array(
            [
                (R[2][1] - R[1][2]) / s,
                (R[0][2] - R[2][0]) / s,
                (R[1][0] - R[0][1]) / s,
                qw,
            ]
        )

    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s

        q = np.array([qx, qy, qz, qw])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s

        q = np.array([qx, qy, qz, qw])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # // S = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

        q = np.array([qx, qy, qz, qw])

    return q


def quaternion2rotation_matrix(
    quaternion: Union[np.ndarray, List[float]]
) -> np.ndarray:
    """
    Returns a rotation matrix from a quaternion

    Args:
        quaternion (np.ndarray|list[float]): unit quaternion 1 quaternion: [x,y,z,w] (shape used in Bullet Physics) to rotation matrix
            see: http://www.mrelusive.com/publications/papers/SIMD-From-Quaternion-to-Matrix-and-Back.pdf

    Returns:
        (np.ndarray): A shape-(4,4) array representing a rotation matrix
    """

    if isinstance(quaternion, list):
        quaternion = np.array(quaternion, dtype=np.float64)

    n = np.dot(quaternion, quaternion)
    if n < 1e-12:
        return np.identity(3)
    q = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
        ]
    )


def position_quaternion2matrix(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Converts a position and orientation quaternion to a 4x4 transformation matrix

    Args:
        p (numpy.ndarray): A shape-(3,) array
        q (numpy.ndarray): A shape-(4,) array unit quaternion
    Returns:
        (numpy.ndarray): A shape-(4,4) array representing a rigid body transform
    """
    transform = np.eye(4)

    R = quaternion2rotation_matrix(q)

    transform[0:3, 0:3] = R
    transform[:3, 3] = p

    return transform


def matrix2position_quaternion(matrix: np.ndarray) -> np.ndarray:
    """
    Converts a rotation matrix to an orientation quaternion.

    Args:
        matrix (numpy.ndarray): A shape-(4, 4) array
    Returns:
        (numpy.ndarray): A shape-(4,) array representing a unity quaternion
    """

    p = position_from_matrix(matrix)
    q = quaternion_from_matrix(matrix)

    return p, q


def make_slerp(key_times: np.ndarray, key_quaternions: List[np.ndarray]) -> Slerp:
    """
    Creates a spherical linear interpolation object

    Args:
        key_times (numpy.ndarray): A shape-(N,) array
        key_quaternions (list[np.ndarray]): A list of orientation quaternions
    Returns:
        (scipy.spatial.transform.Slerp): A slerp object
    """

    rotations = R.from_quat(key_quaternions)

    slerp = Slerp(key_times, rotations)

    return slerp


def quaternion_mult(
    q1: Union[np.ndarray, List[float]], q2: Union[np.ndarray, List[float]]
) -> np.ndarray:
    """
    Quaternion multiplication convention: same order as matrix multiplication
    so Ra*Rb is same as qa*qb
    See "Why and How to Avoid the Flipped Quaternion Multiplication"
    paper: https://arxiv.org/abs/1801.07478
    comment: same convention as ROS

    Args:
        q1 (np.ndarray|list[float]): unit quaternion 1
        q2 (np.ndarray|list[float]): unit quaternion 2
    Returns:
        (np.ndarray): quaternion multiplication q1*q2
    """

    if isinstance(q1, list):
        q1 = np.array(q1)
    if isinstance(q2, list):
        q2 = np.array(q2)

    s1 = q1[3]
    s2 = q2[3]
    v1 = q1[0:3]
    v2 = q2[0:3]

    dot = np.multiply(s1, s2) - np.dot(v1, v2)
    mult1 = np.multiply(s1, v2)
    mult2 = np.multiply(s2, v1)
    result = np.hstack([mult1 + mult2 + np.cross(v1, v2), dot])

    return result


def quat_multiply(
    q1: Union[np.ndarray, List[float]], q0: Union[np.ndarray, List[float]]
) -> np.ndarray:
    """Return multiplication of two quaternions.

    Args:
        q1 (np.ndarray|list[float]): array-like shape-(4,) unit quaternion 1
        q2 (np.ndarray|list[float]): array-like shape-(4,) unit quaternion 2
    Returns:
        (np.ndarray): shape-(4,) quaternion multiplication q1*q2
    Examples:
        >>> q = quat_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
        >>> numpy.allclose(q, [-44, -14, 48, 28])
        True
    """
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    return np.array(
        (
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ),
        dtype=np.float64,
    )


def quaternion_diff(
    q1: Union[np.ndarray, List[float]], q2: Union[np.ndarray, List[float]]
) -> np.ndarray:
    """Return the difference between two quaternions.
    The error can be used for cartesian feedback controllers and has been originally proposed by:
    Yuan88: https://www.cs.cmu.edu/~cga/dynopt/readings/Yuan88-quatfeedback.pdf

    Args:
        q1 (np.ndarray|list[float]): array-like shape-(4,) unit quaternion 1
        q2 (np.ndarray|list[float]): array-like shape-(4,) unit quaternion 2
    Returns:
        (np.ndarray): shape-(4,) q1*quaternion_conj(q2)
    """
    return quaternion_mult(q1, quaternion_conj(q2))


def quaternion_error(
    q1: Union[np.ndarray, List[float]], q2: Union[np.ndarray, List[float]]
) -> float:
    """Return the quaternion error, which is the magnitude of the quaternion difference

    Args:
        q1 (np.ndarray|list[float]): array-like shape-(4,) unit quaternion 1
        q2 (np.ndarray|list[float]): array-like shape-(4,) unit quaternion 2
    Returns:
        (float): quaternion error

    """
    qdiff = quaternion_diff(q1, q2)

    return np.linalg.norm(qdiff[:3])


def compute_log(q: np.ndarray) -> np.ndarray:
    """Return the logarithmic map of the quaternion

    Args:
        q (np.ndarray): array-like shape-(4,) unit quaternion 1
    Returns:
        (np.ndarray): shape-(3,) array representing the log map of the unit quaternion

    """
    v = q[:3]
    norm_q = np.linalg.norm(q)
    norm_v = np.linalg.norm(v)

    tolerance = 1e-6

    if norm_v < tolerance:
        return np.zeros(3)
    else:
        return 2 * np.arccos(q[3] / norm_q) * v / norm_v


def exp_map(omg: np.ndarray) -> np.ndarray:
    """Return the exponential map of 3d vector

    Args:
        q (np.ndarray): array-like shape-(3,) vector
    Returns:
        (np.ndarray): a unit quaternion shape-(4,) array representing the exp map of the vector

    """

    angle = np.linalg.norm(omg)
    if angle == 0:
        return np.asarray([0, 0, 0, 1])
    return axis_angle2quaternion(omg / angle, angle)


def compute_omg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Return the angular difference between two quaternions

    Args:
        q1 (np.ndarray): array-like shape-(4,) unit quaternion 1
        q2 (np.ndarray): array-like shape-(4,) unit quaternion 2
    Returns:
        (np.ndarray): A shape-(3,) array representing 3D angular difference (radians)

    """
    dot = np.dot(q1, q2)

    if dot < 0:
        q2 = -q2

    return 2.0 * compute_log(quaternion_mult(q1, quaternion_conj(q2)))


def rpy2quaternion(rpy: Union[np.ndarray, List[float]]) -> np.ndarray:
    """
    Return a quaternion from roll pitch yaw vector (radians)

    Args:
        rpy (np.ndarray): array-like shape-(3,) [roll, pitch, yaw] (radians)
    Returns:
        (np.ndarray): a unit quaternion shape-(4,)

    """
    return euler2quaternion(rpy[0], rpy[1], np.fmod(rpy[2], 2 * np.pi), axes="sxyz")


def quaternion2rpy(q: np.ndarray) -> np.ndarray:
    """
    Return a roll pitch yaw vector (radians) from a quaternion

    Args:
        q (np.ndarray): array-like shape-(4,) unit quaternion
    Returns:
        (np.ndarray): An array shape-(3,) containing the roll pitch and yaw angles (radians)

    """
    return quat2euler(q, axes="sxyz")


def euler2quaternion(ai, aj, ak, axes="sxyz"):
    """Return quaternion from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    Examples:
        >>> q = _euler2quaternion(1, 2, 3, 'ryxz')
        >>> numpy.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])
        True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = np.cos(ai)
    si = np.sin(ai)
    cj = np.cos(aj)
    sj = np.sin(aj)
    ck = np.cos(ak)
    sk = np.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    quaternion = np.empty((4,), dtype=np.float64)
    if repetition:
        quaternion[i] = cj * (cs + sc)
        quaternion[j] = sj * (cc + ss)
        quaternion[k] = sj * (cs - sc)
        quaternion[3] = cj * (cc - ss)
    else:
        quaternion[i] = cj * sc - sj * cs
        quaternion[j] = cj * ss + sj * cc
        quaternion[k] = cj * cs - sj * sc
        quaternion[3] = cj * cc + sj * ss
    if parity:
        quaternion[j] *= -1

    return quaternion


def quaternion_conj(q: np.ndarray) -> np.ndarray:
    """
    Return the conjugate of the given quaternion

    Args:
        q (np.ndarray): array-like shape-(4,) unit quaternion
    Returns:
        (np.ndarray): quaternion conjugate

    """
    return np.hstack([-q[0:3], q[3]])


def axis_angle2quaternion(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """
    Return a quaternion from an axis angle representation

    Args:
        axis (np.ndarray): array-like shape-(3,) unit vector
        angle (float): angle in radians
    Returns:
        (np.ndarray): array shape-(4,) unit quaternion

    """
    return np.hstack([np.array(axis) * np.sin(angle * 0.5), np.cos(angle * 0.5)])
