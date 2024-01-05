import numpy as np


def normalised(x: np.ndarray) -> np.ndarray:
    """Return a normalised vector (unit vector)

    Args:
        x (np.ndarray): array-like shape-(N,) vector
    Returns:
        (np.ndarray): a unit/normalised vector shape-(N,)

    """
    norm = np.linalg.norm(x)

    if np.isclose(norm, 0.0):
        return x

    return x / norm


def as_transform(
    u: np.ndarray, v: np.ndarray, n: np.ndarray, p: np.ndarray = np.zeros(3)
) -> np.ndarray:
    """Return a 4x4 homogenous transformation matrix constructed from its column vector parts

    Args:
        u (np.ndarray): array-like shape-(3,) vector
        v (np.ndarray): array-like shape-(3,) vector
        n (np.ndarray): array-like shape-(3,) vector
        p (np.ndarray): array-like shape-(3,) vector
    Returns:
        (np.ndarray): An array shape-(4,4) representing a rigid transformation matrix

    """
    frame = np.vstack(
        [
            np.hstack(
                [u.reshape(3, 1), v.reshape(3, 1), n.reshape(3, 1), p.reshape(3, 1)]
            ),
            [0, 0, 0, 1],
        ]
    )

    return frame


def proj(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Project u over v

    Args:
        u (np.ndarray): array-like shape-(N,) vector
        v (np.ndarray): array-like shape-(N,) vector
    Returns:
        (np.ndarray): An array shape-(N,4) representing projection of u over v
    """

    dot_uv = np.dot(u, v)
    return dot_uv * v / np.dot(v, v)


def plane_vector_directors(plane_normal: np.ndarray) -> np.ndarray:
    """
    Computes the vector directors from a hyper-plane normal vector

    Args:
        plane_normal (np.ndarray): array-like shape-(N,) hyper-plane normal vector
    Returns:
        (np.ndarray): An array shape-(N,N) where each row is a vector director of the hyper-plane
    """

    plane_mat = np.vstack([plane_normal, np.zeros(3), np.zeros(3)])
    U, _, _ = np.linalg.svd(plane_mat, full_matrices=True)

    U[:, 0] = np.zeros(3)

    return U


def project_on_plane(
    plane_origin: np.ndarray, plane_normal: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """
    Projects points on plane

    Args:
        plane_origin (np.ndarray): array-like shape-(N,) hyper-plane origin
        plane_normal (np.ndarray): array-like shape-(N,) hyper-plane normal vector
        points (np.ndarray): array-like shape-(M,N) representing a list of N-dimensional points
    Returns:
        (np.ndarray): array-like shape-(M,N) representing a list of N-dimensional points projected on the plane
    """

    mat = plane_vector_directors(plane_normal)
    flat_points = plane_origin + np.dot(np.dot(points - plane_origin, mat), mat.T)

    return flat_points


def multivariate_normal(
    mean: np.ndarray,
    cov: np.ndarray,
    decomposition_type: str = "cholesky",
    check_valid=False,
    samples=1,
    eps=0.000000001,
):
    """Generates multivariate gaussian noise by sampling the cov matrix and centering
    the distribution about the given mean

    Args:
        mean (np.ndarray): Array of N means in the shape N*M.
        cov (np.ndarray): Array of N covariance matrices in the shape NxMxM, where the covariance matrices are of dimension MxM.
        decomposition_type (str, optional): Type of decomposition to be implemented on the covariance matrix
        "svd" and "cholesky" are implemented. Take note that cholesky is faster but is only applicable for
        covariance matrices that are positive semi-definite / non-degenerative distributions.
        i.e. sigma !=0 for all dimensions of the noise
        . Defaults to "cholesky".
        check_valid (bool, optional): Determines if a check for validity is executed. Checks may cause significant
        slowdown in execution time Defaults to False.
        samples (int, optional): Number of points to sample from each covariance matrix. Defaults to 1.
        eps (float, optional): Due to rounding errors, a negative sigma might be
        calculated if sigma is suppose to be 0. Sigma with magnitude less that eps
        are rounded to z Defaults to 0.000000001. Only usd for svd decomposition as cholesky is not
        applicable for positive semi-definite

    Raises:
        NotImplementedError: Current LDL decomposition is not yet implemented

    Returns:
        np.ndarray: Array of sampled point of shape N*M*S whereby N is the number of covariance matrices.
        M is the dimension of the noise, S is the number of samples
    """

    valid_decompsitions = ["svd", "cholesky"]

    if decomposition_type not in valid_decompsitions:
        raise NotImplementedError("invalid decomposition type")
    no_cov = cov.shape[0]
    x = np.random.standard_normal((no_cov, mean.shape[-1], samples))
    if decomposition_type == "svd":
        (sigma, vec) = np.linalg.eig(cov)

        # Due to rounding errors, a negative sigma might be calculated if sigma is suppose to be 0
        sigma[np.abs(sigma) < eps] = 0
        if check_valid:
            assert np.all(vec > 0)

        rot_stretch_mat = vec * np.sqrt(sigma)[:, None, :]

    elif decomposition_type == "cholesky":
        rot_stretch_mat = np.linalg.cholesky(cov)

    elif decomposition_type == "ldl":
        raise NotImplementedError("LDLT decomposition has yet to be implemented")

    x_projected = np.matmul(rot_stretch_mat, x)
    x_projected += mean[:, :, None]

    return x_projected
