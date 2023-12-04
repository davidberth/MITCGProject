import numpy as np
from numba import njit, f8


@njit
def dot_product(a, b):
    """
    Compute the dot product of two vectors.

    :param a: The first vector (list)
    :param b: The second vector (list)
    :return: The dot product of a and b
    """
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit
def cross_product(a, b):
    """
    Compute the cross product of two 3D vectors.

    :param a: The first vector (numpy array)
    :param b: The second vector (numpy array)
    :return: The cross product of a and b
    """
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


@njit(fastmath=True)
def intersect(ray_origin, ray_direction, geom):
    """
    Compute the intersection of a ray and a triangle.

    :param ray_origin: The origin of the ray (numpy array)
    :param ray_direction: The direction of the ray (numpy array)
    :param triangle_vertices: A list of three vertices,
    each a numpy array, representing a triangle
    :return: The value t along the ray where it intersects the triangle
    or None if no intersection.
    """
    # Unpack the triangle vertices
    v0, v1, v2 = (
        geom[0:3],
        geom[3:6],
        geom[6:9],
    )

    # Find vectors for two edges sharing v0
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Begin calculating determinant - also used to calculate u parameter
    pvec = cross_product(ray_direction, edge2)

    # If determinant is near zero, ray lies in plane of triangle or ray is
    # parallel to plane of triangle
    det = dot_product(edge1, pvec)

    # NOT CULLING
    if abs(det) < 1e-8:
        return -9999.0, np.array((0.0, 1.0, 0.0), dtype=np.float32)

    inv_det = 1.0 / det

    # Calculate distance from v0 to ray origin
    tvec = ray_origin - v0

    # Calculate u parameter and test bound
    u = dot_product(tvec, pvec) * inv_det

    # The intersection lies outside of the triangle
    if u < 0 or u > 1:
        return -9999.0, np.array((0.0, 1.0, 0.0), dtype=np.float32)

    # Prepare to test v parameter
    qvec = cross_product(tvec, edge1)

    # Calculate v parameter and test bound
    v = dot_product(ray_direction, qvec) * inv_det

    # The intersection lies outside of the triangle
    if v < 0 or u + v > 1:
        return -9999.0, np.array((0.0, 1.0, 0.0), dtype=np.float32)

    # Calculate t, ray intersects triangle
    t = dot_product(edge2, qvec) * inv_det

    # for now, return the normal of the first vertex
    return t, geom[9:12]


def get_aabb(geom):
    xmin = min(geom[0], geom[3], geom[6])
    ymin = min(geom[1], geom[4], geom[7])
    zmin = min(geom[2], geom[5], geom[8])
    xmax = max(geom[0], geom[3], geom[6])
    ymax = max(geom[1], geom[4], geom[7])
    zmax = max(geom[2], geom[5], geom[8])
    return np.array([xmin, ymin, zmin, xmax, ymax, zmax])
