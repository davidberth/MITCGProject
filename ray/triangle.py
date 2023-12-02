import numpy as np
from numba import njit, f8


@njit(f8(f8[:], f8[:], f8[:], f8), fastmath=True)
def intersect(ray_origin, ray_direction, triangle_vertices, t):
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
        triangle_vertices[0:3],
        triangle_vertices[3:6],
        triangle_vertices[6:9],
    )

    # Find vectors for two edges sharing v0
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Begin calculating determinant - also used to calculate u parameter
    pvec = np.cross(ray_direction, edge2)

    # If determinant is near zero, ray lies in plane of triangle or ray is
    # parallel to plane of triangle
    det = edge1.dot(pvec)

    # NOT CULLING
    if abs(det) < 1e-8:
        return -9999.0

    inv_det = 1.0 / det

    # Calculate distance from v0 to ray origin
    tvec = ray_origin - v0

    # Calculate u parameter and test bound
    u = tvec.dot(pvec) * inv_det

    # The intersection lies outside of the triangle
    if u < 0 or u > 1:
        return -9999.0

    # Prepare to test v parameter
    qvec = np.cross(tvec, edge1)

    # Calculate v parameter and test bound
    v = ray_direction.dot(qvec) * inv_det

    # The intersection lies outside of the triangle
    if v < 0 or u + v > 1:
        return -9999.0

    # Calculate t, ray intersects triangle
    t = edge2.dot(qvec) * inv_det

    return t
