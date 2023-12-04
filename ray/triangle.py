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
def intersect(ray_origin, ray_direction, triangle_data):
    # Extract triangle vertices and normals
    v0, v1, v2 = triangle_data[:3], triangle_data[3:6], triangle_data[6:9]
    n0, n1, n2 = (
        triangle_data[9:12],
        triangle_data[12:15],
        triangle_data[15:18],
    )

    # Compute edges and h
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = cross_product(ray_direction, edge2)
    a = dot_product(edge1, h)

    # Check if ray is parallel to the triangle
    if abs(a) < 1e-6:
        return -9999.0, np.array(
            (0.0, 1.0, 0.0), dtype=np.float32
        )  # No intersection

    # Compute intersection point
    f = 1.0 / a
    s = ray_origin - v0
    u = f * dot_product(s, h)

    if u < 0.0 or u > 1.0:
        return -9999.0, np.array((0.0, 1.0, 0.0), dtype=np.float32)

    q = np.cross(s, edge1)
    v = f * dot_product(ray_direction, q)

    if v < 0.0 or u + v > 1.0:
        return -9999.0, np.array(
            (0.0, 1.0, 0.0), dtype=np.float32
        )  # No intersection

    # Compute intersection time 't'
    t = f * dot_product(edge2, q)

    if t < 1e-6:
        return -9999.0, np.array(
            (0.0, 1.0, 0.0), dtype=np.float32
        )  # No intersection

    # Interpolate normal at intersection point
    interpolated_normal = (1 - u - v) * n0 + u * n1 + v * n2

    return t, interpolated_normal.astype(np.float32)


def get_aabb(geom):
    xmin = min(geom[0], geom[3], geom[6])
    ymin = min(geom[1], geom[4], geom[7])
    zmin = min(geom[2], geom[5], geom[8])
    xmax = max(geom[0], geom[3], geom[6])
    ymax = max(geom[1], geom[4], geom[7])
    zmax = max(geom[2], geom[5], geom[8])
    return np.array([xmin, ymin, zmin, xmax, ymax, zmax])
