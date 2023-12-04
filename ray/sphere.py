import numpy as np
from numba import njit


@njit(fastmath=True)
def intersect(origin, direction, geom):
    """
    Computes the intersection of a ray and a sphere.

    Parameters:
    origin (numpy.ndarray): The origin of the ray.
    direction (numpy.ndarray): The direction of the ray.
    geom (numpy.ndarray): The geometry of the sphere,
    where the first three elements are the center of the
    sphere and the fourth element is the radius.
    t (float): The time parameter for the ray.

    Returns:
    float: The time at which the ray intersects the sphere.
    Returns -9999.0 if there is no intersection.
    """
    center = geom[:3]
    radius = geom[3]
    a = (
        direction[0] * direction[0]
        + direction[1] * direction[1]
        + direction[2] * direction[2]
    )
    b = 2 * (
        direction[0] * origin[0]
        + direction[1] * origin[1]
        + direction[2] * origin[2]
    ) - 2 * (
        center[0] * direction[0]
        + center[1] * direction[1]
        + center[2] * direction[2]
    )
    c = (
        origin[0] * origin[0]
        + origin[1] * origin[1]
        + origin[2] * origin[2]
        + center[0] * center[0]
        + center[1] * center[1]
        + center[2] * center[2]
        - 2
        * (
            center[0] * origin[0]
            + center[1] * origin[1]
            + center[2] * origin[2]
        )
        - radius * radius
    )
    d = b * b - 4 * a * c

    if d < 0:
        return -9999.0, np.array((0.0, 1.0, 0.0), dtype=np.float32)
    else:
        d = np.sqrt(d)
        t_plus = (-b + d) / (2 * a)
        t_minus = (-b - d) / (2 * a)

        if t_minus < 0:
            t = t_plus
        else:
            t = t_minus

    loc = origin + t * direction
    normal = (loc - center).astype(np.float32)
    normal /= np.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
    return t, normal


def get_aabb(geom):
    return np.array(
        (
            geom[0] - geom[3],
            geom[1] - geom[3],
            geom[2] - geom[3],
            geom[0] + geom[3],
            geom[1] + geom[3],
            geom[2] + geom[3],
        )
    )
