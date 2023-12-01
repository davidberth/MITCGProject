import numpy as np
from numba import njit, f8


@njit(f8(f8[:], f8[:], f8[:], f8), fastmath=True)
def intersect(origin, direction, geom, t):
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
        return -9999.0
    else:
        d = np.sqrt(d)
        t_plus = (-b + d) / (2 * a)
        t_minus = (-b - d) / (2 * a)

        if t_minus < 0:
            return t_plus
        else:
            return t_minus
