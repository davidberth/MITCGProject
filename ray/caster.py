import numpy as np
from numba import njit, f8, u1, void, prange


@njit(u1[:](f8[:], f8[:], f8[:, :], f8[:], u1[:, :]), fastmath=True)
def cast_ray(origin, direction, centers, radii, colors):
    t = 999999.0
    col = np.array((0, 0, 0), dtype=np.uint8)
    for center, radii, color in zip(centers, radii, colors):
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
            * (center[0] * origin[0] + center[1] * origin[1] + center[2] * origin[2])
            - radii * radii
        )
        d = b * b - 4 * a * c

        if d > 0:
            d = np.sqrt(d)

            t_plus = (-b + d) / (2 * a)
            t_minus = (-b - d) / (2 * a)
            if t_plus > 0 or t_minus > 0:
                tp = t_plus if t_minus < 0 else t_minus
                if tp < t:
                    t = tp
                    col = color

    return col


@njit(
    void(f8[:], f8[:, :, :], u1[:, :, :], f8[:, :], f8[:], u1[:, :]),
    fastmath=True,
    parallel=True,
)
def cast_rays(origin, directions, buffer, centers, radii, colors):
    width, height = directions.shape[:2]

    for x in prange(width):
        for y in prange(height):
            buffer[x, y, :] = cast_ray(
                origin, directions[x, y, :], centers, radii, colors
            )
