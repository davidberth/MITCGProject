import numpy as np
from numba import njit, f8, u1, void, prange

from ray import sphere
from ray import triangle


@njit(u1[:](f8[:], f8[:], u1[:], f8[:, :], u1[:, :]), fastmath=True)
def cast_ray(origin, direction, gtypes, geoms, colors):
    t = 999999.0
    col = np.array((0, 0, 0), dtype=np.uint8)
    for gtype, geom, color in zip(gtypes, geoms, colors):
        if gtype == 0:
            tp = sphere.intersect(origin, direction, geom, t)
        elif gtype == 1:
            tp = triangle.intersect(origin, direction, geom, t)
        if tp > 0 and tp < t:
            t = tp
            col = color

    return col


@njit(
    void(f8[:], f8[:, :, :], u1[:, :, :], u1[:], f8[:, :], u1[:, :]),
    fastmath=True,
    parallel=True,
)
def cast_rays(origin, directions, buffer, gtypes, geoms, colors):
    width, height = directions.shape[:2]

    for x in prange(width):
        for y in prange(height):
            buffer[x, y, :] = cast_ray(
                origin, directions[x, y, :], gtypes, geoms, colors
            )
