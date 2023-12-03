import numpy as np
from numba import njit, f4, u1, void, prange

from ray import sphere
from ray import triangle
from ray import aabb


@njit(u1[:](f4[:], f4[:], u1[:], f4[:, :], f4[:, :], u1[:, :]), fastmath=True)
def cast_ray(origin, direction, gtypes, geoms, aabbs, colors):
    t = 999999.0
    col = np.array((0, 0, 0), dtype=np.uint8)
    for gtype, laabb, geom, color in zip(gtypes, aabbs, geoms, colors):
        tp = aabb.intersect(origin, direction, laabb)
        if tp > 0:
            if gtype == 0:
                tp = sphere.intersect(origin, direction, geom)
            elif gtype == 1:
                tp = triangle.intersect(origin, direction, geom)
            if tp > 0 and tp < t:
                t = tp
                col = color

    return col


@njit(
    void(f4[:], f4[:, :, :], u1[:, :, :], u1[:], f4[:, :], f4[:, :], u1[:, :]),
    fastmath=True,
    parallel=True,
)
def cast_rays(origin, directions, buffer, gtypes, geoms, aabbs, colors):
    width, height = directions.shape[:2]

    for x in prange(width):
        for y in prange(height):
            buffer[x, y, :] = cast_ray(
                origin, directions[x, y, :], gtypes, geoms, aabbs, colors
            )
