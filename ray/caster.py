import numpy as np
from numba import njit, f4, u1, void, prange

from ray import sphere
from ray import triangle
from ray import aabb


@njit(fastmath=True)
def cast_ray(origin, direction, gtypes, geoms, aabbs, colors, haabbs, hi):
    t = 999999.0
    col = np.array((0, 0, 0), dtype=np.uint8)

    for i in range(haabbs.shape[0]):
        ha = haabbs[i, :]
        if aabb.intersect(origin, direction, ha) > 0:
            for j in hi[i]:
                if j == -1:
                    break
                gtype = gtypes[j]
                laabb = aabbs[j, :]
                if aabb.intersect(origin, direction, laabb) > 0:
                    if gtype == 0:
                        tp = sphere.intersect(origin, direction, geoms[j, :])
                    elif gtype == 1:
                        tp = triangle.intersect(origin, direction, geoms[j, :])
                    if tp > 0 and tp < t:
                        t = tp
                        col = colors[j, :]

    return col


@njit(
    fastmath=True,
    parallel=True,
)
def cast_rays(
    origin, directions, buffer, gtypes, geoms, aabbs, colors, haabbs, hi
):
    width, height = directions.shape[:2]

    for x in prange(width):
        for y in prange(height):
            buffer[x, y, :] = cast_ray(
                origin,
                directions[x, y, :],
                gtypes,
                geoms,
                aabbs,
                colors,
                haabbs,
                hi,
            )
