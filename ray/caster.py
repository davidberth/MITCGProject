import numpy as np
from numba import njit, prange

from ray import sphere
from ray import triangle
from ray import aabb


@njit(fastmath=True)
def cast_shadow(
    origin, direction, gtypes, geoms, aabbs, haabbs, hi, hk, labs, labsc
):
    # TODO check intersection not past the light source
    t = 999999.0
    obji = -1
    for k in range(labs.shape[0]):
        if aabb.intersect(origin, direction, labs[k, :]) > 0.0:
            for i in labsc[k, :]:
                ha = haabbs[i, :]
                if hk[i] > 0 and aabb.intersect(origin, direction, ha) > 0.0:
                    for j in hi[i]:
                        if j == -1:
                            break
                        gtype = gtypes[j]
                        laabb = aabbs[j, :]
                        if aabb.intersect(origin, direction, laabb) > 0.0:
                            if gtype == 0:
                                tp, _ = sphere.intersect(
                                    origin, direction, geoms[j, :]
                                )
                            elif gtype == 1:
                                tp, _ = triangle.intersect(
                                    origin, direction, geoms[j, :]
                                )
                            if tp > 0 and tp < t:
                                obji = j
                                return obji
    return obji


@njit(fastmath=True)
def cast_ray(
    origin,
    direction,
    gtypes,
    geoms,
    aabbs,
    materials,
    haabbs,
    hi,
    hk,
    labs,
    labsc,
    light_pos,
    light_prop,
):
    t = 999999.0
    col = np.array((0, 0, 0), dtype=np.float32)
    hit = 0
    rnorm = np.array((0.0, 1.0, 0.0), dtype=np.float32)
    obji = -1
    for k in range(labs.shape[0]):
        if aabb.intersect(origin, direction, labs[k, :]) > 0:
            for i in labsc[k, :]:
                ha = haabbs[i, :]
                if hk[i] > 0 and aabb.intersect(origin, direction, ha) > 0:
                    for j in hi[i]:
                        if j == -1:
                            break
                        gtype = gtypes[j]
                        laabb = aabbs[j, :]
                        if aabb.intersect(origin, direction, laabb) > 0:
                            if gtype == 0:
                                tp, norm = sphere.intersect(
                                    origin, direction, geoms[j, :]
                                )
                            elif gtype == 1:
                                tp, norm = triangle.intersect(
                                    origin, direction, geoms[j, :]
                                )
                            if tp > 0 and tp < t:
                                t = tp
                                rnorm = norm
                                obji = j

    if obji > -1:
        hit = 1
        diffuse = materials[obji, :3]
        ambient = materials[obji, 3:6]
        specular = materials[obji, :3]
        col = ambient
        shininess = 25.0
        pos = origin + t * direction

        num_lights = light_pos.shape[0]
        for li in range(num_lights):
            light_dir = light_pos[li] - pos
            lsquared = (
                light_dir[0] * light_dir[0]
                + light_dir[1] * light_dir[1]
                + light_dir[2] * light_dir[2]
            )

            if lsquared < light_prop[li, 4]:
                light_dir /= np.sqrt(lsquared)
                light_intens = 1.0 - lsquared / light_prop[li, 4]

                posa = pos + 0.0001 * light_dir
                # let's find out if we are in shadow
                shadow_obj = cast_shadow(
                    posa,
                    light_dir,
                    gtypes,
                    geoms,
                    aabbs,
                    haabbs,
                    hi,
                    hk,
                    labs,
                    labsc,
                )

                if shadow_obj == -1:
                    lcolor = light_prop[li, 1:4]
                    ldot = (
                        rnorm[0] * light_dir[0]
                        + rnorm[1] * light_dir[1]
                        + rnorm[2] * light_dir[2]
                    )
                    if ldot < 0.0:
                        ldot = 0.0

                    # Compute reflection of the light around the normal
                    reflection = 2 * ldot * rnorm - light_dir
                    reflection /= np.sqrt(
                        reflection[0] * reflection[0]
                        + reflection[1] * reflection[1]
                        + reflection[2] * reflection[2]
                    )

                    # Compute view direction
                    view_dir = -direction
                    view_dir /= np.sqrt(
                        view_dir[0] * view_dir[0]
                        + view_dir[1] * view_dir[1]
                        + view_dir[2] * view_dir[2]
                    )
                    rdot = max(
                        0,
                        reflection[0] * view_dir[0]
                        + reflection[1] * view_dir[1]
                        + reflection[2] * view_dir[2],
                    )
                    # phong model
                    col = col + (
                        lcolor
                        * light_intens
                        * (diffuse * ldot + specular * (rdot**shininess))
                    ).astype(np.float32)

    return col, hit


@njit(
    fastmath=True,
    parallel=True,
)
def cast_rays(
    origin,
    directions,
    buffer,
    hit,
    gtypes,
    geoms,
    aabbs,
    materials,
    haabbs,
    hi,
    hk,
    labs,
    labsc,
    light_pos,
    light_prop,
):
    width, height = directions.shape[:2]

    for y in prange(height):
        print("processing row", y + 1, "of", height)
        for x in prange(width):
            buffer[x, y, :], hit[x, y] = cast_ray(
                origin,
                directions[x, y, :],
                gtypes,
                geoms,
                aabbs,
                materials,
                haabbs,
                hi,
                hk,
                labs,
                labsc,
                light_pos,
                light_prop,
            )
