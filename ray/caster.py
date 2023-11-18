import numpy as np
from numba import njit, f8, u1


@njit(f8[:, :, :](u1[:, :, :], f8[:], f8, f8), fastmath=True)
def cast_rays(frame, camera_pos, viewport_z, viewport_radius):
    width = frame.shape[0]
    height = frame.shape[1]

    half_width = width / 2
    half_height = height / 2

    output = np.zeros((width, height, 3))

    for x in range(frame.shape[0]):
        xx = viewport_radius * (x / half_width - 1.0)
        for y in range(frame.shape[1]):
            yy = viewport_radius * (y / half_height - 1.0)
            dir = np.array([xx, yy, viewport_z]) - camera_pos
            dir = dir / (xx**2 + yy**2 + viewport_z**2) ** 0.5
            output[x, y, :] = camera_pos + dir * 40.0
    return output
