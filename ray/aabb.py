from numba import njit
import numpy as np


@njit
def intersect(ray_origin, ray_direction, aabb):
    """
    Determine if a ray intersects with an axis-aligned bounding box.

    :param ray_origin: numpy array [x, y, z] of the ray origin
    :param ray_direction: numpy array [x, y, z] of the ray direction
    :param aabb: numpy array [min_x, min_y, min_z, max_x, max_y, max_z] of the AABB
    :return: distance along the ray where the intersection occurs, or -9999 if no intersection
    """

    # Define a small epsilon value to prevent division by zero
    epsilon = 1.0e-8

    # Initialize the parametric min and max distances as negative and positive infinity
    t_min = -99999.0
    t_max = 99999.0

    # Iterate over each axis to find intersections
    for i in range(3):
        if abs(ray_direction[i]) < epsilon:
            # Ray is parallel to the slab. No hit if origin not within slab
            if ray_origin[i] < aabb[i] or ray_origin[i] > aabb[i + 3]:
                return -99999.0
        else:
            # Compute intersection t value of ray with near and far plane of slab
            t1 = (aabb[i] - ray_origin[i]) / ray_direction[i]
            t2 = (aabb[i + 3] - ray_origin[i]) / ray_direction[i]

            # Swap t1 and t2 if needed
            if t1 > t2:
                t1, t2 = t2, t1

            # Update t_min and t_max
            if t1 > t_min:
                t_min = t1
            if t2 < t_max:
                t_max = t2

            # Exit with no collision as soon as slab intersection becomes empty
            if t_min > t_max:
                return -99999.0

    # If t_min is less than zero, the intersection is behind the ray's origin
    if t_min < 0 and t_max < 0:
        return -99999.0

    if t_min < 0:
        return t_max
    else:
        return t_min


@njit
def aabb_intersects(aabb1, aabb2):
    """
    Check if two axis-aligned bounding boxes intersect.

    :param aabb1: numpy array [min_x, min_y, min_z, max_x, max_y, max_z] of the first AABB
    :param aabb2: numpy array [min_x, min_y, min_z, max_x, max_y, max_z] of the second AABB
    :return: True if the AABBs intersect, False otherwise
    """

    # Check for overlap along each axis
    for i in range(3):
        if aabb1[i + 3] < aabb2[i] or aabb2[i + 3] < aabb1[i]:
            return False

    return True


@njit(fastmath=True)
def build_haabbs(aabbs, nb):
    haabbs = []
    haabbsi = []
    haabbsk = []
    xmin = np.min(aabbs[:, 0])
    ymin = np.min(aabbs[:, 1])
    zmin = np.min(aabbs[:, 2])
    xmax = np.max(aabbs[:, 3])
    ymax = np.max(aabbs[:, 4])
    zmax = np.max(aabbs[:, 5])

    xd = (xmax - xmin) / nb
    yd = (ymax - ymin) / nb
    zd = (zmax - zmin) / nb

    print(xmin, ymin, zmin, xmax, ymax, zmax, xd, yd, zd)

    for z in np.arange(zmin, zmax - 0.02, zd):
        for y in np.arange(ymin, ymax - 0.02, yd):
            for x in np.arange(xmin, xmax - 0.02, xd):
                oaabb = np.array(
                    (x, y, z, x + xd, y + yd, z + zd),
                    dtype=np.float32,
                )

                li = np.zeros((1500), dtype=np.int32)
                li[:] = -1
                k = 0
                for i, laabb in enumerate(aabbs):
                    if aabb_intersects(laabb, oaabb):
                        if k < 1500:
                            li[k] = i
                        k += 1

                haabbs.append(oaabb)
                haabbsi.append(li)
                haabbsk.append(k)

    return (haabbs, haabbsi, haabbsk)


@njit(fastmath=True)
def build_labs(aabbs, nb, nbl):
    xmin = np.min(aabbs[:, 0])
    ymin = np.min(aabbs[:, 1])
    zmin = np.min(aabbs[:, 2])
    xmax = np.max(aabbs[:, 3])
    ymax = np.max(aabbs[:, 4])
    zmax = np.max(aabbs[:, 5])

    xd = (xmax - xmin) / nb
    yd = (ymax - ymin) / nb
    zd = (zmax - zmin) / nb
    # build level 1 AABBs
    labs = []
    labsc = []
    # assume divisible
    nbli = int((nb + 0.001) // nbl)

    for zi, z in enumerate(np.arange(zmin, zmax - 0.002, zd * nbli)):
        for yi, y in enumerate(np.arange(ymin, ymax - 0.002, yd * nbli)):
            for xi, x in enumerate(np.arange(xmin, xmax - 0.002, xd * nbli)):
                oaabb = np.array(
                    (x, y, z, x + xd * nbli, y + yd * nbli, z + zd * nbli),
                    dtype=np.float32,
                )

                lic = np.zeros((nbli * nbli * nbli), dtype=np.int32)

                lid = 0
                for zz in range(nbli):
                    for yy in range(nbli):
                        for xx in range(nbli):
                            kidz = zi * nbli + zz
                            kidy = yi * nbli + yy
                            kidx = xi * nbli + xx
                            kid = kidx + kidy * nb + kidz * nb * nb
                            lic[lid] = kid
                            lid += 1

                labs.append(oaabb)
                labsc.append(lic)

    return (labs, labsc)
