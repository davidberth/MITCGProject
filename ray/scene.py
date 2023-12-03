import numpy as np
from ray import triangle
from ray import aabb
import params


class Scene:
    geometry = None
    colors = None

    def __init__(self):
        self.geometry = []
        self.colors = []
        self.gtypes = []
        self.aabbs = []

    def finalize(self):
        self.geometry = np.array(self.geometry, dtype=np.float32)
        self.colors = np.array(self.colors, dtype=np.uint8)
        self.gtypes = np.array(self.gtypes).astype(np.uint8)

        # compute the AABBs
        print("computing AABBs")
        for gtype, geom in zip(self.gtypes, self.geometry):
            if gtype == 1:
                self.aabbs.append(triangle.get_aabb(geom))
        self.aabbs = np.array(self.aabbs, dtype=np.float32)
        print(" done")

        # build the haabbs
        print("building haabbs")
        haabbs = []
        haabbsi = []
        xmin = np.min(self.aabbs[:, 0])
        ymin = np.min(self.aabbs[:, 1])
        zmin = np.min(self.aabbs[:, 2])
        xmax = np.max(self.aabbs[:, 3])
        ymax = np.max(self.aabbs[:, 4])
        zmax = np.max(self.aabbs[:, 5])

        nb = params.num_haabbs
        xd = (xmax - xmin) / nb
        yd = (ymax - ymin) / nb
        zd = (zmax - zmin) / nb

        print(xmin, ymin, zmin, xmax, ymax, zmax, xd, yd, zd)
        for x in np.arange(xmin, xmax, xd):
            for y in np.arange(ymin, ymax, yd):
                for z in np.arange(zmin, zmax, zd):
                    oaabb = np.array(
                        (x, y, z, x + xd, y + yd, z + zd),
                        dtype=np.float32,
                    )
                    haabbs.append(oaabb)
                    li = []
                    for i, laabb in enumerate(self.aabbs):
                        if aabb.aabb_intersects(laabb, oaabb):
                            li.append(i)
                    print(x, y, z, len(li))
                    haabbsi.append(li)

        print(" done building haabbs")

    def add_triangle(self, v1, v2, v3, col):
        self.gtypes.append(1)
        self.geometry.append(
            np.array(
                (v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2])
            )
        )
        self.colors.append(np.array(col).astype(np.uint8))
