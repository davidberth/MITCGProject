import numpy as np
from ray import triangle
from ray import sphere
from ray import aabb
import params
import time


class Scene:
    geometry = None
    colors = None

    def __init__(self):
        self.geometry = []
        self.colors = []
        self.gtypes = []
        self.aabbs = []
        self.light_pos = []
        self.light_prop = []

        self.add_sphere((1.1, 1.1, 1.1), 0.1, (1.0, 0, 0))
        self.add_light((1.1, 1.1, 1.1), (1.0, 0.7, 0.7))

    def finalize(self):
        self.geometry = np.array(self.geometry, dtype=np.float32)
        self.colors = np.array(self.colors, dtype=np.float32)
        self.gtypes = np.array(self.gtypes).astype(np.uint8)
        self.light_pos = np.array(self.light_pos).astype(np.float32)
        self.light_prop = np.array(self.light_prop).astype(np.float32)

        # compute the AABBs
        print("computing AABBs")
        for gtype, geom in zip(self.gtypes, self.geometry):
            if gtype == 0:
                self.aabbs.append(sphere.get_aabb(geom))
            if gtype == 1:
                self.aabbs.append(triangle.get_aabb(geom))
        self.aabbs = np.array(self.aabbs, dtype=np.float32)
        print(" done")

        # build the haabbs
        print("building haabbs")
        b = time.time()
        haabbs, haabbsi, haabbsk = aabb.build_haabbs(
            self.aabbs, params.num_haabbs
        )
        labs, labsc = aabb.build_labs(
            self.aabbs, params.num_haabbs, params.num_hals
        )

        e = time.time()
        self.haabbs = np.array(haabbs, dtype=np.float32)
        self.haabbsi = np.array(haabbsi, dtype=np.int32)
        self.haabbsk = np.array(haabbsk, dtype=np.int32)
        self.labs = np.array(labs, dtype=np.float32)
        self.labsc = np.array(labsc, dtype=np.int32)
        print(" done building haabbs in", e - b)

    def add_triangle(self, v1, v2, v3, n1, n2, n3, col):
        self.gtypes.append(1)
        self.geometry.append(
            np.array(
                (
                    v1[0],
                    v1[1],
                    v1[2],
                    v2[0],
                    v2[1],
                    v2[2],
                    v3[0],
                    v3[1],
                    v3[2],
                    n1[0],
                    n1[1],
                    n1[2],
                    n2[0],
                    n2[1],
                    n2[2],
                    n3[0],
                    n3[1],
                    n3[2],
                )
            )
        )
        self.colors.append(np.array(col).astype(np.float32))

    def add_sphere(self, center, radius, col):
        self.gtypes.append(0)
        self.geometry.append(
            np.array(
                (
                    center[0],
                    center[1],
                    center[2],
                    radius,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            )
        )
        self.colors.append(np.array(col).astype(np.float32))

    def add_light(self, pos, prop):
        self.light_pos.append(pos)
        self.light_prop.append(prop)
