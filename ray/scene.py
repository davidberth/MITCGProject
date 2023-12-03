import numpy as np
from ray import triangle
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
        b = time.time()
        haabbs, haabbsi, haabbsk, labs, labsc = aabb.build_haabbs(
            self.aabbs, params.num_haabbs
        )
        e = time.time()
        self.haabbs = np.array(haabbs, dtype=np.float32)
        self.haabbsi = np.array(haabbsi, dtype=np.int32)
        self.haabbsk = np.array(haabbsk, dtype=np.int32)
        self.labs = np.array(labs, dtype=np.float32)
        self.labsc = np.array(labsc, dtype=np.int32)
        print(" done building haabbs in", e - b)

    def add_triangle(self, v1, v2, v3, col):
        self.gtypes.append(1)
        self.geometry.append(
            np.array(
                (v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2])
            )
        )
        self.colors.append(np.array(col).astype(np.uint8))
