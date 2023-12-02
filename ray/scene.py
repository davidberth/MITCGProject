import numpy as np


class Scene:
    geometry = None
    colors = None

    def __init__(self):
        self.geometry = []
        self.colors = []
        self.gtypes = []

    def finalize(self):
        self.geometry = np.array(self.geometry)
        self.colors = np.array(self.colors)
        self.gtypes = np.array(self.gtypes).astype(np.uint8)

    def add_triangle(self, v1, v2, v3, col):
        self.gtypes.append(1)
        self.geometry.append(
            np.array(
                (v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2])
            )
        )
        self.colors.append(np.array(col).astype(np.uint8))
