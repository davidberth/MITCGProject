import numpy as np


class Scene:
    centers = None
    radii = None
    colors = None

    def __init__(self):
        cent = []
        rad = []
        col = []
        for i in range(50):
            cent.append(
                (
                    np.random.rand() * 10.0 - 5.0,
                    np.random.rand() * 10.0 - 5.0,
                    np.random.rand() - 0.5,
                )
            )
            rad.append(np.random.rand() + 0.2)
            col.append(
                (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                )
            )
        self.centers = np.array(cent)
        self.radii = np.array(rad)
        self.colors = np.array(col, dtype=np.uint8)
