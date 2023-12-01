import numpy as np


class Scene:
    geometry = None
    colors = None

    def __init__(self):
        self.geometry = []
        self.colors = []
        self.gtypes = []

        self.gtypes.append(1)
        self.geometry.append(
            np.array((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0))
        )
        self.colors.append(np.array((255, 0, 0)).astype(np.uint8))

        self.gtypes.append(1)
        self.geometry.append(
            np.array((0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 4.0, 4.0, 1.0))
        )
        self.colors.append(np.array((0, 0, 255)).astype(np.uint8))

        """for i in range(500):
            cent = (
                np.random.rand() * 10.0 - 5.0,
                np.random.rand() * 10.0 - 5.0,
                np.random.rand() - 0.5,
            )

            rad = np.random.rand() + 0.2
            color = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            )
            self.gtypes.append(0)

            self.geometry.append(np.array((cent[0], cent[1], cent[2], rad)))
            self.colors.append(np.array(color).astype(np.uint8))"""
        self.geometry = np.array(self.geometry)
        self.colors = np.array(self.colors)
        self.gtypes = np.array(self.gtypes).astype(np.uint8)
