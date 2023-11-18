import numpy as np
from ray import caster
from PIL import Image


class Frame:
    frame = None
    camera_pos = None

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.half_width = width / 2
        self.half_height = height / 2

        self.frame = np.zeros((width, height, 3), np.uint8)

        # x, y, z
        self.camera_pos = np.array([0.0, 0.0, -10.0])
        self.viewport_z = -5.0
        self.viewport_radius = 2.0

    def cast_rays(self):
        directions = caster.cast_rays(
            self.frame, self.camera_pos, self.viewport_z, self.viewport_radius
        )
        print(directions)

    def write(self, file: str):
        img = Image.fromarray(self.frame, "RGB")
        img.save(file, "PNG")
