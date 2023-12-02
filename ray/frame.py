import numpy as np
from ray import caster
from ray.scene import Scene
from PIL import Image
import time
import collect
import build


class Frame:
    frame = None
    camera_pos = None
    scene = None

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

        self.scene = Scene()

    def get_rays(self):
        xx = np.linspace(
            -self.viewport_radius,
            self.viewport_radius,
            self.width,
        )
        yy = np.linspace(
            -self.viewport_radius,
            self.viewport_radius,
            self.height,
        )
        x, y = np.meshgrid(xx, yy)
        z = np.full_like(x, self.viewport_z)
        target_points = np.dstack((x, y, z))
        directions = target_points - self.camera_pos
        norms = np.linalg.norm(directions, axis=2)
        directions[:, :, 0] /= norms
        directions[:, :, 1] /= norms
        directions[:, :, 2] /= norms

        return directions

    def cast_rays(self):
        directions = self.get_rays()
        start = time.time()
        caster.cast_rays(
            self.camera_pos,
            directions,
            self.frame,
            self.scene.gtypes,
            self.scene.geometry,
            self.scene.colors,
        )
        end = time.time()
        print("total time", end - start)

    def build_scene(self, address):
        # now we build the triangles
        collect.process_address(address)
        build.raster_to_mesh("work/dem.tif", "work/dem.obj", self.scene)
        self.scene.finalize()

    def write(self, file: str):
        img = Image.fromarray(self.frame, "RGB")
        img.save(file, "PNG")
