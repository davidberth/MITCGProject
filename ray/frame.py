import numpy as np
from ray import caster
from ray.scene import Scene
from PIL import Image
import time
import collect
import build
from numba import njit
import params


@njit
def normalize(v):
    norm = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
    return [v[0] / norm, v[1] / norm, v[2] / norm]


@njit
def cross_product(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


class Frame:
    frame = None
    camera_pos = None
    camera_look = None
    camera_up = None
    scene = None

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.half_width = width / 2
        self.half_height = height / 2

        self.frame = np.zeros((width, height, 3), np.uint8)

        # x, y, z
        self.camera_pos = np.array(params.camera_pos)
        self.camera_look = np.array(params.camera_look)
        self.camera_up = np.array(params.camera_up)

        self.scene = Scene()

    def generate_rays(self, fov=60):
        b = time.time()
        # Normalize the camera direction and up vectors
        camera_dir = self.camera_look - self.camera_pos
        camera_dir = camera_dir / np.linalg.norm(camera_dir)
        camera_up = self.camera_up / np.linalg.norm(self.camera_up)

        # Compute the aspect ratio
        aspect_ratio = self.width / self.height

        # Calculate the right vector (perpendicular to both camera_dir and camera_up)
        camera_right = np.cross(camera_dir, camera_up)
        camera_right = camera_right / np.linalg.norm(camera_right)

        # Adjust camera_up to be exactly perpendicular to camera_dir
        camera_up = np.cross(camera_right, camera_dir)

        # Calculate the width and height of the image plane based on the field of view
        plane_height = 2 * np.tan(np.radians(fov) / 2)
        plane_width = plane_height * aspect_ratio

        # Calculate the center of the image plane
        plane_center = self.camera_pos + camera_dir

        # Calculate the start (bottom-left) corner of the image plane
        plane_start = (
            plane_center
            - (camera_right * plane_width / 2)
            - (camera_up * plane_height / 2)
        )

        # Create a grid of pixel coordinates
        u = np.linspace(0, 1, self.width)
        v = np.linspace(0, 1, self.height)
        u_grid, v_grid = np.meshgrid(u, v, indexing="xy")

        # Calculate the position of each pixel in the image plane
        pixel_positions = (
            plane_start
            + camera_right[None, None, :] * u_grid[..., None] * plane_width
            + camera_up[None, None, :] * v_grid[..., None] * plane_height
        )

        # Calculate the direction of each ray
        ray_dirs = pixel_positions - self.camera_pos
        ray_dirs = (
            ray_dirs / np.linalg.norm(ray_dirs, axis=2)[..., None]
        )  # Normalize ray directions

        e = time.time()

        print("rays generated in ", e - b)
        return ray_dirs.astype(np.float32)

    def cast_rays(self):
        directions = self.generate_rays()
        start = time.time()
        caster.cast_rays(
            self.camera_pos.astype(np.float32),
            directions,
            self.frame,
            self.scene.gtypes,
            self.scene.geometry,
            self.scene.aabbs,
            self.scene.colors,
            self.scene.haabbs,
            self.scene.haabbsi,
        )
        end = time.time()
        print("total time", end - start)

    def build_scene(self, address):
        # now we build the triangles
        if params.collect:
            collect.process_address(address)
        build.raster_to_mesh("work/dem.tif", "work/dem.obj", self.scene)
        self.scene.finalize()

    def write(self, file: str):
        self.frame = self.frame[::-1, :, :]
        img = Image.fromarray(self.frame, "RGB")
        img.save(file, "PNG")
