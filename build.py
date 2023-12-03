import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter


def generate_color(z):
    red = z * 255.0
    red = np.clip(red, 0, 255)
    blue = 255.0 - red
    return [red, 255, blue]


def raster_to_mesh(input_raster, output_obj, scene, sigma=1):
    """
    Converts a raster to a mesh of triangles and writes it to an .obj file.

    Parameters:
    input_raster (str): The path to the input raster file.
    output_obj (str): The path to the output .obj file.
    scene: our scene
    sigma (float): The standard deviation for the Gaussian kernel. The default is 1.
    """

    # Open the raster file
    with rasterio.open(input_raster) as ds:
        # Read the raster data
        h = ds.read(1)

    # Apply a Gaussian filter to the raster data

    # Normalize z
    h = (h - np.min(h)) * 0.05
    h += 1

    h = gaussian_filter(h, sigma=sigma)

    # Create 2D arrays of x and y coordinates
    y, x = np.mgrid[: h.shape[0], : h.shape[1]]

    # Normalize x and y coordinates to [-1, 1]
    x = 2 * (x / h.shape[1]) - 1
    y = 2 * (y / h.shape[0]) - 1

    # Convert x and y to spherical coordinates
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    phi = r * np.pi / 2

    # Convert spherical coordinates to 3D coordinates
    x = np.cos(theta) * np.sin(phi) * h
    y = np.sin(theta) * np.sin(phi) * h
    z = np.cos(phi) * h

    for i in range(z.shape[0] - 1):
        if i % 10 == 0:
            print("adding triangles for row ", i)
        for j in range(z.shape[1] - 1):
            # Calculate indices of the corners
            upper_left = (i, j)
            upper_right = (i, j + 1)
            lower_left = (i + 1, j)
            lower_right = (i + 1, j + 1)

            # Only include triangles in the hemisphere
            if (
                phi[i, j] <= np.pi / 2
                and phi[i, j + 1] <= np.pi / 2
                and phi[i + 1, j] <= np.pi / 2
            ):
                scene.add_triangle(
                    (x[upper_left], z[upper_left], y[upper_left]),
                    (x[lower_left], z[lower_left], y[lower_left]),
                    (x[lower_right], z[lower_right], y[lower_right]),
                    generate_color(h[upper_right] - 1),
                )

            if (
                phi[i, j] <= np.pi / 2
                and phi[i + 1, j + 1] <= np.pi / 2
                and phi[i, j + 1] <= np.pi / 2
            ):
                scene.add_triangle(
                    (x[upper_left], z[upper_left], y[upper_left]),
                    (x[lower_right], z[lower_right], y[lower_right]),
                    (x[upper_right], z[upper_right], y[upper_right]),
                    generate_color(h[upper_right] - 1),
                )
