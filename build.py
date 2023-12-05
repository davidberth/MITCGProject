import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
import geopandas as gpd
import params


def generate_color(lv, materials):
    diffuse = materials[int(lv + 0.01), :]
    ambient = diffuse * 0.05
    return np.array(
        (
            diffuse[0],
            diffuse[1],
            diffuse[2],
            ambient[0],
            ambient[1],
            ambient[2],
        ),
        dtype=np.float32,
    )


def geo_to_mesh(
    dem_raster, land_raster, use_raster, buildings, scene, sigma=1.4
):
    with rasterio.open(land_raster) as ds:
        land = ds.read(1)
    with rasterio.open(use_raster) as ds:
        use = ds.read(1)

    # Open the raster file
    ds = rasterio.open(dem_raster)
    # Read the raster data
    h = ds.read(1)
    height, width = ds.shape

    materials = np.genfromtxt("materials/materials.csv", delimiter=",")
    materials = materials[:, 1:]

    scale = params.scale

    land[use == 55] = 24

    # Apply a Gaussian filter to the raster data

    # Normalize z
    h = (h - np.min(h)) * params.height_scale
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

    lengths = np.sqrt(x**2 + y**2 + z**2)
    nx = x / lengths
    ny = y / lengths
    nz = z / lengths

    for i in range(z.shape[0] - 1):
        if i % 10 == 0:
            print("adding triangles for row ", i)
        for j in range(z.shape[1] - 1):
            # Calculate indices of the corners
            upper_left = (i, j)
            upper_right = (i, j + 1)
            lower_left = (i + 1, j)
            lower_right = (i + 1, j + 1)

            lv = land[i, j]

            # Only include triangles in the hemisphere
            if (
                phi[i, j] <= np.pi / 2 + 0.015
                and phi[i, j + 1] <= np.pi / 2 + 0.015
                and phi[i + 1, j] <= np.pi / 2 + 0.015
            ):
                scene.add_triangle(
                    (x[upper_left], z[upper_left], y[upper_left]),
                    (x[lower_left], z[lower_left], y[lower_left]),
                    (x[lower_right], z[lower_right], y[lower_right]),
                    (nx[upper_left], nz[upper_left], ny[upper_left]),
                    (nx[lower_left], nz[lower_left], ny[lower_left]),
                    (nx[lower_right], nz[lower_right], ny[lower_right]),
                    generate_color(lv, materials),
                )

                # add the dark world triangle
                scene.add_triangle(
                    (x[upper_left], -z[upper_left], y[upper_left]),
                    (x[lower_left], -z[lower_left], y[lower_left]),
                    (x[lower_right], -z[lower_right], y[lower_right]),
                    (nx[upper_left], -nz[upper_left], ny[upper_left]),
                    (nx[lower_left], -nz[lower_left], ny[lower_left]),
                    (nx[lower_right], -nz[lower_right], ny[lower_right]),
                    generate_color(lv, materials),
                )

            if (
                phi[i, j] <= np.pi / 2 + 0.015
                and phi[i + 1, j + 1] <= np.pi / 2 + 0.015
                and phi[i, j + 1] <= np.pi / 2 + 0.015
            ):
                scene.add_triangle(
                    (x[upper_left], z[upper_left], y[upper_left]),
                    (x[lower_right], z[lower_right], y[lower_right]),
                    (x[upper_right], z[upper_right], y[upper_right]),
                    (nx[upper_left], nz[upper_left], ny[upper_left]),
                    (nx[lower_right], nz[lower_right], ny[lower_right]),
                    (nx[upper_right], nz[upper_right], ny[upper_right]),
                    generate_color(lv, materials),
                )

                # add the dark world triangle
                scene.add_triangle(
                    (x[upper_left], -z[upper_left], y[upper_left]),
                    (x[lower_right], -z[lower_right], y[lower_right]),
                    (x[upper_right], -z[upper_right], y[upper_right]),
                    (nx[upper_left], -nz[upper_left], ny[upper_left]),
                    (nx[lower_right], -nz[lower_right], ny[lower_right]),
                    (nx[upper_right], -nz[upper_right], ny[upper_right]),
                    generate_color(lv, materials),
                )

    # create the building geometries
    bds = gpd.read_file(buildings)
    # bds["geometry"] = bds["geometry"].centroid
    for i, row in bds.iterrows():
        # get the coordinates of the building
        coords = row["geometry"].exterior.coords.xy
        y = []
        x = []
        hh = []
        for i in range(len(coords[0])):
            ly, lx = ds.index(coords[0][i], coords[1][i])
            if lx >= 0 and lx < h.shape[1] and ly >= 0 and ly < h.shape[0]:
                lhh = h[int(ly), int(lx)] + 0.001
                y.append(ly)
                x.append(lx)
                hh.append(lhh)
        if len(y) > 1:
            print("placing building at ", y[0], x[0])

            hh = np.array(hh, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            x = np.array(x, dtype=np.float32)
            # Normalize x and y coordinates to [-1, 1]
            gx = 2 * (x / h.shape[1]) - 1
            gy = 2 * (y / h.shape[0]) - 1

            # Convert x and y to spherical coordinates
            r = np.sqrt(gx**2 + gy**2)
            theta = np.arctan2(gy, gx)
            phi = r * np.pi / 2

            # if phi < np.pi / 2 + 0.015:

            gx = np.cos(theta) * np.sin(phi) * hh
            gy = np.sin(theta) * np.sin(phi) * hh
            gz = np.cos(phi) * hh

            gn = np.sqrt(gx**2 + gy**2 + gz**2)
            nx = gx / gn
            ny = gy / gn
            nz = gz / gn

            if phi[0] < np.pi / 2 + 0.015:
                for i in range(len(gx)):
                    j = (i + 1) % len(gx)
                    lx = gx[i]
                    ly = gy[i]
                    lz = gz[i]
                    lnx = nx[i]
                    lny = ny[i]
                    lnz = nz[i]
                    lpx = gx[j]
                    lpy = gy[j]
                    lpz = gz[j]
                    tx = lx + lnx * 0.01
                    ty = ly + lny * 0.01
                    tz = lz + lnz * 0.01
                    tpx = lpx + lnx * 0.01
                    tpy = lpy + lny * 0.01
                    tpz = lpz + lnz * 0.01

                    scene.add_triangle(
                        (lx, lz, ly),
                        (tx, tz, ty),
                        (tpx, tpz, tpy),
                        (1, 0, 0),
                        (1, 0, 0),
                        (1, 0, 0),
                        (255, 255, 255, 128, 128, 128),
                    )

                    scene.add_triangle(
                        (lx, lz, ly),
                        (tpx, tpz, tpy),
                        (lpx, lpz, lpy),
                        (1, 0, 0),
                        (1, 0, 0),
                        (1, 0, 0),
                        (255, 255, 255, 128, 128, 128),
                    )
