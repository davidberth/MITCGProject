import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
import geopandas as gpd
import params
import noise
import trimesh


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


def generate_building_color():
    rv = np.random.rand() / 5.0 + 0.3
    col = [rv, rv + 0.1, rv - 0.1]
    return col


def create_background():
    shape = (params.window_height, params.window_width)
    scale = 100.0
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0
    print("generating noise")
    background = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            background[i, j] = noise.pnoise2(
                i / scale,
                j / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=0,
            )
    print(" done")
    background = background * params.background_scale + params.background_scale
    return background


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
    transform = ds.transform
    itransform = ~transform

    materials = np.genfromtxt("materials/materials.csv", delimiter=",")
    materials = materials[:, 1:]

    land[use == 55] = 24

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
                phi[i, j] <= np.pi / 2 + 0.025
                and phi[i, j + 1] <= np.pi / 2 + 0.025
                and phi[i + 1, j] <= np.pi / 2 + 0.025
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
                phi[i, j] <= np.pi / 2 + 0.025
                and phi[i + 1, j + 1] <= np.pi / 2 + 0.025
                and phi[i, j + 1] <= np.pi / 2 + 0.025
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
        area = row["geometry"].area
        cx, cy = row["geometry"].centroid.coords.xy
        lcx, lcy = itransform * (cx[0], cy[0])
        # Normalize x and y coordinates to [-1, 1]
        gx = 2 * (lcx / h.shape[1]) - 1
        gy = 2 * (lcy / h.shape[0]) - 1

        # Convert x and y to spherical coordinates
        r = np.sqrt(gx**2 + gy**2)
        phi = r * np.pi / 2
        if (
            phi < np.pi / 2 - 0.01
            and lcx >= 0
            and lcx < h.shape[1]
            and lcy >= 0
            and lcy < h.shape[0]
        ):
            base_height = h[int(lcy), int(lcx)] - 0.001
            b_height = area / 11000.0
            b_height = np.clip(b_height, 0.002, 0.025)
            b_color = generate_building_color()

            # now we transform the coords into local 3d flat space
            y = []
            x = []

            for i in range(len(coords[0])):
                lx, ly = itransform * (coords[0][i], coords[1][i])
                if lx >= 0 and lx < h.shape[1] and ly >= 0 and ly < h.shape[0]:
                    y.append(ly)
                    x.append(lx)

            if len(y) > 3:
                print("placing building at ", y[0], x[0])

                y = np.array(y, dtype=np.float32)
                x = np.array(x, dtype=np.float32)
                verts = np.array([y, x]).T

                ind = [(i, (i + 1) % len(y)) for i in range(len(y))]
                poly = trimesh.path.polygons.edges_to_polygons(ind, verts)
                extruded_polygon = trimesh.creation.extrude_polygon(
                    poly[0], b_height
                )

                for triangle in extruded_polygon.triangles:
                    y = triangle[:, 0]
                    x = triangle[:, 1]
                    j = triangle[:, 2] + base_height

                    # Normalize x and y coordinates to [-1, 1]
                    gx = 2 * (x / h.shape[1]) - 1
                    gy = 2 * (y / h.shape[0]) - 1

                    # Convert x and y to spherical coordinates
                    r = np.sqrt(gx**2 + gy**2)
                    theta = np.arctan2(gy, gx)
                    phi = r * np.pi / 2

                    gx = np.cos(theta) * np.sin(phi) * j
                    gy = np.sin(theta) * np.sin(phi) * j
                    gz = np.cos(phi) * j

                    scene.add_triangle(
                        (gx[0], gz[0], gy[0]),
                        (gx[1], gz[1], gy[1]),
                        (gx[2], gz[2], gy[2]),
                        (1, 0, 0),
                        (1, 0, 0),
                        (1, 0, 0),
                        (
                            b_color[0],
                            b_color[1],
                            b_color[2],
                            b_color[0] / 10.0,
                            b_color[1] / 10.0,
                            b_color[2] / 10.0,
                        ),
                    )
