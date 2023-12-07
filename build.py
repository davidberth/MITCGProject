import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
import geopandas as gpd
import params
import noise
import trimesh


def generate_color(lv, materials, background, idx):
    diffuse = materials[int(lv + 0.01), :3]
    ns = materials[int(lv + 0.01), 4] * 20.0
    nv = background[idx[0], idx[1]]
    nc = 1 + nv * ns * 3.0 - nv * ns * 1.5 - 0.2
    ambient = diffuse * 0.02
    return np.array(
        (
            diffuse[0] * nc,
            diffuse[1] * nc,
            diffuse[2] * nc,
            ambient[0] * nc,
            ambient[1] * nc,
            ambient[2] * nc,
        ),
        dtype=np.float32,
    )


def generate_building_color():
    rv = np.random.rand() / 5.0 + 0.1
    col = [rv + 0.05, rv, rv]
    return np.array(col)
    # return [0.9, 0.9, 0.9]


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
                repeatx=params.window_width,
                repeaty=params.window_height,
                base=0,
            )
    print(" done")
    background = background * params.background_scale + params.background_scale
    return background


def geo_to_mesh(
    dem_raster,
    land_raster,
    use_raster,
    buildings,
    scene,
    background,
    sigma=0.5,
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
    land = np.clip(land, 0, 25)
    ha = materials[land.astype(np.int32), 4]

    h = h + ha * params.materials_height_adjust

    sc = 15
    indices_y = (
        np.arange(0, h.shape[0] * sc, sc) % background.shape[1]
    ).astype(np.int32)
    indices_x = (
        np.arange(0, h.shape[1] * sc, sc) % background.shape[0]
    ).astype(np.int32)
    indices = np.ix_(indices_x, indices_y)
    hba = background[indices]
    h = h + hba * 0.2

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
                    generate_color(lv, materials, hba, lower_left),
                )

                # add the dark world triangle
                scene.add_triangle(
                    (x[upper_left], -z[upper_left], y[upper_left]),
                    (x[lower_left], -z[lower_left], y[lower_left]),
                    (x[lower_right], -z[lower_right], y[lower_right]),
                    (nx[upper_left], -nz[upper_left], ny[upper_left]),
                    (nx[lower_left], -nz[lower_left], ny[lower_left]),
                    (nx[lower_right], -nz[lower_right], ny[lower_right]),
                    generate_color(lv, materials, hba, lower_left),
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
                    generate_color(lv, materials, hba, lower_left),
                )

                # add the dark world triangle
                scene.add_triangle(
                    (x[upper_left], -z[upper_left], y[upper_left]),
                    (x[lower_right], -z[lower_right], y[lower_right]),
                    (x[upper_right], -z[upper_right], y[upper_right]),
                    (nx[upper_left], -nz[upper_left], ny[upper_left]),
                    (nx[lower_right], -nz[lower_right], ny[lower_right]),
                    (nx[upper_right], -nz[upper_right], ny[upper_right]),
                    generate_color(lv, materials, hba, lower_left),
                )

    # place trees
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            if land[i, j] == 9 or land[i, j] == 10:
                if np.random.rand() > 0.75:
                    ty = 2 * (i / h.shape[1]) - 1
                    tx = 2 * (j / h.shape[0]) - 1
                    tz = h[i, j] + 0.025 + np.random.random() * 0.01
                    # Convert x and y to spherical coordinates
                    r = np.sqrt(tx**2 + ty**2)
                    theta = np.arctan2(ty, tx)
                    phi = r * np.pi / 2

                    if phi < np.pi / 2 - 0.01:
                        # Convert spherical coordinates to 3D coordinates
                        tx = np.cos(theta) * np.sin(phi) * tz
                        ty = np.sin(theta) * np.sin(phi) * tz
                        tz = np.cos(phi) * tz
                        print("adding tree", tx, ty, tz)
                        rad = np.random.rand() * 0.0030 + 0.0012
                        scene.add_sphere(
                            (tx, tz, ty),
                            rad,
                            (0.0, 0.3, 0.0, 0.0, 0.01, 0.0),
                        )
                        scene.add_sphere(
                            (tx, -tz, ty),
                            rad,
                            (0.0, 0.3, 0.0, 0.0, 0.01, 0.0),
                        )

    # create the building geometries
    bds = gpd.read_file(buildings)
    bds = bds.explode()
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
                ly, lx = itransform * (coords[0][i], coords[1][i])
                if lx >= 0 and lx < h.shape[1] and ly >= 0 and ly < h.shape[0]:
                    y.append(ly)
                    x.append(lx)

            if len(y) > 3:
                # print("placing building at ", y[0], x[0])

                y = np.array(y, dtype=np.float32)
                x = np.array(x, dtype=np.float32)
                verts = np.array([y, x]).T

                ind = [(i, (i + 1) % len(y)) for i in range(len(y))]
                poly = trimesh.path.polygons.edges_to_polygons(ind, verts)
                extruded_polygon = trimesh.creation.extrude_polygon(
                    poly[0], b_height
                )

                # convert the vertices of the mesh to our spherical coordinates
                x = extruded_polygon.vertices[:, 0]
                y = extruded_polygon.vertices[:, 1]
                z = extruded_polygon.vertices[:, 2] + base_height

                gx = 2 * (x / h.shape[1]) - 1
                gy = 2 * (y / h.shape[0]) - 1

                # Convert x and y to spherical coordinates
                r = np.sqrt(gx**2 + gy**2)
                theta = np.arctan2(gy, gx)
                phi = r * np.pi / 2

                gx = np.cos(theta) * np.sin(phi) * z
                gz = np.sin(theta) * np.sin(phi) * z
                gy = np.cos(phi) * z

                verts = []
                faces = extruded_polygon.faces
                for bx, by, bz in zip(gx, gy, gz):
                    verts.append((bx, by, bz))
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                # trimesh.repair.fix_winding(mesh)
                # mesh.fix_normals()

                # build top detection
                tops = []
                for tri in extruded_polygon.triangles:
                    zs = tri[:, 2]
                    if np.all(zs > 0.001):
                        is_top = True
                    else:
                        is_top = False
                    tops.append(is_top)

                if np.sum(tops) < 0.5:
                    print("BAAAAAAAAAAD")

                for triangle, top in zip(mesh.triangles, tops):
                    x = triangle[:, 0]
                    y = triangle[:, 1]
                    z = triangle[:, 2]

                    if top:
                        ln = np.sqrt(x[0] ** 2 + y[0] ** 2 + z[0] ** 2)
                        norm = np.array((x[0] / ln, y[0] / ln, z[0] / ln))
                        col = b_color
                    else:
                        # norm = np.array((0.0, 0.0, 0.0))
                        col = b_color / 3.0
                        norm = np.array((0.0, 0.0, 0.0))
                    col2 = np.array((col[2], col[1], col[0])) * 0.7

                    scene.add_triangle(
                        (x[0], y[0], z[0]),
                        (x[1], y[1], z[1]),
                        (x[2], y[2], z[2]),
                        norm,
                        norm,
                        norm,
                        (
                            col[0],
                            col[1],
                            col[2],
                            col[0] / 12.0,
                            col[1] / 12.0,
                            col[2] / 12.0,
                        ),
                    )

                    # add the dark side
                    scene.add_triangle(
                        (x[0], -y[0], z[0]),
                        (x[1], -y[1], z[1]),
                        (x[2], -y[2], z[2]),
                        norm,
                        norm,
                        norm,
                        (
                            col2[0],
                            col2[1],
                            col2[2],
                            col2[0] / 12.0,
                            col2[1] / 12.0,
                            col2[2] / 12.0,
                        ),
                    )
