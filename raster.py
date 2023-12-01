from osgeo import gdal, ogr


def shapefile_to_raster(
    shapefile_path, reference_raster_path, output_raster_path, attribute_name
):
    """
    Converts a shapefile to a raster using the extents and resolution of a
    reference raster.
    The raster values are based on a specified attribute from the shapefile.

    Args:
    shapefile_path (str): The file path to the input shapefile.
    reference_raster_path (str): The file path to the reference raster.
    output_raster_path (str): The file path for the output raster.
    attribute_name (str): The attribute name in the shapefile to be used for
    raster values.
    """

    # Load the reference raster
    ref_raster = gdal.Open(reference_raster_path)
    geo_transform = ref_raster.GetGeoTransform()
    projection = ref_raster.GetProjection()
    x_min = geo_transform[0]
    y_max = geo_transform[3]
    x_res = ref_raster.RasterXSize
    y_res = ref_raster.RasterYSize
    pixel_width = geo_transform[1]
    pixel_height = -geo_transform[5]

    # Read the shapefile
    shapefile = ogr.Open(shapefile_path)
    layer = shapefile.GetLayer()

    # Create a new raster
    target_ds = gdal.GetDriverByName("GTiff").Create(
        output_raster_path, x_res, y_res, 1, gdal.GDT_Int32
    )
    target_ds.SetGeoTransform((x_min, pixel_width, 0, y_max, 0, -pixel_height))
    target_ds.SetProjection(projection)
    band = target_ds.GetRasterBand(1)
    NoData_value = -999999
    band.SetNoDataValue(NoData_value)
    band.FlushCache()

    # Rasterize (burn) the shapefile
    gdal.RasterizeLayer(
        target_ds, [1], layer, options=["ATTRIBUTE=" + attribute_name]
    )

    # Save and close the new raster
    target_ds = None
