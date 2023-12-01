import geocoder
from pyproj import Transformer
import geopandas as gpd
import glob
import os
from osgeo import gdal
import sys
import raster

width = 1000
half_width = width / 2

"""
ogr2ogr = "ogr2ogr.exe "
gdal_translsate = "gdal_translate.exe "
"""

transformer = Transformer.from_crs("EPSG:4326", "EPSG:26919")

g = geocoder.osm(sys.argv[1])
lat, lon = g.latlng
print("lat/lon center", lat, lon)

cx, cy = transformer.transform(lat, lon)
print("projected center", cy, cx)

ymin = cy - half_width
ymax = cy + half_width
xmin = cx - half_width
xmax = cx + half_width

files = glob.glob("work/*")
for f in files:
    try:
        os.remove(f)
    except:
        print(f"did not remove file {f}")

sources = ["data/landcover/landcover.gpkg", "data/buildings/buildings.gpkg"]
destinations = ["work/land.shp", "work/buildings.shp"]
layers = ["LANDCOVER_LANDUSE_POLY", None]

for source, destination, layer in zip(sources, destinations, layers):
    print("subsetting ", source)
    gpd.read_file(source, bbox=(xmin, ymin, xmax, ymax), layer=layer).to_file(
        destination
    )

source = "data/bare_elev/Lidar_Elevation_2013to2021.jp2"
destination = "work/dem.tif"


print("subsetting DEM")

ds = gdal.Open(source)
ds = gdal.Translate(destination, ds, projWin=[xmin, ymax, xmax, ymin])
ds = None

print(" done")

# now let's convert the land shapefile to a raster

raster.shapefile_to_raster(
    "work/land.shp", "work/dem.tif", "work/land.tif", "COVERCODE"
)
