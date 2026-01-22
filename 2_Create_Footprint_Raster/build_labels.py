"""
Build Labels Helper Script
--------------------------

Purpose:
    This script facilitates the creation of a building footprint raster from a vector file (SHP or GEOJSON).
    It ensures the output raster matches the dimensions, projection, and geotransform of a reference source raster.

Usage:
    Run the script and provide the following inputs when prompted:
    1. Path to the original source raster (image).
    2. Path to the vector building footprint file (SHP/GEOJSON).
    3. Output folder path.
    4. Name for the output raster file (without extension).

Methodology:
    1. Takes user input for file paths.
    2. Calls `create_building_mask` to generate the initial raster mask from the vector data.
    3. Post-processes the generated raster to ensure it exactly matches the spatial reference system (Projection, GeoTransform, GCPs) of the source image using GDAL.

Author:
    - Lonnie Byrnside III

Dependencies:
    - create_building_mask (local module)
    - osgeo.gdal
"""

import os
from osgeo import gdal
from create_building_mask import create_building_mask


def main(src_raster_path, src_vector_path, dst_path):
    create_building_mask(
            src_raster_path, src_vector_path, npDistFileName=dst_path,
            noDataValue=0, burn_values=255
    )


src_raster_path = input("Enter original image raster path: ")
src_vector_path = input("Enter vector building footprint SHP/GEOJSON path: ")
dst_path_fldr = input("Enter output folder path: ")
dst_path_name = input("Enter output image name: ")
dst_path = os.path.join(dst_path_fldr, dst_path_name + ".tif")


main(src_raster_path, src_vector_path, dst_path)


dataset = gdal.Open( src_raster_path )
projection   = dataset.GetProjection()
geotransform = dataset.GetGeoTransform()

if projection is not None and geotransform is not None:
    dataset2 = gdal.Open( dst_path, gdal.GA_Update )
    if geotransform is not None and geotransform != (0,1,0,0,0,1):
        dataset2.SetGeoTransform( geotransform )
    if projection is not None and projection != '':
        dataset2.SetProjection( projection )
    gcp_count = dataset.GetGCPCount()
    if gcp_count != 0:
        dataset2.SetGCPs( dataset.GetGCPs(), dataset.GetGCPProjection() )
