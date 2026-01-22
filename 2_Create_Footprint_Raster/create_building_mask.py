#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np

###############################################################################
def create_building_mask(rasterSrc, vectorSrc, npDistFileName='', 
                            noDataValue=0, burn_values=1):

    """
    Creates a binary mask raster from a vector source, matching the geometry of a reference raster.
    Refactored to use rasterio and geopandas instead of osgeo.gdal/ogr.

    Args:
        rasterSrc (str): Path to the source raster file (template for size/projection).
        vectorSrc (str): Path to the vector file (SHP/GEOJSON) containing building footprints.
        npDistFileName (str): Output path for the generated mask raster.
        noDataValue (int, optional): Value to use for no-data pixels. Defaults to 0.
        burn_values (int, optional): Value to burn into the raster for polygons found in the vector source. Defaults to 1.
    """
    
    # 1. Open reference raster to get metadata
    with rasterio.open(rasterSrc) as src:
        meta = src.meta.copy()
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs

    # 2. Read vector source
    try:
        gdf = gpd.read_file(vectorSrc)
    except Exception as e:
        print(f"Error reading vector file {vectorSrc}: {e}")
        return

    # Ensure CRS matches
    if gdf.crs != crs:
        # Reproject if necessary (though usually matched in SpaceNet)
        try:
            gdf = gdf.to_crs(crs)
        except:
            print("Warning: Could not reproject vector data. Assuming matching CRS.")

    # 3. Rasterize
    if not gdf.empty:
        # Create (geometry, value) pairs
        shapes = [(geom, burn_values) for geom in gdf.geometry]
        
        mask = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=noDataValue,
            dtype=rasterio.uint8
        )
    else:
        # Empty mask if no buildings
        mask = np.full((height, width), noDataValue, dtype=np.uint8)

    # 4. Write Output
    meta.update(count=1, dtype=rasterio.uint8, nodata=noDataValue, compress='lzw')
    
    with rasterio.open(npDistFileName, 'w', **meta) as dst:
        dst.write(mask, 1)
    
    return 