"""
Shadow-Overlapping Building Height Estimation Tool
--------------------------------------------------

Purpose:
    This script estimates building heights from a given building footprint raster and a shadow raster using a shadow-overlapping algorithm.
    It simulates shadows at different heights for each building and compares them with the actual shadow raster to find the best match.

Inputs (User Prompts):
    1. Shadow image path.
    2. Building footprint image path.
    3. Output folder path.
    4. Output image name.
    5. Input image cell size in feet (resolution).
    6. Input image solar elevation angle (degrees).
    7. Input image solar azimuth angle (degrees).

Outputs:
    - GeoTIFF raster where pixel values represent the estimated building height (in feet, divided by 1000).
      Note: To retrieve actual height values in GIS, multiply the pixel values by 1000.

Methodology:
    1. **Pre-processing**: Reads inputs, pads images to avoid boundary issues, and converts shadow image to binary if needed.
    2. **Solar Geometry**: Calculates shadow direction based on solar azimuth and elevation.
    3. **Region Analysis**: Iterates through each connected building region in the footprint raster.
    4. **Shadow Simulation**:
        - For each building, it simulates hypothetical shadows at incremental heights (from 20ft, step 5ft).
        - Uses a boundary tracing algorithm to determine the projected shadow shape.
    5. **Matching**: Compares variation of simulated shadows with the actual shadow raster using the Jaccard Score (IoU).
    6. **Height Estimation**: Selects the height that produces the highest Jaccard score (best overlap).
    7. **Post-processing**: writes the estimated heights to a new raster, preserving the original spatial reference system.

Author:
    - Lonnie Byrnside III

References:
    - Shadow-overlapping algorithm inspired by: N. Kadhim and M. Mourshed (2018).
    - Boundary tracing algorithm adapted from: Sebastian Wallkötter (2020).
"""

import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from datetime import datetime
startTime = datetime.now()

import matplotlib.pyplot as plt
import matplotlib.cm as cmap

import numpy as np

from skimage import io
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.color import rgb2gray
from skimage.draw import line_aa, polygon_perimeter
from skimage.morphology import closing, square, erosion, dilation, rectangle
from skimage.segmentation import clear_border

from osgeo import gdal

from enum import IntEnum

from scipy import ndimage as ndi

import math

from sklearn.metrics import jaccard_score


def pol2cart(r, theta): # converts polar to cartesian coordinates, as required for pixel transformations
    x = int(r * math.cos(math.radians(theta)))
    y = int(r * math.sin(math.radians(theta)))
    return (x, y)

def find_highest_score(list_to_check):
    max_value = list_to_check[0][1]
    for entry in list_to_check:
        if entry[1] > max_value:
             max_value = entry[1]
    return max_value

def reverse_pad(arr: np.ndarray, padding: tuple):
    reversed_padding = [
        slice(start_pad, dim - end_pad)  # dim tracks dimension length
        for ((start_pad, end_pad), dim) in zip(padding, arr.shape)
    ]
    return arr[reversed_padding]

#------------------------------------------------------------------

class Directions(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


NORTH = Directions.NORTH
EAST = Directions.EAST
SOUTH = Directions.SOUTH
WEST = Directions.WEST

def trace_boundary(image):
    padded_img = np.pad(image, 1)

    img = padded_img[1:-1, 1:-1]
    img_north = padded_img[:-2, 1:-1]
    img_south = padded_img[2:, 1:-1]
    img_east = padded_img[1:-1, 2:]
    img_west = padded_img[1:-1, :-2]

    border = np.zeros((4, *padded_img.shape), dtype=np.intp)

    border[NORTH][1:-1, 1:-1] = (img == 1) & (img_north == 0)
    border[EAST][1:-1, 1:-1] = (img == 1) & (img_east == 0)
    border[SOUTH][1:-1, 1:-1] = (img == 1) & (img_south == 0)
    border[WEST][1:-1, 1:-1] = (img == 1) & (img_west == 0)

    adjacent = np.zeros((4, *image.shape), dtype=np.intp)
    adjacent[NORTH] = np.argmax(np.stack(
        (border[WEST][:-2, 2:],
         border[NORTH][1:-1, 2:],
         border[EAST][1:-1, 1:-1])
    ), axis=0)
    adjacent[EAST] = np.argmax(np.stack(
        (border[NORTH][2:, 2:],
         border[EAST][2:, 1:-1],
         border[SOUTH][1:-1, 1:-1])
    ), axis=0)
    adjacent[SOUTH] = np.argmax(np.stack(
        (border[EAST][2:, :-2],
         border[SOUTH][1:-1, :-2],
         border[WEST][1:-1, 1:-1])
    ), axis=0)
    adjacent[WEST] = np.argmax(np.stack(
        (border[SOUTH][:-2, :-2],
         border[WEST][:-2, 1:-1],
         border[NORTH][1:-1, 1:-1])
    ), axis=0)

    directions = np.zeros((len(Directions), *image.shape, 3, 3), dtype=np.intp)
    directions[NORTH][..., :] = [(3, -1, 1), (0, 0, 1), (1, 0, 0)]
    directions[EAST][..., :] = [(-1, 1, 1), (0, 1, 0), (1, 0, 0)]
    directions[SOUTH][..., :] = [(-1, 1, -1), (0, 0, -1), (1, 0, 0)]
    directions[WEST][..., :] = [(-1, -1, -1), (0, -1, 0), (-3, 0, 0)]

    proceding_edge = directions[
        np.arange(len(Directions))[:, np.newaxis, np.newaxis],
        np.arange(image.shape[0])[np.newaxis, :, np.newaxis],
        np.arange(image.shape[1])[np.newaxis, np.newaxis, :],
        adjacent
    ]

    unprocessed_border = border[:, 1:-1, 1:-1].copy()
    borders = list()
    for start_pos in zip(*np.nonzero(unprocessed_border)):
        if not unprocessed_border[start_pos]:
            continue

        idx = len(borders)
        borders.append(list())
        start_arr = np.array(start_pos, dtype=np.intp)
        current_pos = start_arr
        while True:
            unprocessed_border[tuple(current_pos)] = 0
            borders[idx].append(tuple(current_pos[1:]))
            current_pos += proceding_edge[tuple(current_pos)]
            if np.all(current_pos == np.array(start_pos)):
                break

    # match np.nonzero style output
    border_pos = list()
    for border in borders:
        border = np.array(border)
        border_pos.append([border[:, 0], border[:, 1]])

    return border_pos

#------------------------------------------------------------------
# INITIALIZE IMAGE DATA

shadow_img_path = input("Enter shadow image path: ")
footpt_img_path = input("Enter footprint image path: ")
output_img_fldr = input("Enter output folder path: ")
output_img_name = input("Enter output image name: ")
output_img_path = os.path.join(output_img_fldr, output_img_name + ".tif")

cell_size_feet = input("Enter image cell size in feet: ") #1.607612 # image resolution: 1 pixel = 1.607612 feet
elevation = input("Enter image solar elevation angle: ") # 50.01
azimuth = input("Enter image solar azimuth angle: ") # 167


imagergb_shadows = io.imread(shadow_img_path)
imagergb_shadows = rgb2gray(imagergb_shadows)
imagergb_shadows = np.pad(imagergb_shadows, pad_width=[(500, 500),(500, 500)], mode='constant')
image_true = imagergb_shadows #image_shadows > thresh_true

imagergb = io.imread(footpt_img_path)
padding = [(500, 500),(500, 500)]
imagergb = np.pad(imagergb, pad_width=padding, mode='constant')
dims = imagergb.shape

label_image = label(imagergb)

final_with_meas = np.zeros((dims[0], dims[1]), dtype=float)

region_num = 0

#------------------------------------------------------------------
# PROCESS VARIABLES

if azimuth >= 180:
    shadow_bearing = azimuth - 180
if azimuth < 180:
    shadow_bearing = azimuth + 180

if shadow_bearing <= 90:
    #quadrant = 'I'
    polar_angle = 90 - shadow_bearing
else:
    if shadow_bearing <= 180:
        #quadrant = 'IV'
        polar_angle = (180 - shadow_bearing) + 270
    else:
        if shadow_bearing <= 270:
            #quadrant = 'II'
            polar_angle = (270 - shadow_bearing) + 180
        else:
            if shadow_bearing <= 360:
                #quadrant = 'II'
                polar_angle = (360 - shadow_bearing) + 90

region_num = 0

#------------------------------------------------------------------
# RUN ANALYSIS

region_list = regionprops(label_image)
print(len(region_list))
del region_list

for region in regionprops(label_image):

    minr, minc, maxr, maxc = region.bbox

    error = False
    j_score_list = []
    test_list = []
    init_height = 20 # initial test height (in feet)
    strikes = 0

    temp_img = np.zeros((dims[0], dims[1]), dtype=float)
    coords = tuple(region.coords.T)
    temp_img[coords] = 1
    borders = trace_boundary(temp_img)
    del temp_img

    bd = borders[0]
    iterator = len(bd[0])

    for l in range(init_height, 80, 5):
        height = l / cell_size_feet # height in pixels
        length = int(height / (math.tan(math.radians(elevation))))
        
        return_img = np.zeros((dims[0], dims[1]), dtype=float)

        x, y = pol2cart(length, polar_angle)

        for i in range(iterator):
            r1 = bd[0][i]-y
            c1 = bd[1][i]+x

            if bd[0][i]-y < 0:
                #print("error!")
                error = True
                break
            if bd[0][i]-y > dims[0]:
                #print("error!")
                error = True
                break
            if bd[1][i]+x < 0:
                #print("error!")
                error = True
                break
            if bd[1][i]+x > dims[1]:
                #print("error!")
                error = True
                break
            
            rr, cc, val = line_aa(bd[0][i], bd[1][i], r1, c1)
            return_img[rr, cc] = 1

        if error:
            break
        
        return_img[coords] = 0
        
        j = jaccard_score(image_true[minr-300:maxr+300, minc-300:maxc+300], return_img[minr-300:maxr+300, minc-300:maxc+300], average='micro')
        #print(j)
        entry = [l, j]
        if j not in j_score_list:
            j_score_list.append(j)
            test_list.append(entry)
            if j != find_highest_score(test_list):
                #break
                strikes += 1
                if strikes == 1:
                    break

##        fig, ax = plt.subplots(figsize=(10, 6))
##        ax.imshow(image_true)
##
##        ax.set_axis_off()
##        plt.tight_layout()
##        plt.show()
##
##        fig, ax = plt.subplots(figsize=(10, 6))
##        ax.imshow(return_img)
##
##        ax.set_axis_off()
##        plt.tight_layout()
##        plt.show()

        del return_img

    if not error:
        if test_list != []:
            try:
                best_result = test_list[-2] # the index is the negative of (max number of strikes plus one)
            except:
                best_result = test_list[-1] # the index is the negative of (max number of strikes plus one)
            #print("Region: " + str(region_num) + ", Height: " + str(best_result[0]))
            final_with_meas[coords] = best_result[0] / 1000

    region_num += 1


print(datetime.now() - startTime)


final_with_meas = reverse_pad(final_with_meas, padding)
io.imsave(output_img_path, final_with_meas)

dataset = gdal.Open( shadow_img_path )
projection   = dataset.GetProjection()
geotransform = dataset.GetGeoTransform()

if projection is not None and geotransform is not None:
    dataset2 = gdal.Open( output_img_path, gdal.GA_Update )
    if geotransform is not None and geotransform != (0,1,0,0,0,1):
        dataset2.SetGeoTransform( geotransform )
    if projection is not None and projection != '':
        dataset2.SetProjection( projection )
    gcp_count = dataset.GetGCPCount()
    if gcp_count != 0:
        dataset2.SetGCPs( dataset.GetGCPs(), dataset.GetGCPProjection() )
