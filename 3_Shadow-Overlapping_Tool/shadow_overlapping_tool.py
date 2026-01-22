
"""
Shadow-Overlapping Building Height Estimation Tool (Refactored)
---------------------------------------------------------------
Refactored to be callable as a module and use rasterio/geopandas.
"""

import os
import warnings
import math
import numpy as np
from datetime import datetime
import rasterio
from skimage import io, measure, morphology
from skimage.measure import label, regionprops
from skimage.color import rgb2gray
from skimage.draw import line_aa
from sklearn.metrics import jaccard_score
from enum import IntEnum

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Directions(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

NORTH = Directions.NORTH
EAST = Directions.EAST
SOUTH = Directions.SOUTH
WEST = Directions.WEST

def pol2cart(r, theta):
    x = int(r * math.cos(math.radians(theta)))
    y = int(r * math.sin(math.radians(theta)))
    return (x, y)

def find_highest_score(list_to_check):
    if not list_to_check:
        return 0
    max_value = list_to_check[0][1]
    for entry in list_to_check:
        if entry[1] > max_value:
             max_value = entry[1]
    return max_value

def reverse_pad(arr: np.ndarray, padding: tuple):
    reversed_padding = [
        slice(start_pad, dim - end_pad)
        for ((start_pad, end_pad), dim) in zip(padding, arr.shape)
    ]
    return arr[tuple(reversed_padding)]

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
    # North
    adjacent[NORTH] = np.argmax(np.stack(
        (border[WEST][:-2, 2:], border[NORTH][1:-1, 2:], border[EAST][1:-1, 1:-1])
    ), axis=0)
    # East
    adjacent[EAST] = np.argmax(np.stack(
        (border[NORTH][2:, 2:], border[EAST][2:, 1:-1], border[SOUTH][1:-1, 1:-1])
    ), axis=0)
    # South
    adjacent[SOUTH] = np.argmax(np.stack(
        (border[EAST][2:, :-2], border[SOUTH][1:-1, :-2], border[WEST][1:-1, 1:-1])
    ), axis=0)
    # West
    adjacent[WEST] = np.argmax(np.stack(
        (border[SOUTH][:-2, :-2], border[WEST][:-2, 1:-1], border[NORTH][1:-1, 1:-1])
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

    border_pos = list()
    for border in borders:
        border = np.array(border)
        border_pos.append([border[:, 0], border[:, 1]])

    return border_pos

def estimate_building_heights(shadow_img_path, footpt_img_path, output_img_path, 
                              cell_size_feet=1.607612, elevation=50.01, azimuth=167,
                              min_height=20, max_height=80, step=5):
    
    startTime = datetime.now()
    print(f"Starting Height Estimation for {shadow_img_path}")
    print(f"Solar Elev: {elevation}, Azimuth: {azimuth}")

    # Read Inputs
    try:
        imagergb_shadows = io.imread(shadow_img_path)
        if len(imagergb_shadows.shape) > 2:
            imagergb_shadows = rgb2gray(imagergb_shadows)
        
        # PADDING
        pad_size = 500
        padding = [(pad_size, pad_size), (pad_size, pad_size)]
        imagergb_shadows = np.pad(imagergb_shadows, pad_width=padding, mode='constant')
        image_true = imagergb_shadows

        imagergb = io.imread(footpt_img_path)
        imagergb = np.pad(imagergb, pad_width=padding, mode='constant')
        dims = imagergb.shape
        
        label_image = label(imagergb)
        final_with_meas = np.zeros((dims[0], dims[1]), dtype=float)

        # Process Variables
        if azimuth >= 180:
            shadow_bearing = azimuth - 180
        else:
            shadow_bearing = azimuth + 180

        if shadow_bearing <= 90:
            polar_angle = 90 - shadow_bearing
        elif shadow_bearing <= 180:
            polar_angle = (180 - shadow_bearing) + 270
        elif shadow_bearing <= 270:
            polar_angle = (270 - shadow_bearing) + 180
        else:
            polar_angle = (360 - shadow_bearing) + 90

        region_list = regionprops(label_image)
        print(f"Found {len(region_list)} building regions to process.")

        # Run Analysis
        for region in region_list:
            minr, minc, maxr, maxc = region.bbox
            
            error = False
            j_score_list = []
            test_list = []
            strikes = 0

            temp_img = np.zeros((dims[0], dims[1]), dtype=float)
            coords = tuple(region.coords.T)
            temp_img[coords] = 1
            borders = trace_boundary(temp_img)
            
            if not borders:
                continue
                
            bd = borders[0]
            iterator = len(bd[0])

            # Dynamic Loop
            for l in range(min_height, max_height, step):
                height_px = l / cell_size_feet 
                length = int(height_px / (math.tan(math.radians(elevation))))
                
                return_img = np.zeros((dims[0], dims[1]), dtype=float)
                x, y = pol2cart(length, polar_angle)

                # Project Shadow
                for i in range(iterator):
                    r1 = bd[0][i]-y
                    c1 = bd[1][i]+x
                    
                    # Boundary Check
                    if r1 < 0 or r1 > dims[0] or c1 < 0 or c1 > dims[1]:
                        error = True
                        break
                    
                    rr, cc, val = line_aa(bd[0][i], bd[1][i], r1, c1)
                    # Clip coordinates to image dimensions just in case
                    rr = np.clip(rr, 0, dims[0]-1)
                    cc = np.clip(cc, 0, dims[1]-1)
                    return_img[rr, cc] = 1

                if error:
                    break
                
                # Mask out the building itself (shadows don't appear *inside* the casting building usually, 
                # or at least we care about the cast part)
                return_img[coords] = 0
                
                # Jaccard Score
                # Expand bounding box safely
                pad_check = 300
                r_start = max(0, minr - pad_check)
                r_end = min(dims[0], maxr + pad_check)
                c_start = max(0, minc - pad_check)
                c_end = min(dims[1], maxc + pad_check)

                j = jaccard_score(
                    image_true[r_start:r_end, c_start:c_end].flatten(), 
                    return_img[r_start:r_end, c_start:c_end].flatten(), 
                    average='micro'
                )
                
                entry = [l, j]
                if j not in j_score_list:
                    j_score_list.append(j)
                    test_list.append(entry)
                    # Heuristic: stop if score decreases
                    if j != find_highest_score(test_list):
                        strikes += 1
                        if strikes == 1: # Strict stopping
                            break
                
            if not error and test_list:
                try:
                    best_result = test_list[-2] 
                except:
                    best_result = test_list[-1]
                
                # Store Height
                final_with_meas[coords] = best_result[0]

        # Save Output
        print(f"Processing time: {datetime.now() - startTime}")
        final_with_meas = reverse_pad(final_with_meas, padding)
        
        # Write using Rasterio for metadata
        with rasterio.open(shadow_img_path) as src:
            profile = src.profile.copy()
            
        # Update profile for float output (heights)
        # Note: Original script divided by 1000? "pixel values represent ... in feet, divided by 1000"
        # That seems like a float storage trick. Let's just store Float32 directly if format allows.
        # But if we want to match original "divided by 1000", we should do it.
        # Actually, let's store ACTUAL height in Float32 to be accurate.
        
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
        
        with rasterio.open(output_img_path, 'w', **profile) as dst:
            dst.write(final_with_meas.astype(rasterio.float32), 1)
            
        print(f"Height map saved to {output_img_path}")

    except Exception as e:
        print(f"Error in estimate_building_heights: {e}")
        raise e

