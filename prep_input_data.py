# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 08:29:51 2022

@author: maie_ja
"""

# This script is for preparation and preprocessing of the input data
# The input features for all years will be stacked and masked to lima region

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

proj_dir = "H:/Masterarbeit/population_prediction/"

# loop over years and stack all the data to retreive an array [20,7,888,888]:
seq = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']

###########################################
# read all input data and change na to 0
###########################################
## static features
# slope from Copernicus DEM
slope = io.imread(proj_dir + 'data/input_data/Slope_PSAD_deg_repr_epsg4326.tif')
slope[np.isna(slope)] = 0

# Distance to streets from OSM
streets_dist = io.imread(proj_dir + 'data/input_data/highway_distances.tif')

# Distance to water from Copernicus DEM waterbodies and OSM waterways
water_dist = io.imread(proj_dir + 'data/input_data/water_distances.tif')

# Distance to city center
center_dist = io.imread(proj_dir + 'data/input_data/city_center_distances.tif')


###########################################
# resample to pop data
###########################################
pop = io.imread(proj_dir + 'data/input_data/yearly_pop/2001_UNadj_clipped.tif')

from osgeo import gdal

# open reference file and get resolution
referenceTrans = pop.GetGeoTransform()
x_res = referenceTrans[1]
y_res = -referenceTrans[5]  # make sure this value is positive

# specify input and output filenames
inputFile = "Path to input file"
outputFile = "Path to output file"

# call gdal Warp
kwargs = {"format": "GTiff", "xRes": x_res, "yRes": y_res}
ds = gdal.Warp(outputFile, inputFile, **kwargs)

pop <- raster(paste0(path, "Population/yearly_pop/2001_UNadj_clipped.tif"))
slope_r <- resample(slope, pop, method='ngb')
streets_dist_r <- resample(streets_dist, pop, method='ngb')
water_dist_r <- resample(water_dist, pop, method='ngb')
center_dist_r <- resample(center_dist, pop, method='ngb')


for (y in seq){
print(y)
## multitemporal features
# WorldPop population grid
pop <- raster(paste0(path, "Population/yearly_pop/20",
                     y, "_UNadj_clipped.tif"))
pop[is.na(pop)] <- 0
# Distance to urban extent (derived from population grid)
urb_dist <- raster(paste0(path, "Distance_grids/Urban_extent_distances/20",
                          y, "_urban_ext_pop10_distances.tif"))
urb_dist[is.na(urb_dist)] <- 0
# Modis Land Cover
lc <- raster(paste0(path, "Land_cover/MODIS_yearly/land_cover_20", y, ".tif"))
lc[is.na(lc)] <- 999

# resample to same resolution
urb_dist_r <- resample(urb_dist, pop, method='ngb')
lc_r <- resample(lc, pop, method='ngb')

# create one raster brick
b <- brick(list(pop, urb_dist_r, lc_r, slope_r, streets_dist_r, water_dist_r, center_dist_r))


# crop to small study area
b_cropped <- crop(b,study_area)



# save raster brick
file <- writeRaster(b_cropped, filename=paste0(path, 'Code/population_prediction/data/yearly_no_na/brick_20', y, '.tif'), 
            format="GTiff", overwrite=FALSE)

}

