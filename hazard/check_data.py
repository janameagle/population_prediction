# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:17:31 2022

@author: maie_ja
"""
proj_dir = 'H:/Masterarbeit/population_prediction/'

import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from osgeo import gdal, gdalconst

config = {
        "l1": 64, #2 ** np.random.randint(2, 8), # [4, 8, 16, 32, 64, 128, 256]
        "l2": 'na', #2 ** np.random.randint(2, 8), # 'na', # 
        "lr": 0.0012, # round(np.random.uniform(0.01, 0.00001), 4), # (0.1, 0.00001)
        "batch_size": 6, #random.choice([2, 4, 6, 8]),
        "epochs": 50,
        "model_n" : '02-20_3y',
        "save" : True,
        "model": 'random_forest_reg', # 'ConvLSTM', 'LSTM', 'BiConvLSTM', 'linear_reg', 'multivariate_reg',' 'random_forest_reg'
        "factors" : 'pop' # 'all', 'static', 'pop'
    }

hazard_model_name = 'A'


reg = True if config['model'] in ['linear_reg', 'multivariate_reg', 'random_forest_reg'] else False
conv = False if config['model'] in ['LSTM' , 'GRU'] else True
if conv == False: # LSTM and GRU
    config['batch_size'] = 1
    
save_path = proj_dir + 'data/test/{}_{}_{}/'.format(config['model'], config['model_n'], config['factors'])
if reg == False:
    save_path = save_path + 'lr{}_bs{}_1l{}_2l{}/'.format(config["lr"], config["batch_size"], config["l1"], config["l2"])
   
# read peak ground motion
hazard_dir = proj_dir + 'data/hazard/analysis/'
# pgm = rasterio.open(hazard_dir + 'peak_ground_motion_GA_clipped.tif') # peak ground motion
# plt.imshow(pgm.read(1))
# pred = rasterio.open(save_path + 'pred.tif')
# plt.imshow(pred.read(1))




# reproject hazard to pred data with rasterio
import rasterio as rio
import numpy as np
from rasterio.warp import reproject, Resampling

hazard_model = 'peak_ground_motion_' + hazard_model_name
src = rio.open(save_path + "pred.tif")
pgm = rio.open(hazard_dir + hazard_model + '_clipped.tif')
profile = src.profile

data , affine = reproject(
        source=rio.band(pgm, 1),
        destination=np.zeros_like(src.read(1)),
        src_transform=pgm.transform,
        src_crs=pgm.crs,
        dst_transform=src.transform,
        dst_crs=src.crs,
        resampling = Resampling.nearest)


with rasterio.open(hazard_dir + hazard_model + '_rep.tif', 'w', **profile) as dst:
    dst.write(data, 1)


# change pgm values to integers (some were interpolated in nearest neighbor resampling)
with rasterio.open(hazard_dir + hazard_model + '_rep.tif') as src:
    profile = src.profile
    rast_int = src.read(1).astype(int)

with rasterio.open(hazard_dir + hazard_model + '_rep_int.tif', 'w', **profile) as rast: 
    rast.write(rast_int,  1)
    
    

# open and show reprojected data
pgm_repr = gdal.Open(hazard_dir + hazard_model + '_rep.tif', gdalconst.GA_ReadOnly)
pgm = gdal.Open(hazard_dir + hazard_model + '_clipped.tif').ReadAsArray()

# check nr of pixels, matches pred
pgm_repr.RasterXSize
pgm_repr.RasterYSize

# check values of reprojected pgm, matches original pgm (ints)
pgm_repr_arr= pgm_repr.ReadAsArray()
np.unique(pgm_repr_arr)
np.unique(pgm)

plt.imshow(pgm_repr_arr)



######################################################
# reproject max flow depth
hazard_model_name = '10m'
hazard_model = 'maximum_flow_depth_' + hazard_model_name
src = rio.open(save_path + "pred.tif")
fd = rio.open(hazard_dir + hazard_model + '_new.tif')
profile = src.profile

data, affine = reproject(
        source=rio.band(fd, 1),
        destination=np.zeros_like(src.read(1)),
        src_transform=fd.transform,
        src_crs=fd.crs,
        dst_transform=src.transform,
        dst_crs=src.crs,
        resampling = Resampling.nearest)


with rasterio.open(hazard_dir + hazard_model + '_rep.tif', 'w', **profile) as dst:
    dst.write(data, 1)


# change pgm values to integers (some were interpolated in nearest neighbor resampling)
with rasterio.open(hazard_dir + hazard_model + '_rep.tif') as src:
    profile = src.profile
    rast_int = src.read(1).astype(int)

with rasterio.open(hazard_dir + hazard_model + '_rep_int.tif', 'w', **profile) as rast: 
    rast.write(rast_int,  1)


# open and show reprojected data
fd_repr = gdal.Open(hazard_dir + hazard_model + '_rep_int.tif', gdalconst.GA_ReadOnly)
fd = gdal.Open(hazard_dir + hazard_model + '_new.tif').ReadAsArray()

# check nr of pixels, matches pred
fd_repr.RasterXSize
fd_repr.RasterYSize

# check values of reprojected fd, matches original fd (ints)
fd_repr_arr= fd_repr.ReadAsArray()
np.unique(fd_repr_arr)
np.unique(fd)

plt.imshow(fd_repr_arr)
plt.imshow(fd)


# trials
######################################################

# # open with gdal
# pgm = gdal.Open(hazard_dir + 'peak_ground_motion_GA_clipped.tif', gdalconst.GA_ReadOnly)
# pgm_arr = pgm.ReadAsArray()
# plt.imshow(pgm_arr)
# pgm.GetMetadata()
# pgm_proj = pgm.GetProjection()
# pgm_geotrans = pgm.GetGeoTransform()
# widthpgm = pgm.RasterXSize
# heightpgm = pgm.RasterYSize


# pred = gdal.Open(save_path + 'pred.tif', gdalconst.GA_ReadOnly)
# pred_arr = pred.ReadAsArray()
# plt.imshow(pred_arr)
# pred_proj = pred.GetProjection()
# pred_geotrans = pred.GetGeoTransform()
# width = pred.RasterXSize
# height = pred.RasterYSize


# # reproject hazard data to match prediction data
# output_file = hazard_dir + 'peak_ground_motion_GA_rep.tif'
# dst = gdal.GetDriverByName('GTiff').Create(output_file, width, height, 1, gdalconst.GDT_Float32)
# dst.SetGeoTransform(pred_geotrans)
# dst.SetProjection(pred_proj)

# gdal.ReprojectImage(pgm, dst, pgm_proj, pred_proj, gdalconst.GRA_Bilinear)


# # reproject pred to match hazard data
# output_file = hazard_dir + 'pred_repr.tif'
# dst = gdal.GetDriverByName('GTiff').Create(output_file, widthpgm, heightpgm, 1, gdalconst.GDT_Float32)
# dst.SetGeoTransform(pgm_geotrans)
# dst.SetProjection(pgm_proj)
# gdal.ReprojectImage(pred, dst, pred_proj, pgm_proj, gdalconst.GRA_Bilinear)


# # open and show reprojected data
# pgm_repr = gdal.Open(hazard_dir + 'peak_ground_motion_GA_repr.tif')
# pgm_repr_arr = pgm_repr.ReadAsArray()
# plt.imshow(pgm_repr_arr)


# pred_repr = gdal.Open(hazard_dir + 'pred_repr.tif')
# pred_repr_arr = pred_repr.ReadAsArray()
# plt.imshow(pred_repr_arr)

# del dst


# # https://rasterio.readthedocs.io/en/latest/topics/reproject.html
