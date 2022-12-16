# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 13:25:22 2022

@author: maie_ja
"""

# assign the correct projection (as from the input data) 
# analyze the trends per districts
# district boundaries from https://data.humdata.org/dataset/cod-ab-per


proj_dir = "D:/Masterarbeit/population_prediction/"


import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

# define config
config = {
        "l1": 64, #2 ** np.random.randint(2, 8), # [4, 8, 16, 32, 64, 128, 256]
        "l2": 'na', #2 ** np.random.randint(2, 8), # 'na', # 
        "lr": 0.0012, # round(np.random.uniform(0.01, 0.00001), 4), # (0.1, 0.00001)
        "batch_size": 6, #random.choice([2, 4, 6, 8]),
        "epochs": 50,
        "model_n" : '02-20_3y',
        "save" : True,
        "model": 'BiLSTM', # 'ConvLSTM', 'LSTM', 'BiConvLSTM', 'linear_reg', 'multivariate_reg',' 'random_forest_reg'
        "factors" : 'all', # 'all', 'static', 'pop'
        "run" : 'run5',
        "forecast": 35
    }

reg = True if config['model'] in ['linear_reg', 'multivariate_reg', 'random_forest_reg'] else False
conv = False if config['model'] in ['LSTM' , 'BiLSTM'] else True
if conv == False: # LSTM and GRU
    config['batch_size'] = 1
    
save_path = proj_dir + 'data/test/{}_{}_{}/'.format(config['model'], config['model_n'], config['factors'])

if reg == False:
    save_path = save_path + 'lr{}_bs{}_1l{}_2l{}/{}/'.format(config["lr"], config["batch_size"], config["l1"], config["l2"], config['run']) #, config['run'] , config['forecast'])
   
pred_path =  save_path + "pred_msk_eval_rescaled.npy"
diff_path = save_path + "diff20pred.npy"
# gt_path = proj_dir + 'data/ori_data/input_all_unnormed.npy'


pred_path = 'D:/Masterarbeit/population_prediction/data/0_figures/forecast_tifs/flow_depth_89.npy'
save_path = 'D:/Masterarbeit/population_prediction/data/0_figures/forecast_tifs/flow_depth_89.tif'
pred = np.load(pred_path)
# diff20pred = np.load(diff_path)


# get data with correct projection 
lima = gdal.Open(proj_dir + 'data/ori_data/Lima_MA.tif')
print(lima.GetMetadata())
GeoT = lima.GetGeoTransform() # upper-left coordinates, pixel X, Y direction size, rotation and other information
projection = lima.GetProjection()

x_pixels = lima.RasterXSize #the width of the raster data (the number of pixels in the X direction)
y_pixels = lima.RasterYSize # height of raster data (number of pixels in the Y direction)
x_min = GeoT[0]  
y_max = GeoT[3]
pixel_size = GeoT[1]


# newGeoT
x_pixels = 89
y_pixels = 89
pixel_width = (GeoT[1]*888)/x_pixels
pixel_height = (GeoT[5]*888)/y_pixels
new_GeoT = (GeoT[0], pixel_width, GeoT[2], GeoT[3], GeoT[4], pixel_height)


# assign projection 
save_name = save_path #+ 'pred.tif'
# save_name_diff = save_path + 'diff20pred.tif'
driver = gdal.GetDriverByName('GTiff')
dataset = driver.Create(
        save_name,
        x_pixels,
        y_pixels,
        1,
        gdal.GDT_Float32, )

dataset.SetGeoTransform(new_GeoT) # GeoT
# dataset.SetGeoTransform((
#         x_min,    # 0
#         pixel_size,  # 1
#         0,                      # 2
#         y_max,    # 3
#         0,                      # 4
#         -pixel_size))  

dataset.SetProjection(projection)
dataset.GetRasterBand(1).WriteArray(pred)
dataset.FlushCache()  # Write to disk.


# save diff20pred
# driver = gdal.GetDriverByName('GTiff')
# dataset = driver.Create(
#         save_name_diff,
#         x_pixels,
#         y_pixels,
#         1,
#         gdal.GDT_Float32, )

# dataset.SetGeoTransform(GeoT)
# # dataset.SetGeoTransform((
# #         x_min,    # 0
# #         pixel_size,  # 1
# #         0,                      # 2
# #         y_max,    # 3
# #         0,                      # 4
# #         -pixel_size))  

# dataset.SetProjection(projection)
# dataset.GetRasterBand(1).WriteArray(diff20pred)
# dataset.FlushCache()  # Write to disk.


# # read it back in for checking
# def read_geotiff(filename):
#     ds = gdal.Open(filename)
#     band = ds.GetRasterBand(1)
#     arr = band.ReadAsArray()
#     return arr, ds

# arr, ds = read_geotiff(save_path + 'pred.tif')
# plt.imshow(arr)
