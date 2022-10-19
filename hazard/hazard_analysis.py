# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:48:25 2022

@author: maie_ja
"""
proj_dir = 'H:/Masterarbeit/population_prediction/'

import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from osgeo import gdal, gdalconst
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



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
hazard_model = 'peak_ground_motion_' + hazard_model_name

pgm = rasterio.open(hazard_dir + hazard_model + '_rep.tif') # peak ground motion
plt.imshow(pgm.read(1))

pred = rasterio.open(save_path + 'pred.tif')
plt.imshow(pred.read(1))


# read files as array
pgm_fl = rasterio.open(hazard_dir + hazard_model + '_rep_int.tif').read(1)
pgm = pgm_fl.astype(int) # reprojected and resampled pgm
pred = rasterio.open(save_path + 'pred.tif').read(1).astype(int)



# convert to dataframe
df_all = pd.DataFrame({'pgm': pgm.reshape(-1),
                   'pred': pred.reshape(-1)})

df = df_all.groupby(df_all['pgm']).aggregate('sum')


# https://realpython.com/pandas-plot-python/
#######################################################
# barchart
df.plot.bar(rot=90)


fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(df.index.values, df.pred)
plt.xticks(range(180,214), rotation='vertical')
plt.xlim((180,214))
plt.xlabel('Peak Ground Motion')
plt.ylabel('Predicted population')
plt.title('Predicted Population affected by earthquake')
plt.show()


# pgm_fl[pgm_fl==0] = np.nan
# plt.imshow(pgm_fl)



######################################################
# analyze max flow depth
# read peak ground motion
hazard_model_name = '100m'
hazard_model = 'maximum_flow_depth_' + hazard_model_name

fd = rasterio.open(hazard_dir + hazard_model + '_rep.tif') # peak ground motion
plt.imshow(fd.read(1))

pred = rasterio.open(save_path + 'pred.tif')
plt.imshow(pred.read(1))


# read files as array
fd_fl = rasterio.open(hazard_dir + hazard_model + '_rep_int.tif').read(1)
fd = fd_fl.astype(int) # reprojected and resampled pgm
pred = rasterio.open(save_path + 'pred.tif').read(1).astype(int)



# convert to dataframe
df_all = pd.DataFrame({'pgm': fd.reshape(-1),
                   'pred': pred.reshape(-1)})

df = df_all.groupby(df_all['pgm']).aggregate('sum')
df = df.drop(0)


# https://realpython.com/pandas-plot-python/
#######################################################
# barchart
df.plot.bar(rot=90)


fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(df.index.values, df.pred)
# plt.xticks(range(180,214), rotation='vertical')
# plt.xlim((0,10))
plt.xlabel('Maximum FLow Depth')
plt.ylabel('Predicted population')
plt.title('Predicted Population affected by tsunami')
plt.show()







