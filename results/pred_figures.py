# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:59:45 2022

@author: maie_ja
"""

import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show



proj_dir = "D:/Masterarbeit/population_prediction/"

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
        "factors" : 'pop', # 'all', 'static', 'pop'
        "run": 'run2'
    }

reg = True if config['model'] in ['linear_reg', 'multivariate_reg', 'random_forest_reg'] else False
conv = False if config['model'] in ['LSTM' , 'BiLSTM'] else True
if conv == False: # LSTM and GRU
    config['batch_size'] = 1
    
save_path = proj_dir + 'data/test/{}_{}_{}/'.format(config['model'], config['model_n'], config['factors'])

if reg == False:
    save_path = save_path + 'lr{}_bs{}_1l{}_2l{}/{}/'.format(config["lr"], config["batch_size"], config["l1"], config["l2"], config['run'])
   
# read prediction (with projection) 
pred = rasterio.open(save_path + 'pred.tif')
pred_arr = pred.read(1)
affine = pred.transform

# read the districts vector file
districts = gpd.read_file(proj_dir + 'data/ori_data/Lima_MA_districts/Lima_MA_districts.shp')



# plotting raster and districts together
fig, ax = plt.subplots(1,1)
img_hidden = ax.imshow(pred_arr, cmap = 'inferno')
img = show(pred, ax=ax, title = 'Prediction', cmap = 'inferno')
fig.colorbar(img_hidden, ax=ax)
# districts.plot(ax=ax, facecolor = 'None', edgecolor = 'lightgrey', linewidth = 0.3)
plt.ylim(-12.4, -11.7)
plt.show()



