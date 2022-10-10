# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:59:45 2022

@author: maie_ja
"""
# raterstats: https://www.youtube.com/watch?v=VIr-pejky6E&ab_channel=GeoDeltaLabs
# xarray_spatial: https://carpentries-incubator.github.io/geospatial-python/12-zonal-statistics-raster/index.html
# rasterio: https://deepnote.com/@carlos-mendez/PYTHON-Zonal-statistics-bb1139f4-53a2-4943-982e-9cedca5dee16
# rasterize vector: https://opensourceoptions.com/blog/zonal-statistics-algorithm-with-python-in-4-steps/


import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.plot import show
import rasterstats
import matplotlib.pyplot as plt
import xarray as xr
import os
from glob import glob
import numpy as np
import rioxarray as rxr


proj_dir = "H:/Masterarbeit/population_prediction/"

# define config
config = {
        "l1": 64, #2 ** np.random.randint(2, 8), # [4, 8, 16, 32, 64, 128, 256]
        "l2": 'na', #2 ** np.random.randint(2, 8), # 'na', # 
        "lr": 0.0012, # round(np.random.uniform(0.01, 0.00001), 4), # (0.1, 0.00001)
        "batch_size": 6, #random.choice([2, 4, 6, 8]),
        "epochs": 50,
        "model_n" : '02-20_3y',
        "save" : True,
        "model": 'LSTM', # 'ConvLSTM', 'LSTM', 'BiConvLSTM', 'linear_reg', 'multivariate_reg',' 'random_forest_reg'
        "factors" : 'all' # 'all', 'static', 'pop'
    }

reg = True if config['model'] in ['linear_reg', 'multivariate_reg', 'random_forest_reg'] else False
conv = False if config['model'] in ['LSTM' , 'GRU'] else True
if conv == False: # LSTM and GRU
    config['batch_size'] = 1
    
save_path = proj_dir + 'data/test/{}_{}_{}/'.format(config['model'], config['model_n'], config['factors'])

if reg == False:
    save_path = save_path + 'lr{}_bs{}_1l{}_2l{}/'.format(config["lr"], config["batch_size"], config["l1"], config["l2"])
   
# read prediction (with projection) 
pred = rasterio.open(save_path + 'pred.tif')

# read the districts vector file
districts = gpd.read_file(proj_dir + 'data/ori_data/Lima_MA_districts/Lima_MA_districts.shp')



# plotting raster and districts together
fig, ax = plt.subplots(1,1)
show(pred, ax = ax, title = 'Prediction')
districts.plot(ax = ax, facecolor = 'None', edgecolor = 'yellow')
plt.show()



###############################################################################
# predicted population zonal stats
###############################################################################
# raster values to numpy nd array
pred_arr = pred.read(1)
affine = pred.transform # same for all the data?

# calculate zonal statistics, output is list of dicts
mean_pop = rasterstats.zonal_stats(districts, pred_arr,
                                   affine = affine,
                                   stats = ['mean'],
                                   geojson_out = True)

# extracting the mean data from the list, output is list of dicts with property info
mean_population = []
i = 0
while i < len(mean_pop):
    mean_population.append(mean_pop[i]['properties'])
    i = i+1
    
    
# Transfer info from list to pandas DataFrame
mean_pop_df = pd.DataFrame(mean_population)
mean_pop_df = mean_pop_df.loc[:,['ADM3_ES', 'mean']]
mean_pop_df.columns = ['district', 'mean_pop']
# print(mean_pop_df)


# mean predicted population per district
# mean_pop_df.plot.bar(x = 'district', y = 'mean_pop') 






###############################################################################
# yearly zonal change rate
###############################################################################
# read actual population (with projection) 
input_years = ['02', '05', '08', '11', '14', '17', '20']
pop_all_years = [] # list of arrays of population data
pop_all_years_rast = []
for i in input_years:
    # year = xr.open_rasterio(proj_dir + 'data/yearly_no_na/brick_20{}.tif'.format(i))
    year = rasterio.open(proj_dir + 'data/yearly_no_na/brick_20{}.tif'.format(i))
    # year = rxr.open_rasterio(proj_dir + 'data/yearly_no_na/brick_20{}.tif'.format(i))
    # pop_all_years_rast.append(year[0])
    year_arr = year.read(1) # read population data as array
    pop_all_years.append(year_arr)

pop_all_years.append(pred_arr)

years = input_years + ['pred']
# calculate change per pixel between years
# change_rates = []
change_zonal = []

for i in range(len(years)-1):
    y1 = years[i]
    y2 = years[i+1]
    pop1 = pop_all_years[i]
    pop2 = pop_all_years[i+1]
    # pop1[pop==0] = np.nan # to avoid dividing through 0
    change = (pop2-pop1)/pop1 # results in infinite values (if pop1 = 0)
    change[np.isinf(change)] = np.nan
    # change_rates.append(change)
    mean_change = rasterstats.zonal_stats(districts, change,
                                       affine = affine,
                                       stats = ['mean', 'median'],
                                       geojson_out = True) # outputs a list of dicts
    

    # extracting the mean data from the list
    mean_change_zonal = []
    i = 0
    while i < len(mean_change):
        mean_change_zonal.append(mean_change[i]['properties'])
        i = i+1
        
    # Transfer info from list to pandas DataFrame
    mean_change_zonal_df = pd.DataFrame(mean_change_zonal)
    mean_change_zonal_df = mean_change_zonal_df.loc[:,['ADM3_ES', 'mean']]
    mean_change_zonal_df.columns = ['district', 'mean_' + y1+y2]
    # print(mean_change_zonal_df)
    
    change_zonal.append(mean_change_zonal_df) # list of dataframes containing the districts and mean change rates



# change_zonal is a list of DataFrames containing the zonal stats for each interval
#merge all DataFrames into one
from functools import reduce
final_df = reduce(lambda  left,right: pd.merge(left,right,on=['district'],
                                            how='outer'), change_zonal)



# plotting a line plot of the change rates for each district and each time interval
plot_df = final_df.T
plot_df.columns = plot_df.iloc[0]
plot_df = plot_df.iloc[1:]

myplot = plot_df.iloc[:,40:].plot.line(legend=False, rot=90)
# myplot.set_ylim([-0.2, 0.5])
plt.legend(loc='lower left')
plt.show()


plot_all = plot_df.plot.line(legend = False, rot=90) # all districts
plot_all.set_ylim([-0.1,0.3])

# test_dist = 'Santa Maria del Mar'
# test_dist = 'San Bartolo'
# test_dist = 'Lima'
test = plot_df.iloc[:,47]
test.plot(rot=90)



# plotting districts
# fig, ax = plt.subplots(1,1)
# districts.plot(ax = ax, facecolor = 'None', edgecolor = 'yellow')
# districts[districts['ADM3_ES'] == 'Carabayllo'].plot(ax = ax, facecolor = 'red', edgecolor = 'red')
# plt.show()



###############################################################################
# yearly change for one district
###############################################################################

dist = districts[districts.ADM3_ES == 'Los Olivos']
shape = dist['geometry']

pop_all_years_crop = []
input_years = ['02', '05', '08', '11', '14', '17', '20']
for i in input_years:
    year = rasterio.open(proj_dir + 'data/yearly_no_na/brick_20{}.tif'.format(i))
    out_img, out_transf = rasterio.mask.mask(year, shape, crop= True)
    img = out_img[0]
    img[img<0] = np.nan # masked area is -3.8e+xxx
    pop_all_years_crop.append(img)


# crop and add pred
pred_crop, transf = rasterio.mask.mask(pred, shape, crop = True)
pred_crop = pred_crop[0]
pred_crop[pred_crop<0] = np.nan
pop_all_years_crop.append(pred_crop)

# calculate change rates without zonal stats
change_all = []
for i in range(len(pop_all_years_crop)-1):
    # y1 = years[i]
    # y2 = years[i+1]
    pop1 = pop_all_years_crop[i]
    pop2 = pop_all_years_crop[i+1]
    # pop1[pop==0] = np.nan # to avoid dividing through 0
    change = (pop2-pop1)/pop1 # results in infinite values (if pop1 = 0)
    change[np.isinf(change)] = np.nan
    change_all.append(change)
    


# array with row per pixel and column per year
# flatten arrays:
change_all_flat = []
for i in range(len(change_all)):
    flat = change_all[i].reshape(-1)
    change_all_flat.append(flat) # list of 1D arrays
    

nr_pixels = change_all_flat[0].shape[0]
pix_arr = np.empty((nr_pixels, len(change_all))) # empty array of shape nr pixels x nr years
for i in range(nr_pixels):
    for j in range(len(change_all_flat)):
        pix_arr[i,j] = change_all_flat[j][i] 

# remove rows with nan
pix_arr = pix_arr[~np.isnan(pix_arr).any(axis=1)]

# lineplot of changes per pixel
for i in range(1500,1550):
    plt.plot(pix_arr[i,:])

plt.xticks([0,1,2,3,4,5,6], ['0205', '0208', '0811', '1114', '1417', '1720', '20pred'])
plt.show()
 
# out_meta = year.meta
    # out_meta.update({'driver': 'GTiff',
    #                  'height': img.shape[0],
    #                  'width': img.shape[1],
    #                  'transform': out_transf})



