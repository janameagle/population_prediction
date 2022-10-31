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
from functools import reduce


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
# fig, ax = plt.subplots(1,1)
# show(pred, ax = ax, title = 'Prediction')
# districts.plot(ax = ax, facecolor = 'None', edgecolor = 'yellow')
# plt.show()



###############################################################################
# predicted population zonal stats
###############################################################################

def pop_stats(pred):
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
# yearly zonal change
###############################################################################

def change_zonal(pred_arr, years = ['02', '05', '08', '11', '14', '17'], include_pred = True, 
                 absolute = True, district = 'all'):
    change_type = 'absolute change' if absolute == True else 'change rate'
    
    # read actual population (with projection) 
    pop_all_years = [] # list of arrays of population data
    pop_all_years_rast = []
    for i in years:
        # year = xr.open_rasterio(proj_dir + 'data/yearly_no_na/brick_20{}.tif'.format(i))
        year = rasterio.open(proj_dir + 'data/yearly_no_na/brick_20{}.tif'.format(i))
        # year = rxr.open_rasterio(proj_dir + 'data/yearly_no_na/brick_20{}.tif'.format(i))
        # pop_all_years_rast.append(year[0])
        year_arr = year.read(1) # read population data as array
        pop_all_years.append(year_arr)
    
    if include_pred == True:
        pop_all_years.append(pred_arr)
        years = years + ['pred']
    
    # calculate change per pixel between years
    # change_rates = []
    change_zonal = []
    
    for i in range(len(years)-1):
        y1 = years[i]
        y2 = years[i+1]
        pop1 = pop_all_years[i]
        pop2 = pop_all_years[i+1]
        if absolute == True:
            change = pop2-pop1 # absolute change
        else:
            change = (pop2-pop1)/pop1 # results in infinite values (if pop1 = 0)
            change[np.isinf(change)] = np.nan

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
    final_df = reduce(lambda  left,right: pd.merge(left,right,on=['district'],
                                                how='outer'), change_zonal)
    
    
    
    # plotting a line plot of the change rates for each district and each time interval
    plot_df = final_df.T
    plot_df.columns = plot_df.iloc[0] # first row as column names
    plot_df = plot_df.iloc[1:] # delete first row (column names)
    
    
    if district == 'all':
        myplot = plot_df.iloc[:,:].plot.line(legend=False, rot=90)
        myplot.set_title('Per district {}'.format(change_type))
        # myplot.set_ylim([-0.2, 0.5])
        # plt.legend(loc='lower left')
        plt.show()
        out = plot_df
    elif district == 'grouped':
        grouped_df = final_df.copy()
        grouped_df['trend'] = (grouped_df.mean_1417)/abs(grouped_df.mean_1417)
        group_df = grouped_df.groupby('trend').aggregate('mean')
        group_df = group_df.T
        group_df.plot(rot=90)  
        out = group_df
    else:
        dist = plot_df.loc[:,district]
        dist.plot(rot=90)
        plt.show()
        out = dist
        
    return out
    

###############################################################################
# change of one district compared with lstm and rf
###############################################################################
def mean_change_one_dist(dist = 'Lima'): 
    rf_path = proj_dir + 'data/test/random_forest_reg_02-20_3y_pop/'
    biLSTM_path = proj_dir + 'data/test/BiLSTM_02-20_3y_pop/lr0.0012_bs1_1l64_2lna/run2/'
    # read prediction (with projection) 
    rf_pred = rasterio.open(rf_path + 'pred.tif').read(1)
    biLSTM_pred = rasterio.open(biLSTM_path + 'pred.tif').read(1)

    
    rf_out = change_zonal(rf_pred, years = ['02', '05', '08', '11', '14', '17'], include_pred = True, district = dist)
    rf = pd.DataFrame(rf_out)
    if dist == 'grouped':
        rf.columns = ['rf', 'rf'] #['rf_neg', 'rf_pos']
    else:
        rf.columns = ['rf']
    
    biLSTM_out = change_zonal(biLSTM_pred, years = ['02', '05', '08', '11', '14', '17'], include_pred = True, district = dist)
    biLSTM = pd.DataFrame(biLSTM_out)
    if dist == 'grouped':
        biLSTM.columns= ['biLSTM', 'biLSTM'] # ['biLSTM_neg', 'biLSTM_pos']
    else:
        biLSTM.columns = ['biLSTM']
    
    gt_out = change_zonal(pred, years = ['02', '05', '08', '11', '14', '17', '20'], include_pred = False, district = dist)
    gt = pd.DataFrame(gt_out)
    if dist == 'grouped':
        gt.columns = ['gt', 'gt'] # ['gt_neg', 'gt_pos']
    else:
        gt.columns = ['gt']
    
    current_dist = biLSTM.join(rf)
    current_dist.rename(index={'mean_17pred':'mean_1720'},inplace=True)
    current_dist = current_dist.join(gt)
    
    myplot = current_dist.plot.line(legend = True, rot = 90).legend(loc='lower left')
    myplot.set_title('Absolute change {}'.format(dist))
    plt.show()
    
mean_change_one_dist(dist = 'grouped')

###############################################################################
# diff predicted change and actual change
###############################################################################

def change_pred_gt(years = ['17', '20'], absolute = True):  
    change_type = 'absolute change' if absolute == True else 'change rate'
                                                
    # read actual population (with projection) 
    pop_all_years = [] # list of arrays of population data
    pop_all_years_rast = []
    for i in years:
        year = rasterio.open(proj_dir + 'data/yearly_no_na/brick_20{}.tif'.format(i))
        year_arr = year.read(1) # read population data as array
        pop_all_years.append(year_arr)
    
    pop_all_years.append(pred_arr)
    years = years + ['pred']
    
    change_zonal = []
    for i in range(len(years)-1):
        y1 = years[0]
        y2 = years[i+1]
        pop1 = pop_all_years[0]
        pop2 = pop_all_years[i+1]
        if absolute == True:
            change = pop2-pop1 # absolute change
        else:
            change = (pop2-pop1)/pop1 # results in infinite values (if pop1 = 0)
            change[np.isinf(change)] = np.nan

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
        
        change_zonal.append(mean_change_zonal_df) # list of dataframes containing the districts and mean change rates
    
    final_df = reduce(lambda  left,right: pd.merge(left,right,on=['district'],
                                                how='outer'), change_zonal)
    
     
    # plotting a line plot of the change rates for each district and each time interval
    plot_df = final_df.T
    plot_df.columns = plot_df.iloc[0] # first row as column names
    plot_df = plot_df.iloc[1:] # delete first row (column names)
    
    myplot = plot_df.iloc[:,:].plot.line(legend=False, xticks=[0, 1])
    myplot.set_title('Per district {}'.format(change_type))
    myplot.set_xticklabels(plot_df.index)
    plt.show()
    
    
    # check if actual positive change is predicted positive, and actual negative change is predicted negative
    pos = final_df.loc[final_df['mean_1720'] > 0]
    neg = final_df.loc[final_df['mean_1720'] < 0]
    falseneg = pos.loc[pos['mean_17pred'] < 0]
    falsepos = neg.loc[neg['mean_17pred'] > 0]
    print(falseneg)
    print(falsepos)
    
    

###############################################################################
# yearly change for one district
###############################################################################

def change_one_dist(years = ['02', '05', '08', '11', '14', '17'], include_pred = True, 
                    absolute = True, district = 'Santa Anita', pixel_plot=80):
    change_type = 'absolute change' if absolute == True else 'change rate'
    dist = districts[districts.ADM3_ES == district]
    shape = dist['geometry']
    
    pop_all_years_crop = []
    for i in years:
        year = rasterio.open(proj_dir + 'data/yearly_no_na/brick_20{}.tif'.format(i))
        out_img, out_transf = rasterio.mask.mask(year, shape, crop = True)
        img = out_img[0]
        img[img<0] = np.nan # masked area is -3.8e+xxx
        pop_all_years_crop.append(img)
    
    
    # crop and add pred
    if include_pred == True:
        pred_crop, transf = rasterio.mask.mask(pred, shape, crop = True)
        pred_crop = pred_crop[0]
        pred_crop[pred_crop<0] = np.nan
        pop_all_years_crop.append(pred_crop)
    
    # calculate change rates without zonal stats
    change_all = []
    for i in range(len(pop_all_years_crop)-1):
        pop1 = pop_all_years_crop[i]
        pop2 = pop_all_years_crop[i+1]
        if absolute == True:
            change = pop2 - pop1 # absolute change
        else:
            change = (pop2-pop1)/pop1 # results in infinite values (if pop1 = 0)
            change[np.isinf(change)] = np.nan
        
        change_all.append(change)
        

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
    for i in range(0,pix_arr.shape[0],round(pix_arr.shape[0]/pixel_plot)): # plot 50 pixels
        plt.plot(pix_arr[i,:])
    plt.title('Per pixel {}'.format(change_type))
    plt.xticks([0,1,2,3,4,5], ['0205', '0508', '0811', '1114', '1417', '1720'])
    plt.show()


###############################################################################
# yearly change for one district
###############################################################################
out = change_zonal(pred_arr)
change_pred_gt()
change_one_dist()
mean_change_one_dist(dist='Lima')