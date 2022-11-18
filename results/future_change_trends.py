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
import rasterstats
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce


proj_dir = "D:/Masterarbeit/population_prediction/"

# define config
config = {
        "l1": 64, #2 ** np.random.randint(2, 8), # [4, 8, 16, 32, 64, 128, 256]
        "l2": 'na', #2 ** np.random.randint(2, 8), # 'na', # 
        "lr": 0.0012, # round(np.random.uniform(0.01, 0.00001), 4), # (0.1, 0.00001)
        "batch_size": 6, #random.choice([2, 4, 6, 8]),
        "epochs": 50,
        "model_n" : '02-20_2y',
        "save" : True,
        "model": 'BiLSTM', # 'ConvLSTM', 'LSTM', 'BiConvLSTM', 'linear_reg', 'multivariate_reg',' 'random_forest_reg'
        "factors" : 'pop', # 'all', 'static', 'pop'
        "run": 'run5'
    }

reg = True if config['model'] in ['linear_reg', 'multivariate_reg', 'random_forest_reg'] else False
conv = False if config['model'] in ['LSTM' , 'BiLSTM'] else True
if conv == False: # LSTM and GRU
    config['batch_size'] = 1
    
save_path = proj_dir + 'data/test/{}_{}_{}/'.format(config['model'], config['model_n'], config['factors'])

if reg == False:
    save_path = save_path + 'lr{}_bs{}_1l{}_2l{}/{}/'.format(config["lr"], config["batch_size"], config["l1"], config["l2"], config['run'])
   
# read prediction (with projection) 
pred = rasterio.open(save_path + 'frc22/'  + 'pred.tif')
pred_arr = pred.read(1)
affine = pred.transform

# read the districts vector file
districts = gpd.read_file(proj_dir + 'data/ori_data/Lima_MA_districts/Lima_MA_districts.shp')


if config['model_n'] == '02-20_2y':
    years = ['02', '04', '06', '08', '10', '12', '14', '16', '18', '20']
    pred_years = ['frc22', 'frc24', 'frc26', 'frc28']
elif config['model_n'] == '02-20_3y':
    years = ['02', '05', '08', '11', '14', '17', '20']
    pred_years = ['frc23', 'frc26', 'frc29', 'frc32', 'frc35']


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

def change_zonal(pred_years, years, absolute = True, district = 'all'):
    change_type = 'absolute change' if absolute == True else 'change rate'
    
    # read actual population (with projection) 
    pop_all_years = [] # list of arrays of population data
    for i in years:
        year = rasterio.open(proj_dir + 'data/yearly_no_na/brick_20{}.tif'.format(i))
        year_arr = year.read(1)
        pop_all_years.append(year_arr)
    
    
    # include pred
    for y in pred_years:
        pred = rasterio.open(save_path + y + '/pred.tif').read(1)
        pop_all_years.append(pred)
    
    years = years + pred_years
    
    # calculate change per pixel between years
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
        j = 0
        while j < len(mean_change):
            mean_change_zonal.append(mean_change[j]['properties'])
            j = j+1
            
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
# yearly pop
###############################################################################

def pop_zonal(pred_years, years, district = 'all'):
    
    # read actual population (with projection) 
    pop_all_years = [] # list of arrays of population data
    for i in years:
        year = rasterio.open(proj_dir + 'data/yearly_no_na/brick_20{}.tif'.format(i))
        year_arr = year.read(1)
        pop_all_years.append(year_arr)
    
    
    # include pred
    for y in pred_years:
        pred = rasterio.open(save_path + y + '/pred.tif').read(1)
        pop_all_years.append(pred)
    
    years = years + pred_years
    
    # calculate change per pixel between years
    zonal_pop= []
    for i in range(len(years)):
        year = years[i]
        mean_pop = rasterstats.zonal_stats(districts, pop_all_years[i],
                                           affine = affine,
                                           stats = ['mean', 'median'],
                                           geojson_out = True) # outputs a list of dicts
        
    
        # extracting the mean data from the list
        mean_pop_zonal = []
        j = 0
        while j < len(mean_pop):
            mean_pop_zonal.append(mean_pop[j]['properties'])
            j = j+1
            
        # Transfer info from list to pandas DataFrame
        mean_pop_zonal_df = pd.DataFrame(mean_pop_zonal)
        mean_pop_zonal_df = mean_pop_zonal_df.loc[:,['ADM3_ES', 'mean']]
        mean_pop_zonal_df.columns = ['district', 'mean_' + year]
        # print(mean_change_zonal_df)
        
        zonal_pop.append(mean_pop_zonal_df) # list of dataframes containing the districts and mean change rates
    
    
    
    # change_zonal is a list of DataFrames containing the zonal stats for each interval
    #merge all DataFrames into one
    final_df = reduce(lambda  left,right: pd.merge(left,right,on=['district'],
                                                how='outer'), zonal_pop)
    
    
    
    # plotting a line plot of the change rates for each district and each time interval
    plot_df = final_df.T
    plot_df.columns = plot_df.iloc[0] # first row as column names
    plot_df = plot_df.iloc[1:] # delete first row (column names)
    
    
    if district == 'all':
        myplot = plot_df.iloc[:,:].plot.line(legend=False, rot=90)
        myplot.set_title('Per district population')
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



out = change_zonal(pred_years, years)
out = pop_zonal(pred_years, years)