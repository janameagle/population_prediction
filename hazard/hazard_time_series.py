# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:48:25 2022

@author: maie_ja
"""
proj_dir = 'D:/Masterarbeit/population_prediction/'

import rasterio
import matplotlib.pyplot as plt
# from rasterio.plot import show
# from osgeo import gdal, gdalconst
# from skimage import io
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import seaborn as sns
from matplotlib.ticker import FuncFormatter

# change matplotlib fontsize globally
plt.rcParams['font.size'] = 16

config = {
        "l1": 64, #2 ** np.random.randint(2, 8), # [4, 8, 16, 32, 64, 128, 256]
        "l2": 'na', #2 ** np.random.randint(2, 8), # 'na', # 
        "lr": 0.0012, # round(np.random.uniform(0.01, 0.00001), 4), # (0.1, 0.00001)
        "batch_size": 6, #random.choice([2, 4, 6, 8]),
        "epochs": 50,
        "model_n" : '02-20_3y',
        "save" : True,
        "model": 'BiLSTM', # 'ConvLSTM', 'LSTM', 'BiConvLSTM', 'linear_reg', 'multivariate_reg',' 'random_forest_reg'
        "factors" : 'pop', # 'all', 'static', 'pop',
        "run" : 'run2'
    }

reg = True if config['model'] in ['linear_reg', 'multivariate_reg', 'random_forest_reg'] else False
conv = False if config['model'] in ['LSTM' , 'BiLSTM'] else True
if conv == False: # LSTM and GRU
    config['batch_size'] = 1
    
save_path = proj_dir + 'data/test/{}_{}_{}/'.format(config['model'], config['model_n'], config['factors'])
if reg == False:
    save_path = save_path + 'lr{}_bs{}_1l{}_2l{}/{}/'.format(config["lr"], config["batch_size"], config["l1"], config["l2"], config['run'])
   
    
# actual pop
lima = np.load(proj_dir + 'data/ori_data/lima_ma.npy') # lima_ma is new lima regions
   


    
   
# read peak ground motion
hazard_dir = proj_dir + 'data/hazard/analysis/'
hazard_model = 'peak_ground_motion_M'

pgm = rasterio.open(hazard_dir + hazard_model + '_rep.tif') # peak ground motion
plt.imshow(pgm.read(1))

pred = rasterio.open(save_path + 'pred.tif')
plt.imshow(pred.read(1))


# read files as array
pgm_fl = rasterio.open(hazard_dir + hazard_model + '_rep_int.tif').read(1)
pgm = pgm_fl.astype(int) # reprojected and resampled pgm
pred = rasterio.open(save_path + 'pred.tif').read(1).astype(int)


if config['model_n'] == '02-20_2y':
    years = ['02', '04', '06', '08', '10', '12', '14', '16', '18', '20']
    pred_years = ['frc22', 'frc24', 'frc26', 'frc28']
elif config['model_n'] == '02-20_3y':
    years = ['02', '05', '08', '11', '14', '17', '20']
    pred_years = ['frc23', 'frc26', 'frc29', 'frc32', 'frc35']

# read actual population (with projection) 
pop_all_years = [] # list of arrays of population data
for i in years:
    year_arr = rasterio.open(proj_dir + 'data/yearly_no_na/brick_20{}.tif'.format(i)).read(1)
    year_arr[lima==0] = np.nan
    pop_all_years.append(year_arr.reshape(-1))


# include pred
for y in pred_years:
    pred = rasterio.open(save_path + y + '/pred.tif').read(1)
    pred[lima==0] = np.nan
    pop_all_years.append(pred.reshape(-1))

# years = years + pred_years
years =['2002', '2005', '2008', '2011', '2014', '2017', '2020',
                    '2023', '2026', '2029', '2032', '2035']

# convert to dataframe
df_years = pd.DataFrame(pop_all_years).T
df_years.columns = years
df_pgm = df_years.copy()
df_pgm['pgm'] = pgm.reshape(-1)


df_pgm_gr = df_pgm.groupby(df_pgm['pgm']).aggregate('sum')




# https://realpython.com/pandas-plot-python/
#######################################################
# barchart
#formatting the y-axis ticks
def millions(x, pos):
    #'The two args are the value and tick position'
    return '%1.fM' % (x*1e-6)
mil_format = FuncFormatter(millions)

def thousands(x, pos):
    #'The two args are the value and tick position'
    return '%1.fK' % (x*1e-3)
th_format = FuncFormatter(thousands)


import matplotlib.colors as colors
# import matplotlib
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# my_cmap = colors.ListedColormap(["#ffffff", "#ffcdcd", "#ff989b", "#ff6669", "#ff1818", "#d20000", "#a30002", "#720001", "#500002", "#670061", "#5e1a8b", "#3d0064", "#555555", "#343434", "#000000"])
# my_cmap = ["#ffcdcd", "#ff989b", "#ff6669", "#ff1818", 
#            "#d20000", "#a30002", "#720001", "#500002", "#670061", 
#            "#5e1a8b", "#3d0064", "#555555", "#343434", "#000000"]
# YlGnBu = matplotlib.cm.YlGnBu
# bounds = np.linspace(1,12,12) # [-1, 2, 5, 7, 12, 15]
# norm = matplotlib.colors.BoundaryNorm(bounds, YlGnBu.N, extend='both')
# my_cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=YlGnBu)



df_pgm_gr.plot.bar(rot=90)
# plt.yaxis.set_major_formatter(formatter)
plt.xlabel('Peak Ground Motion')
plt.ylabel('Predicted population')
plt.title('Predicted Population affected by earthquake')
plt.show()



from pylab import cm

frc_cmap = cm.get_cmap('YlOrRd', 6)    # PiYG
yr_cmap = cm.get_cmap('Blues', 12)

my_cmap = []
for i in range(yr_cmap.N): #, 0, -1):
    my_cmap.append(colors.rgb2hex(yr_cmap(i)))
for i in range(frc_cmap.N-1):
    my_cmap.append(colors.rgb2hex(frc_cmap(i+1)))


# inferno = cm.get_cmap('inferno', 13)    # PiYG
# my_cmap = []
# for i in range(7):
#     my_cmap.append(colors.rgb2hex(inferno(6-i)))
# for i in range(8, inferno.N):
#     my_cmap.append(colors.rgb2hex(inferno(i)))
    
# frc_cmap = cm.get_cmap('YlOrRd', 6)    # PiYG
# yr_cmap = cm.get_cmap('inferno', 15)

# my_cmap = []
# for i in range(7): #, 0, -1):
#     my_cmap.append(colors.rgb2hex(yr_cmap(7-i)))
# for i in range(frc_cmap.N-1):
#     my_cmap.append(colors.rgb2hex(frc_cmap(i+1))) 

fig, ax = plt.subplots(figsize=(8,6))
i=1
for col in reversed(df_pgm_gr.columns):
    ax.bar(df_pgm_gr.index, df_pgm_gr[col], bottom=0, label=col, color=my_cmap[-(i)]) #
    i = i+1
plt.xlim((195,213))
ax.yaxis.set_major_formatter(mil_format)
# plt.xticks(rotation=30)
plt.legend(fontsize=14)
plt.ylabel('Predicted population')
plt.xlabel('Peak Ground Motion')
plt.title('Predicted Population affected by earthquake')


######################################################
# analyze max flow depth
######################################################
hazard_model_name = '10m'
hazard_model = 'maximum_flow_depth_' + hazard_model_name

pred = rasterio.open(save_path + 'pred.tif')
# plt.imshow(pred.read(1))


# read files as array
fd_fl = rasterio.open(hazard_dir + hazard_model + '_rep_int.tif').read(1)
fd = fd_fl.astype(int) # reprojected and resampled pgm
pred = rasterio.open(save_path + 'pred.tif').read(1).astype(int)


# convert to dataframe
df_fd = df_years.copy()
df_fd['fd'] = fd.reshape(-1)

df_fd_gr = df_fd.groupby(df_fd['fd']).aggregate('sum')
df_fd_gr = df_fd_gr.drop(0)


# bins
bins = np.arange(0, df_fd_gr.index.values.max(), 10)
groups = df_fd_gr.groupby(np.digitize(df_fd_gr.index.values, bins)).aggregate('sum')
groups.index = bins


i=1
fig, ax = plt.subplots(figsize=(8,6))
for col in reversed(groups.columns):
    ax.bar(groups.index, groups[col], bottom=0, label=col, color=my_cmap[-i], width = 8)
    i = i+1
# plt.xlim((195,213))
# plt.xticks(rotation=30)
plt.legend(fontsize=14)
ax.yaxis.set_major_formatter(th_format)
plt.ylabel('Predicted population')
plt.xlabel('Maximum FLow Depth',)
plt.title('Predicted Population affected by tsunami')