# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:48:25 2022

@author: maie_ja
"""
proj_dir = 'D:/Masterarbeit/population_prediction/'

import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from osgeo import gdal, gdalconst
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter



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

hazard_model_name = 'M'

reg = True if config['model'] in ['linear_reg', 'multivariate_reg', 'random_forest_reg'] else False
conv = False if config['model'] in ['LSTM' , 'BiLSTM'] else True
if conv == False: # LSTM and GRU
    config['batch_size'] = 1
    
save_path = proj_dir + 'data/test/{}_{}_{}/'.format(config['model'], config['model_n'], config['factors'])
if reg == False:
    save_path = save_path + 'lr{}_bs{}_1l{}_2l{}/{}/'.format(config["lr"], config["batch_size"], config["l1"], config["l2"], config['run'])
   
    
#rf_path = proj_dir + 'data/test/{}_{}_{}/'.format('random_forest_reg', config['model_n'],config['factors'])  
lstm_path = proj_dir + 'data/test/{}_{}_{}/lr{}_bs{}_1l{}_2l{}/{}/'.format('BiLSTM', config['model_n'], 'pop',
                                                                           config["lr"], 1, config["l1"], config["l2"], 'run2')   
multiv_path = proj_dir + 'data/test/{}_{}_{}/'.format('multivariate_reg', config['model_n'], 'static') 

# for comparison of best model and base model
#rf_pred = rasterio.open(rf_path + 'pred.tif').read(1)
bilstm_pred = rasterio.open(lstm_path + 'pred.tif').read(1)
multiv_pred = rasterio.open(multiv_path + 'pred.tif').read(1)



# actual pop
lima = np.load(proj_dir + 'data/ori_data/lima_ma.npy') # lima_ma is new lima regions
gt = np.load(proj_dir + 'data/ori_data/input_all_unnormed.npy')
pop17 = gt[-4,1,:,:]  
pop17[lima == 0] = np.nan
pop20 = gt[-1,1,:,:]  
pop20[lima == 0] = np.nan
   
    
   
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
                    'pop20': pop20.reshape(-1),
                    # 'rf': rf_pred.reshape(-1),
                    'multiv': multiv_pred.reshape(-1),
                   'bilstm': bilstm_pred.reshape(-1),
                   # 'pop17': pop17.reshape(-1),
                   })

df = df_all.groupby(df_all['pgm']).aggregate('sum')


# https://realpython.com/pandas-plot-python/
#######################################################
# barchart
df.plot.bar(rot=90)

# formatting the y-axis ticks
def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.1fM' % (x*1e-6)
formatter = FuncFormatter(millions)

df.plot.bar(rot=90)
# plt.yaxis.set_major_formatter(formatter)
plt.xlabel('Peak Ground Motion')
plt.ylabel('Predicted population')
plt.title('Predicted Population affected by earthquake')
plt.show()


# fig, ax = plt.subplots(figsize=(10, 4))
# ax.bar(df.index.values, df.pred)
# plt.xticks(range(180,214), rotation='vertical')
# plt.xlim((180,214))
# ax.yaxis.set_major_formatter(formatter)
# plt.xlabel('Peak Ground Motion')
# plt.ylabel('Predicted population')
# plt.title('Predicted Population affected by earthquake')
# plt.show()


#######################################################
# difference barchart

df_diffs = pd.DataFrame({'pgm': pgm.reshape(-1),
                         'bilstm': bilstm_pred.reshape(-1) - pop20.reshape(-1),
                         # 'random forest': rf_pred.reshape(-1) - pop20.reshape(-1),
                         'multiv': multiv_pred.reshape(-1) - pop20.reshape(-1)
                         })
df_diffs = df_diffs.groupby(df_diffs['pgm']).aggregate('sum')


df_diffs.plot.bar(rot=90)
plt.xlabel('Peak Ground Motion')
plt.ylabel('Falsely predicted population')
plt.title('Falsely predicted population affected by earthquake')
plt.legend(title='Model')
plt.show()






######################################################
# analyze max flow depth
# read peak ground motion
hazard_model_name = '10m'
hazard_model = 'maximum_flow_depth_' + hazard_model_name

# fd = rasterio.open(hazard_dir + hazard_model + '_rep.tif') # peak ground motion
# plt.imshow(fd.read(1))

pred = rasterio.open(save_path + 'pred.tif')
# plt.imshow(pred.read(1))


# read files as array
fd_fl = rasterio.open(hazard_dir + hazard_model + '_rep_int.tif').read(1)
fd = fd_fl.astype(int) # reprojected and resampled pgm
pred = rasterio.open(save_path + 'pred.tif').read(1).astype(int)


# convert to dataframe
df_all = pd.DataFrame({'fd': fd.reshape(-1),
                    'pop20': pop20.reshape(-1),
                    # 'rf': rf_pred.reshape(-1),
                    'multiv': multiv_pred.reshape(-1),
                   'bilstm': bilstm_pred.reshape(-1),
                   # 'pop17': pop17.reshape(-1),
                   })

df = df_all.groupby(df_all['fd']).aggregate('sum')
df = df.drop(0)


# https://realpython.com/pandas-plot-python/
#######################################################
# # barchart
# df.plot.bar(rot=90)


# fig, ax = plt.subplots(figsize=(10, 4))
# ax.bar(df.index.values, df.pred)
# plt.xlabel('Maximum FLow Depth')
# plt.ylabel('Predicted population')
# plt.title('Predicted Population affected by tsunami')
# plt.show()


# bins
bins = np.arange(0, df.index.values.max(), 10)
groups = df.groupby(np.digitize(df.index.values, bins)).aggregate('sum')
groups.index = bins

groups.plot.bar(rot=90)
plt.xlabel('Maximum FLow Depth')
plt.ylabel('Predicted population')
plt.title('Predicted Population affected by tsunami')
plt.show()


#######################################################
# difference barchart

df_diffs_fd = pd.DataFrame({'fd': fd.reshape(-1),
                         'bilstm': bilstm_pred.reshape(-1) - pop20.reshape(-1),
                         # 'random forest': rf_pred.reshape(-1) - pop20.reshape(-1),
                         'multiv': multiv_pred.reshape(-1) - pop20.reshape(-1)
                         })
df_diffs_fd = df_diffs_fd.groupby(df_diffs_fd['fd']).aggregate('sum')
df_diffs_fd_groups = df_diffs_fd.groupby(np.digitize(df_diffs_fd.index.values, bins)).aggregate('sum')
df_diffs_fd_groups.index = bins
df_diffs_fd_groups = df_diffs_fd_groups.drop(0)


df_diffs_fd_groups.plot.bar(rot=90)
plt.xlabel('Maximum Flow Depth')
plt.ylabel('Falsely predicted population')
plt.title('Falsely predicted population affected by tsunami')
plt.legend(title='Model')
plt.show()




######################################################
# Bivariate choropleth map




# # ######################################################
# # spatial analysis
# from matplotlib import pyplot
# from rasterio.plot import show
# import earthpy.spatial as es
# import earthpy.plot as ep


# rf_pred = rasterio.open(rf_path + 'pred.tif')
# zeros = np.zeros((888,888))

# out_img = save_path + 'pgm_rf_stack.tif'
# out_meta = rf_pred.meta.copy()
# out_meta.update({'count': 3})

# # stack the pred and pop as two bands and save as file
# file_list = [pgm, pred, zeros]
# with rasterio.open(out_img, 'w', **out_meta) as dest:
#     for band_nr, src in enumerate(file_list, start=1):
#         dest.write(src, band_nr)

# stack = rasterio.open(rf_path + 'pgm_rf_stack.tif')
# img = stack.read(3)
# img[np.isnan(img)] = 0
# img_new = img # (pop, pgm, zeros)
# img_new[0,:,:] = img[1,:,:]
# img_new[1,:,:] = img[2,:,:]
# img_new[2,:,:] = img[0,:,:] # (pgm, zeros, pop)
# image_norm = (img_new - img_new.min()) / (img_new.max() - img_new.min())
# show(image_norm[2])
# show(img[0])


# # with rasterio.open(proj_dir + 'data/ori_data/pop20.tif', 'w', **out_meta) as dest:
# #     dest.write(pop20, 1)
# # with rasterio.open(proj_dir + 'data/ori_data/zeros.tif', 'w', **out_meta) as dest:
# #     dest.write(zeros, 1)
# # pop20_pth = proj_dir + 'data/ori_data/pop20.tif'
# pgm_pth = hazard_dir + hazard_model + '_rep_int.tif'
# rf_pth = rf_path + 'pred.tif'
# zeros_pth = proj_dir + 'data/ori_data/zeros.tif'

# file_list = [pgm_pth, zeros_pth, rf_pth]

# stack, meta = es.stack(file_list, nodata = None)
# # stack[:,lima==0] = np.nan
# # ep.plot_rgb(stack)
# image_norm = stack.copy()
# image_norm[0] = (stack[0] - 191) / (219 - 191)
# image_norm[2] = (stack[2] - 0) / (200 - 0)
# image_norm[:,lima==0] = np.nan
# show(image_norm)


# pgm = pgm.astype(np.float32)
# pgm[pgm==0] = np.nan
# pgm[lima==0] = np.nan
# pred = pred.astype(np.float32)
# pred[lima==0] = np.nan

# # round to 10
# pred_rounded = np.round(pred, decimals = -1)
# pred_grouped = pred_rounded.copy()
# pred_grouped[(pred_rounded%20) == 0] = pred_rounded[(pred_rounded%20 == 0)] + 10
# pgm_grouped = pgm.copy()
# pgm_grouped[(pgm%2) == 0] = pgm[(pgm%2 == 0)] + 1


# # 3 groups
# pred_gr = pred.copy()
# pred_gr[pred_gr <= 110] = 1
# pred_gr[(110 < pred_gr) & (pred_gr <= 220)] = 2
# pred_gr[220 <  pred_gr] = 3

# pgm_gr = pgm.copy()
# pgm_gr[pgm_gr <= 200] = 10
# pgm_gr[(200 < pgm_gr) & (pgm_gr <= 205)] = 20
# pgm_gr[205 <  pgm_gr] = 30

# final = pgm_gr  + pred_gr

# # define colors
# from matplotlib.colors import ListedColormap
# import matplotlib.colors as colors
# cmap = ListedColormap(['#e8e8e8', '#e4acac', '#c85a5a', 
#                        '#b0d5df', '#ad9ea5', '#985356',
#                        '#64acbe', '#627f8c', '#574249'])
# norm = colors.BoundaryNorm([11, 12, 13, 21, 22, 23, 31, 32, 33], 9)
# plt.imshow(final, cmap=cmap, norm=norm)




# fig, ax = plt.subplots()
# ax.imshow(pgm_grouped, alpha=1, vmin=200, vmax=211,  cmap='Blues')
# ax.imshow(pred_grouped, alpha=0.5, vmin=0, vmax=341, cmap='Reds')
# plt.show()



# # vectorize the rasters for bivariate choropleth map
# from osgeo import ogr
# from osgeo import gdal
# output = save_path + 'pred_vectorized.shp'
# shp_driver = ogr.GetDriverByName('ESRI Shapefile')

# output_shapefile = shp_driver.CreateDataSource(output)
# new_shapefile = output_shapefile.CreateLayer(output, srs = None)

# pred = gdal.Open(save_path + 'pred.tif').GetRasterBand(1)
# gdal.Polygonize(pred, None, new_shapefile, -1, [], callback=None)
# new_shapefile.SyncToDisk()

# from rasterio import features
# from shapely.geometry import shape
# from shapely.geometry import Polygon

# shp = rasterio.features.shapes(pred)
# shp_df = pd.DataFrame()
# for poly in shp:
#     df = pd.DataFrame(poly[0])
#     df['value'] = poly[1]
#     print(df)
#     shp_df = pd.concat([shp_df, df], ignore_index = True)

# shp_df_poly = shp_df.copy()
# for i in range(len(shp_df_poly)):
#     shp_df_poly.coordinates[i] = Polygon(shp_df_poly.coordinates[i])


# # save as csv
# shp_df_poly.to_csv(save_path + 'pred_shapes.csv')


# print(shp_df_poly)
# print(type(shp_df_poly))
# shp_df_poly.plot()

# #convert to geodataframe
# import geopandas as gpd
# shp_gdf = gpd.GeoDataFrame(shp_df_poly, geometry = shp_df_poly.coordinates)
# print(shp_gdf)
# shp_gdf.plot('value')

# import plotly.express as px
# px.choropleth(shp_gdf, geojson=shp_gdf.geometry, locations=shp_gdf.index, color='value')



# # extract median pgm value per polygon
# mean = pd.DataFrame(
#     zonal_stats(
#         vectors=shp_gdf['geometry'], 
#         raster=pgm, 
#         stats='mean'
#     )
# )['mean']


# #https://gis.stackexchange.com/questions/384581/raster-to-geopandas
# import rasterio as rio
# with rio.open(save_path + 'pred.tif') as src:
#     crs = src.crs
    
#     # 1D coordinate arrays (pixel center)
#     xmin, ymax = src.xy(0.00, 0.00)
#     xmax, ymin = src.xy(src.height-1, src.width-1)
#     x = np.linspace(xmin, xmax, src.width)
#     y = np.linspace(ymax, ymin, src.height)  # max -> min so coords are top -> bottom

#     # create 2D arrays
#     xs, ys = np.meshgrid(x, y)
#     zs = src.read(1)

#     # Apply NoData mask
#     mask = src.read_masks(1) > 0
#     xs, ys, zs = xs[mask], ys[mask], zs[mask]


# data = {"X": pd.Series(xs.ravel()),
#         "Y": pd.Series(ys.ravel()),
#         "Z": pd.Series(zs.ravel())}

# df = pd.DataFrame(data=data)
# geometry = gpd.points_from_xy(df.X, df.Y)
# gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

# print(gdf.head())
# gdf.plot(column='Z')

# # px.choropleth(gdf, geojson=gdf.geometry, locations=gdf.index, color='Z')


# pgm = rio.open(hazard_dir + hazard_model + '_rep_int.tif').read(1)
# # extract median pgm value per polygon
# from rasterstats import zonal_stats
# median = pd.DataFrame(
#     zonal_stats(
#         vectors=gdf['geometry'], 
#         raster=hazard_dir + hazard_model + '_rep.tif', 
#         stats='median'
#     )
# )['median']

# print(median)


# with rio.open(save_path + 'pred.tif') as src:
#     zs = src.read(1)
#     x = np.linspace(0, zs.shape[0], zs.shape[0]+1)
#     y = np.linspace(zs.shape[1], 0, zs.shape[1]+1)
    
#     # create 2D arrays
#     xs, ys = np.meshgrid(x, y)


# data = {"X": pd.Series(xs.ravel()),
#         "Y": pd.Series(ys.ravel()),
#         "Z": pd.Series(zs.ravel())}

# df = pd.DataFrame(data=data)
# geometry = gpd.points_from_xy(df.X, df.Y)
# gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

# print(gdf.head())
# gdf.plot(column='Z', kind='geo')


# #extract value from pgm
# pgm = rio.open(hazard_dir + hazard_model + '_rep_int.tif').read(1)
# gdf['pgm'] = np.zeros(len(gdf))
# for i in range(len(gdf)):
#     x = int(gdf['geometry'][i].xy[0][0])
#     y = int(gdf['geometry'][i].xy[1][0])
#     gdf['pgm'][i] = pgm[x-1,y-1]
    
    
# # bivariate choropleth map
# from splot.mapping import vba_choropleth
# a = gdf['Z'].values
# b = gdf['pgm'].values
 

# fig, axs = plt.subplots(1,1, figsize=(10,10))
# # use vba_choropleth to create Value-by-Alpha Choropleth
# vba_choropleth(a, b, gdf, rgb_mapclassify=dict(classifier='quantiles'),
#                alpha_mapclassify=dict(classifier='quantiles'),
#                cmap='RdBu', legend=True, ax=axs[1])

# plt.show()








