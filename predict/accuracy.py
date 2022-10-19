# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:39:02 2022

@author: jmaie
"""
# data_dir = "H:/Masterarbeit/Code/population_prediction/"
proj_dir = "H:/Masterarbeit/population_prediction/"
# proj_dir = "C:/Users/jmaie/Documents/Masterarbeit/Code/population_prediction/"


import numpy as np
# from skimage import io
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn import metrics
# from skimage.metrics import structural_similarity
import pandas as pd
import scipy


# define config
config = {
        "l1": 64, #64, #2 ** np.random.randint(2, 8), # [4, 8, 16, 32, 64, 128, 256]
        "l2": 'na', #2 ** np.random.randint(2, 8), # 'na', # 
        "lr": 0.0012, # round(np.random.uniform(0.01, 0.00001), 4), # (0.1, 0.00001)
        "batch_size": 6, #random.choice([2, 4, 6, 8]),
        "epochs": 50,
        "model_n" : '02-20_3y',
        "save" : True,
        "model": 'LSTM', # 'ConvLSTM', 'LSTM', 'BiConvLSTM', 'linear_reg', 'multivariate_reg',' 'random_forest_reg'
        "factors" : 'pop', # 'all', 'static', 'pop'
        "run" : 'lstmlstm'
    }

conv = False if config['model'] in ['LSTM' , 'GRU'] else True
if conv == False: # LSTM and GRU
    config['batch_size'] = 1

    
reg = True if config['model'] in ['linear_reg', 'multivariate_reg', 'random_forest_reg'] else False


interval = int(config['model_n'][-2])
lastyear = 20 - interval

save_path = proj_dir + 'data/test/{}_{}_{}/'.format(config['model'], config['model_n'], config['factors'])

if reg == False:
    save_path = save_path + 'lr{}_bs{}_1l{}_2l{}/{}/'.format(config["lr"], config["batch_size"], config["l1"], config["l2"], config["run"])
   
pred_path =  save_path + "pred_msk_eval_rescaled.npy"
gt_path = proj_dir + 'data/ori_data/input_all_unnormed.npy'

pred = np.load(pred_path)
gt = np.load(gt_path)


# mask lima region -> island will be removed
# set negative predicted values to 0
lima = np.load(proj_dir + 'data/ori_data/lima_ma.npy') # lima_ma is new lima regions
pred[lima == 0] = np.nan # 0
pred[pred < 0] = 0
    
pop = gt[:,1,:,:] # all years   
poplast = pop[lastyear-1,:,:] # last input year
poplast[lima == 0] = np.nan
pop20 = pop[-1,:,:]
pop20[lima == 0] = np.nan

# difference maps to last input year and 2020
difflastpred = pred - poplast
diff20pred = pred - pop20
diffrate20pred = (pred - pop20)/pop20
np.save(save_path + 'diff20pred.npy', diff20pred)

# define colormaps
#cmap = 'seismic' # cm.bwr
#discrete = cm.tab20c
#popcmap = LinearSegmentedColormap.from_list("", ["#fbf4f3", '#fbecc0', "#f9de95", "#f4b43f", "#c64912", "#340f08", "#010000"]) # cm.OrRd
popcmap = 'inferno'

# colors from red to green
reds = cm.get_cmap('Reds', 128)
top = cm.get_cmap('Reds_r', 128)
bottom = cm.get_cmap('Greens', 128)
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
diffcmap = ListedColormap(newcolors, name='RdGn')



###############################################################################
# spatial plot gt, pred, pred difference
###############################################################################

def spatial_plot(poplast, pop20, difflastpred, diff20pred, pred):
    # plot the differences to see prediction accuracy
    # fig, axs = plt.subplots(figsize = (15, 8))
    
    # ax1 = plt.subplot(241)
    # plast = ax1.imshow(poplast[:, 100:], cmap = popcmap, vmin = 0, vmax = 300)
    # ax1.set_title("Pop 20" + str(lastyear))
    # ax1.set_axis_off()
    
    # ax2 = plt.subplot(242)
    # p20 = ax2.imshow(pop20[:, 100:], cmap = popcmap, vmin = 0, vmax = 300)
    # ax2.set_title("Pop 2020")
    # ax2.set_axis_off()
    
    # ax3 = plt.subplot(245)
    # difflast = ax3.imshow(difflastpred[:, 100:], cmap = diffcmap, vmin = -40, vmax = 40)
    # ax3.set_title("Diff pred - 20" + str(lastyear))
    # ax3.set_axis_off()
    
    # ax4 = plt.subplot(246)
    # diff20 = ax4.imshow(diff20pred[:, 100:], cmap = diffcmap, vmin = -40, vmax = 40)
    # ax4.set_title("Diff pred - 2020")
    # ax4.set_axis_off()
    # fig.colorbar(diff20, ax = [ax3, ax4], location = 'bottom')
    
    # ax5 = plt.subplot(122)
    # pr = ax5.imshow(pred[:, 100:], cmap = popcmap, vmin = 0, vmax = 300)
    # ax5.set_title("Prediction of 2020")
    # ax5.set_axis_off()
    # fig.colorbar(pr, ax = ax5)
    
    # # add text
    # if reg == True:
    #     fig.text(0.5, 0.12, 'Model: ' + config['model_n'],
    #              verticalalignment='bottom', fontsize = 12)
    # else:
    #     fig.text(0.5, 0.12, 'Model: {}_{}_{}, lr {}, bs: {}, l1: {}, ep: {}'.format(config["model"], config["model_n"], config["factors"], config["lr"], config["batch_size"], config["l1"], config["epochs"]),
    #              verticalalignment='bottom', fontsize = 12)
    
    
    # # save to file
    # if config["save"] == True:
    #     plt.savefig(save_path + 'spatial_pred_check_LMA.png')
    
    
        # save subplots
        difflast = plt.figure(figsize = (20,16))
        plt.imshow(difflastpred[:, 100:], cmap = diffcmap, vmin = -40, vmax = 40)
        plt.title("Model: {}_{}_{} | Diff pred - 20{}".format(config["model"], config["model_n"], config["factors"], str(lastyear)))
        plt.colorbar(location = 'right')
        difflast.savefig(save_path + 'spatial_pred_check_20{}_LMA.png'.format(lastyear))
        
        
        diff20 = plt.figure(figsize = (20,16))
        plt.imshow(diff20pred[:, 100:], cmap = diffcmap, vmin = -40, vmax = 40)
        plt.title("Model: {}_{}_{} | Diff pred - 2020".format(config["model"], config["model_n"], config["factors"]))
        plt.colorbar(location = 'right')
        diff20.savefig(save_path + 'spatial_pred_check_2020_LMA.png')
    
        diffrate20 = plt.figure(figsize = (20,16))
        plt.imshow(diffrate20pred[:, 100:], cmap = reds, vmin = 0, vmax = 1)
        plt.title("Model: {}_{}_{} | Diff rate pred - 2020".format(config["model"], config["model_n"], config["factors"]))
        plt.colorbar(location = 'right')
        diff20.savefig(save_path + 'spatial_pred_check_2020_rate_LMA.png')
    
    
        # pred20 = plt.figure(figsize = (20,16))
        # plt.imshow(pred[:, 100:], cmap = popcmap, vmin = 0, vmax = 300) # change to percentile? similar to qgis?
        # plt.title("Prediction of 2020")
        # plt.colorbar(location = 'right')
        # pred20.savefig(save_path + 'prediction_2020.png')



###############################################################################
# scatter plot pred and gt
###############################################################################

def scatter_plot(pop20, pred):
    
    #calculate equation for trendline
    z = np.polyfit(pop20.flatten(), pred.flatten(), 1)
    p = np.poly1d(z)
    

####### crop image
    # pred = pred[:, 700:] #190:700]
    # pop20 = pop20[:, 700:] #190:700]
   
   
    ###########################################################
    # pretty scatter plot
    fig, ax = plt.subplots(figsize = (10,7))
    plot = ax.scatter(pop20, pred, c = abs(pred - pop20), cmap = 'YlOrRd', vmin = 0, vmax = 80)
    plt.plot(pop20, p(pop20), color = 'blue', linewidth = 0.02)
    
    # remove box around plot and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    
    # add horizontal grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    
    # labels
    ax.set_title('Scatterplot prediction and ground truth')
    ax.set_ylabel('Prediction of 2020')
    ax.set_xlabel('Ground truth of 2020')
    fig.colorbar(plot, ax = ax).set_label('Diff pred - 2020')
    
    # add text
    if reg == True:
        fig.text(0.5, 0.095, 'Model: ' + config['model_n'])
    else:
        fig.text(0.7, 0.095, 'Model: {}_{}_{}, lr {}, bs: {}, l1: {}, ep: {}'.format(config["model"], config["model_n"], config["factors"], config["lr"], config["batch_size"], config["l1"], config["epochs"]),
                 verticalalignment='bottom', horizontalalignment = 'right', 
                 fontsize = 10)
    
    
    mae, rmse, r2, r, pears_r, medae = error_measures(pred, pop20)
    fig.text(0.1, 0.72, ' R: {} \n R²: {} \n RMSE: {} \n MAE: {} \n MedAE: {}'.format(r, r2, rmse, mae, medae), fontsize = 15)
    
    fig.tight_layout()
    
    # save to file
    if config["save"] == True:
        plt.savefig(save_path + 'scatter_pred_check_LMA.png')




###############################################################################
# density scatter plot
###############################################################################

from scipy.interpolate import interpn

def density_scatter( pop20 , pred, sort = True, bins = 100, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    
    z = np.polyfit(pop20.flatten(), pred.flatten(), 1)
    p = np.poly1d(z)
    
    # just non-zero values
    y = pred[pred>0] # .reshape(pred.shape[0]*pred.shape[1]) # [pop20>0]
    x = pop20[pred>0]  # .reshape(pop20.shape[0]*pop20.shape[1]) # [pop20>0]
    
    
    fig , ax = plt.subplots(figsize = (10,7))
    data , x_e, y_e = np.histogram2d(x, y, bins = bins, density = True )
    q = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    q[np.where(np.isnan(q))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = q.argsort()
        x, y, q = x[idx], y[idx], q[idx]

    plot = ax.scatter( x, y, c=q*100, **kwargs, vmin = 0, vmax = 0.01)
    plt.plot(pop20, p(pop20), color = 'red', linewidth = 0.2)

    # remove box around plot and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    
    # add horizontal grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    
    # labels
    ax.set_title('Scatterplot prediction and ground truth')
    ax.set_ylabel('Prediction of 2020')
    ax.set_xlabel('Ground truth of 2020')
    fig.colorbar(plot, ax = ax).set_label('Point density in %')
    
    
    # add text
    if reg == True:
        fig.text(0.5, 0.095, 'Model: ' + config['model_n'])
    else:
        fig.text(0.7, 0.095, 'Model: {}_{}_{}, lr {}, bs: {}, l1: {}, ep: {}'.format(config["model"], config["model_n"], config["factors"], config["lr"], config["batch_size"], config["l1"], config["epochs"]),
                 verticalalignment='bottom', horizontalalignment = 'right', 
                 fontsize = 10)
    
    mae, rmse, r2, r, pears_r, medae = error_measures(pred, pop20)
    fig.text(0.1, 0.72, ' R: {} \n R²: {} \n RMSE: {} \n MAE: {} \n MedAE: {}'.format(r, r2, rmse, mae, medae), fontsize = 15)

    
    fig.tight_layout()

    # save to file
    if config["save"] == True:
        plt.savefig(save_path + 'scatter_density_check_LMA.png')
    
        

###############################################################################
# error measures
###############################################################################

def error_measures(pop20, pred):
    
    if len(pred.shape) == 2:
        p = pred.reshape(pred.shape[0]*pred.shape[1])
        pop = pop20.reshape(pop20.shape[0]*pop20.shape[1])
    else:
        p = pred
        pop = pop20
    
    # calculate metrics
    mae = round(metrics.mean_absolute_error(pred, pop20), 3)
    rmse = round(metrics.mean_squared_error(pred, pop20, squared = False), 3)
    r2 = round(metrics.r2_score(pred, pop20), 3)
    # ssim = round(structural_similarity(pred, pop20), 3)
    _, _, r, _, _ = scipy.stats.linregress(p, pop) # slope, intercept, rvalue, pvalue, stderr, intercept_std
    r = round(r, 3)
    pears_r, _ = scipy.stats.pearsonr(p, pop)
    medae = round(metrics.median_absolute_error(pred, pop20), 3)
    
    errors = {'measure': ['mae', 'rmse', 'r2', 'r', 'pears_r', 'medae'],
        'value': [mae, rmse, r2, r, pears_r, medae]}
    
    # save in df
    errors_df = pd.DataFrame(errors)
    errors_df.to_csv(save_path + 'error_measures_LMA.csv')
        
    return mae, rmse, r2, r, pears_r, medae

###############################################################################
# do the plotting
###############################################################################

spatial_plot(poplast, pop20, difflastpred, diff20pred, pred)

# use mask for lima metropolitan area
# scatter_plot(pop20[lima==1], pred[lima==1])

density_scatter(pop20[lima==1], pred[lima==1])




###############################################################################
# some testing
# df = error_measures(pred[pop20>0], pop20[pop20>0])
# print(df)


# pred_clean = pred.copy()
# pred_clean[pop20 == 0] = 0

# pred_b = pred.copy()
# pred_b[pred < 0] = -1
# pred_b[pred == 0] = 0
# pred_b[pred > 0] = 1
# plt.imshow(pred_b)
# plt.colorbar(location = 'right')

# pred_b = pred.copy()
# pred_b[(pop20 == 0) & (pred > 0)] = 1
# pred_b[(pop20 == 0) & (pred == 0)] = 0
# pred_b[pred < 0] = -1
# pred_b[pop20 > 0] = 2
# plt.imshow(pred_b)
# plt.colorbar(location = 'right')


# pred_b = np.zeros((pred.shape[0], pred.shape[1]))
# pred_b[(pop20 == 0) & (pred > 0)] = 1
# plt.imshow(pred_b)
# plt.colorbar(location = 'right')

# check = diff20pred
# check[pred_b >= 0] = 0
# check[check < -20] = -20

# plt.imshow(check)
# plt.colorbar(location = 'right')

# pop20notnull = pop20[pop20>10]
# prednotnull = pred[pop20>10]

# scatter_plot(pop20notnull, prednotnull)

# underestimated = np.zeros((pred.shape[0], pred.shape[1]))
# underestimated[(pred<400) & (pop20 > 400)] = 1
# overestimated = np.zeros((pred.shape[0], pred.shape[1]))
# overestimated[(pred>350 )& (pop20 < 350)] = 1
# plt.imshow(underestimated)
# plt.imshow(overestimated)
# negative = np.zeros((pred.shape[0], pred.shape[1]))
# negative[pred<0] = 1
# plt.imshow(negative)

# plt.imshow(check[300:400, :])
# plt.colorbar()
# plt.show()

# plt.scatter(pop20[320:350,770:820], pred[320:350,770:820])
# plt.scatter(pop20[320:350,800:850], pred[320:350,800:850])
