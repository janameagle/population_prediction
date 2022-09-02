# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:39:02 2022

@author: jmaie
"""
# data_dir = "H:/Masterarbeit/Code/population_prediction/"
proj_dir = "H:/Masterarbeit/population_prediction/"
# proj_dir = "C:/Users/jmaie/Documents/Masterarbeit/Code/population_prediction/"


import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn import metrics
from skimage.metrics import structural_similarity
import pandas as pd


# read predicted data
n_years = 20
n_classes = 4

# define config
config = {
        "l1": 64,
        "l2": 'na',
        "lr": 0.0012,
        "batch_size": 6, #1
        "epochs": 50, # 50,
        "model_n" : 'pop_02-20_3y_static_bu',
        "reg": False}

# interval = int(config['model_n'][-2])
# lastyear = 20 - interval
interval = 1
lastyear = 17

if config['reg'] == True:
    save_path = proj_dir + 'data/test/{}/'.format(config['model_n'])
    pred_path = save_path + "pred_msk_eval_rescaled.npy"
else: 
    save_path = proj_dir + 'data/test/{}_buf_bi/lr{}_bs{}_1l{}_2l{}/'.format(config["model_n"], config["lr"], config["batch_size"], config["l1"], config["l2"])
    pred_path =  save_path + "pred_msk_eval_rescaled.npy"

gt_path = proj_dir + 'data/ori_data/pop_pred/input_all_{}y_{}c_no_na_oh_buf.npy'.format(n_years, n_classes)


pred = np.load(pred_path)
gt = np.load(gt_path)




# differences between pred and gt
pop = gt[:,1,:,:]

#### set everything outsite lima region (with buffer) to 0 -> island will be removed
lima_buf = np.load(proj_dir + 'data/ori_data/pop_pred/Lima_region.npy')
pred[lima_buf == 0] = 0
#pred[pred < 0] = 0
# for i in range(20): # not needed bc input data is clipped without buffer?
#     pop_temp = pop[i,:,:]
#     pop_temp[lima_buf == 0] = 0
#     pop[i,:,:] = pop_temp


    
    
poplast = pop[lastyear-1,:,:]
pop20 = pop[-1,:,:]



difflastpred = pred - poplast
diff20pred = pred - pop20

# plt.imshow(pred)
# plt.colorbar()


# define colormaps
cmap = 'seismic' # cm.bwr
discrete = cm.tab20c
popcmap = LinearSegmentedColormap.from_list("", ["#fbf4f3", '#fbecc0', "#f9de95", "#f4b43f", "#c64912", "#340f08", "#010000"]) # cm.OrRd

top = cm.get_cmap('Reds_r', 128)
bottom = cm.get_cmap('Greens', 128)

newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
diffcmap = ListedColormap(newcolors, name='RdGn')

#diffcmap = 'RdYlGn'


###############################################################################
# spatial plot gt, pred, pred difference
###############################################################################

def spatial_plot(poplast, pop20, difflastpred, diff20pred, pred):
    # plot the differences to see prediction accuracy
    fig, axs = plt.subplots(figsize = (15, 8))
    # outer_grid = fig.add_gridspec(4, 4, wspace=0, hspace=0)
    
    ax1 = plt.subplot(241)
    plast = ax1.imshow(poplast[100:750, 0:750], cmap = popcmap, vmin = 0, vmax = 300)
    ax1.set_title("Pop 20" + str(lastyear))
    ax1.set_axis_off()
    #fig.colorbar(plast)
    
    ax2 = plt.subplot(242)
    p20 = ax2.imshow(pop20[100:750, 0:750], cmap = popcmap, vmin = 0, vmax = 300)
    ax2.set_title("Pop 2020")
    ax2.set_axis_off()
    #fig.colorbar(p20, ax = ax2)
    
    ax3 = plt.subplot(245)
    difflast = ax3.imshow(difflastpred[100:750, 0:750], cmap = diffcmap, vmin = -40, vmax = 40)
    ax3.set_title("Diff pred - 20" + str(lastyear))
    ax3.set_axis_off()
    #fig.colorbar(difflast, ax = ax3)
    
    ax4 = plt.subplot(246)
    diff20 = ax4.imshow(diff20pred[100:750, 0:750], cmap = diffcmap, vmin = -40, vmax = 40)
    ax4.set_title("Diff pred - 2020")
    ax4.set_axis_off()
    fig.colorbar(diff20, ax = [ax3, ax4], location = 'bottom')
    
    ax5 = plt.subplot(122)
    pr = ax5.imshow(pred[100:750, 0:750], cmap = popcmap, vmin = 0, vmax = 300)
    ax5.set_title("Prediction of 2020")
    ax5.set_axis_off()
    fig.colorbar(pr, ax = ax5)
    
    # add text
    if config['reg'] == True:
        fig.text(0.5, 0.12, 'Model: ' + config['model_n'],
                 verticalalignment='bottom', fontsize = 12)
    else:
        fig.text(0.5, 0.12, 'Model: {}, lr {}, bs: {}, l1: {}, l2: {}, ep: {}'.format(config["model_n"], config["lr"], config["batch_size"], config["l1"], config["l2"], config["epochs"]),
                 verticalalignment='bottom', fontsize = 12)
    
    
    # finally save to file
    plt.savefig(save_path + 'spatial_pred_check_buf.png')
    
    
    # save subplots
    difflast = plt.figure(figsize = (20,16))
    plt.imshow(difflastpred[100:750, 0:750], cmap = diffcmap, vmin = -40, vmax = 40)
    plt.title("Model: {} | Diff pred - 20{}".format(config['model_n'], str(lastyear)))
    plt.colorbar(location = 'right')
    difflast.savefig(save_path + 'spatial_pred_check_20' + str(lastyear) + '_buf.png')
    
    
    diff20 = plt.figure(figsize = (20,16))
    plt.imshow(diff20pred[100:750, 0:750], cmap = diffcmap, vmin = -40, vmax = 40)
    plt.title("Model: {} | Diff pred - 2020".format(config['model_n']))
    plt.colorbar(location = 'right')
    diff20.savefig(save_path + 'spatial_pred_check_2020_buf.png')

    pred20 = plt.figure(figsize = (20,16))
    plt.imshow(pred[100:750, 0:750], cmap = popcmap, vmin = 0, vmax = 300)
    plt.title("Prediction of 2020")
    plt.colorbar(location = 'right')
    pred20.savefig(save_path + 'prediction_2020_buf.png')



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
   

    # calculate metrics
    mae = round(metrics.mean_absolute_error(pred, pop20), 3)
    rmse = round(metrics.mean_squared_error(pred, pop20, squared = False), 3)
    r2 = round(metrics.r2_score(pred, pop20), 3)
    ssim = round(structural_similarity(pred, pop20), 3)
    
    errors = {'measure': ['mae', 'rmse', 'r2', 'ssim'],
        'value': [mae, rmse, r2, ssim]}
    
    # save in df
    errors_df = pd.DataFrame(errors)
    
    ###########################################################
    # pretty scatter plot
    fig, ax = plt.subplots(figsize = (10,7))
    plot = ax.scatter(pop20, pred, c = abs(pred - pop20), cmap = 'YlOrRd', vmin = 0, vmax = 80)
    # plot = ax.scatter(pop20, pred, color = 'Red', alpha = 0.001)
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
    if config['reg'] == True:
        fig.text(0.5, 0.095, 'Model: ' + config['model_n'])
    else:
        fig.text(0.7, 0.095, 'Model: {}, lr {}, bs: {}, l1: {}, l2: {}, ep: {}'.format(config["model_n"], config["lr"], config["batch_size"], config["l1"], config["l2"], config["epochs"]),
                 verticalalignment='bottom', horizontalalignment = 'right', 
                 fontsize = 10)
    
    fig.text(0.1, 0.72, ' R²: {} \n RMSE: {} \n MAE: {} \n SSIM: {}'.format(r2, rmse, mae, ssim), fontsize = 10)
    
    fig.tight_layout()
    
    # finally save to file
    plt.savefig(save_path + 'scatter_pred_check_buf.png')
    # plt.savefig(save_path + 'scatter_pred_check_trimmed.png')
    
    
    errors_df.to_csv(save_path + 'error_measures_buf.csv')
    # errors_df.to_csv(save_path + 'error_measures_trimmed.csv')


# import tifffile
# gt_save = proj_dir + 'data/ori_data/pop_pred/input_all_{}y_{}c_no_na_oh.tif'.format(n_years, n_classes)
# tifffile.imwrite(gt_save, gt)




###############################################################################
# do the plotting
###############################################################################

# spatial_plot(poplast, pop20, difflastpred, diff20pred, pred)

scatter_plot(pop20, pred)


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
