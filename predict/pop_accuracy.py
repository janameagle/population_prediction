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


# read predicted data
specs = 'lr0.00145_bs2'
n_years = 20
n_classes = 4
pred_path = proj_dir + 'data/test/pop_pred/pop_No_seed_20y_4c_rand_srch_15-20/{}/pred_msk_eval_rescaled.npy'.format(specs)
gt_path = proj_dir + 'data/ori_data/pop_pred/input_all_{}y_{}c_no_na_oh.npy'.format(n_years, n_classes)

pred = np.load(pred_path)
gt = np.load(gt_path)


# differences between pred and gt
pop = gt[:,1,:,:]
pop16 = pop[15,:,:]
pop20 = pop[19,:,:]

diff16pred = pred - pop16
diff20pred = pred - pop20



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

def spatial_plot(pop16, pop20, diff16pred, diff20pred, pred):
    # plot the differences to see prediction accuracy
    fig, axs = plt.subplots(figsize = (15,8))
    outer_grid = fig.add_gridspec(4, 4, wspace=0, hspace=0)
    
    ax1 = plt.subplot(241)
    p16 = ax1.imshow(pop16[100:750, 0:750], cmap = popcmap, vmin = 0, vmax = 300)
    ax1.set_title("Pop 2016")
    ax1.set_axis_off()
    #fig.colorbar(p16)
    
    ax2 = plt.subplot(242)
    p20 = ax2.imshow(pop20[100:750, 0:750], cmap = popcmap, vmin = 0, vmax = 300)
    ax2.set_title("Pop 2020")
    ax2.set_axis_off()
    #fig.colorbar(p20, ax = ax2)
    
    ax3 = plt.subplot(245)
    diff16 = ax3.imshow(diff16pred[100:750, 0:750], cmap = diffcmap, vmin = -40, vmax = 40)
    ax3.set_title("Diff pred - 2016")
    ax3.set_axis_off()
    #fig.colorbar(diff16, ax = ax3)
    
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
    fig.text(0.625, 0.12, 'Model: {}, y15-20'.format(specs),
             verticalalignment='bottom', fontsize = 12)
    
    plt.show()
    
    # finally save to file
    # plt.savefig(save_path + 'Prediction_comparison')



###############################################################################
# scatter plot pred and gt
###############################################################################

def scatter_plot(pop20, pred):
    # plt change style
    #plt.style.use('seaborn')
    
    
    #calculate equation for trendline
    z = np.polyfit(pop20.flatten(), pred.flatten(), 1)
    p = np.poly1d(z)
    
    # # scatterplot with trendline
    # plt.scatter(pop20, pred, c = abs(pred - pop20), cmap = popcmap)
    # plt.title('Scatterplot prediction and ground truth')
    # plt.xlabel('Prediction of 2020')
    # plt.ylabel('Ground truth of 2020')
    # plt.colorbar().set_label('Population')
    # plt.plot(pred, pred, color = 'blue', linewidth = 0.2)
    
    
    # calculate metrics
    mae = round(metrics.mean_absolute_error(pred, pop20), 3)
    rmse = round(metrics.mean_squared_error(pred, pop20, squared = False), 3)
    r2 = round(metrics.r2_score(pred, pop20), 3)
    
    
    ###########################################################
    # pretty scatter plot
    fig, ax = plt.subplots(figsize = (10,7))
    plot = ax.scatter(pop20, pred, c = abs(pred - pop20), cmap = 'YlOrRd')
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
    fig.colorbar(plot, ax = ax).set_label('Population')
    
    # add text
    fig.text(0.8, 0.095, 'Model: {}, y15-20'.format(specs),
             verticalalignment='bottom', horizontalalignment = 'right', 
             fontsize = 10)
    fig.text(0.1, 0.72, ' R²: {} \n RMSE: {} \n MAE: {}'.format(r2, rmse, mae), fontsize = 10)
    
    fig.tight_layout()
    
    # finally save to file
    # plt.savefig(save_path + 'Prediction_comparison')
    




# # seaborn


# sns.scatterplot(pred.flatten(), pop20.flatten(), hue = pred.flatten() - pop20.flatten(), palette = popcmap)
# plt.title('test')
# plt.xlabel('xaxis')








###############################################################################
# do the plotting
###############################################################################

spatial_plot(pop16, pop20, diff16pred, diff20pred, pred)

scatter_plot(pop20, pred)















