# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:39:02 2022

@author: jmaie
"""
# data_dir = "H:/Masterarbeit/Code/population_prediction/"
proj_dir = "D:/Masterarbeit/population_prediction/"
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
import seaborn as sns
from scipy import stats

# change matplotlib fontsize globally
plt.rcParams['font.size'] = 20

# define config
config = {
        "l1": 64, #64, #2 ** np.random.randint(2, 8), # [4, 8, 16, 32, 64, 128, 256]
        "l2": 'na', #2 ** np.random.randint(2, 8), # 'na', # 
        "lr": 0.0012, # round(np.random.uniform(0.01, 0.00001), 4), # (0.1, 0.00001)
        "batch_size": 2, #random.choice([2, 4, 6, 8]),
        "epochs": 50,
        "model_n" : '02-20_3y',
        "save" : True,
        # "model": 'BiLSTM', # 'ConvLSTM', 'LSTM', 'BiConvLSTM', 'linear_reg', 'multivariate_reg',' 'random_forest_reg'
        "factors" : 'all', # 'all', 'static', 'pop'
        # "run" : 'run3'
    }



def main(*kwargs):
    conv = False if config['model'] in ['LSTM' , 'BiLSTM'] else True
    if conv == False: # LSTM and GRU
        config['batch_size'] = 1
    
    global reg    
    reg = True if config['model'] in ['linear_reg', 'multivariate_reg', 'random_forest_reg'] else False
    
    
    interval = int(config['model_n'][-2])
    global lastyear
    lastyear = 20 - interval
    
    global save_path
    global fig_path
    save_path = proj_dir + 'data/test/{}_{}_{}/'.format(config['model'], config['model_n'], config['factors'])
    fig_path = proj_dir + 'data/test/figures/{}_{}_{}_'.format(config['model'], config['model_n'], config['factors'])
    
    if reg == False:
        save_path = save_path + 'lr{}_bs{}_1l{}_2l{}/{}/'.format(config["lr"], config["batch_size"], config["l1"], config["l2"], config["run"])
        fig_path = fig_path + '{}_'.format(config["run"])
      
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
    global difflastpred
    difflastpred = pred - poplast
    global diff20pred
    diff20pred = pred - pop20
    global diffrate20pred
    diffrate20pred = (pred - pop20)/pop20
    # np.save(save_path + 'diff20pred.npy', diff20pred)
    
    # define colormaps
    #cmap = 'seismic' # cm.bwr
    #discrete = cm.tab20c
    #popcmap = LinearSegmentedColormap.from_list("", ["#fbf4f3", '#fbecc0', "#f9de95", "#f4b43f", "#c64912", "#340f08", "#010000"]) # cm.OrRd
    popcmap = 'inferno'
    
    # colors from red to green
    global reds
    reds = cm.get_cmap('Reds', 128)
    top = cm.get_cmap('Reds_r', 128)
    bottom = cm.get_cmap('Greens', 128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    global diffcmap
    diffcmap = ListedColormap(newcolors, name='RdGn')



    ###############################################################################
    # do the plotting
    ###############################################################################
    
    # spatial_plot(poplast, pop20, difflastpred, diff20pred, pred)
    
    # use mask for lima metropolitan area
    # scatter_plot(pop20[lima==1], pred[lima==1])
    
    density_scatter(pop20[lima==1], pred[lima==1])




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
    plt.imshow()
    # save to file
    if config["save"] == True:
        plt.savefig(save_path + 'scatter_pred_check_LMA.png')




###############################################################################
# density scatter plot
###############################################################################

from scipy.interpolate import interpn

def density_scatter( pop20 , pred, name, sort = True, bins = 100, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    
    z = np.polyfit(pop20.flatten(), pred.flatten(), 1)
    p = np.poly1d(z)
    
    # just non-zero values
    y = pred[pred>0] # .reshape(pred.shape[0]*pred.shape[1]) # [pop20>0]
    x = pop20[pred>0]  # .reshape(pop20.shape[0]*pop20.shape[1]) # [pop20>0]
    
    
    fig , ax = plt.subplots(figsize = (10,10))
    data , x_e, y_e = np.histogram2d(x, y, bins = bins, density = True )
    q = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    q[np.where(np.isnan(q))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = q.argsort()
        x, y, q = x[idx], y[idx], q[idx]

    plot = ax.scatter( x, y, c=q*100, **kwargs, vmin = 0, vmax=0.01)
    plt.plot(pop20, p(pop20), color = 'red', linewidth = 2)
    plt.plot(pop20, pop20, color = 'black', linewidth = 2)

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
    # ax.set_title('Predicted and WorldPop values 2020')
    # ax.set_ylabel('Prediction 2020')
    # ax.set_xlabel('WorldPop 2020')
    # fig.colorbar(plot, ax = ax).set_label('Point density in %')
    
    # limit axes
    ax.set_xlim(-5,350)
    ax.set_ylim(-5,370)
    
    
    # add text        
    # if config['model'] == 'random_forest_reg':
    #         n_short = 'RF'
    # elif config['model'] == 'multivariate_reg':
    #         n_short = 'linear'
    # elif config['model'] == 'linear_reg':
    #         n_short = 'linear'
    # elif config['model'] == 'BiLSTM':
    #         n_short = 'LSTM'
    # fig.text(0.5, 0.15, 'Model: ' + n_short, fontsize = 20)
    
    fig.text(0.5, 0.095, 'Model: ' + name)
    # if reg == True:
    #     fig.text(0.5, 0.095, 'Model: ' + name)
    # else:
    #     # fig.text(0.7, 0.095, 'Model: {}_{}_{}, lr {}, bs: {}, l1: {}, ep: {}'.format(config["model"], config["model_n"], config["factors"], config["lr"], config["batch_size"], config["l1"], config["epochs"]),
    #     #           verticalalignment='bottom', horizontalalignment = 'right', 
    #     #           fontsize = 10)
    #     fig.text(0.7, 0.095, 'Model: ' + name,
    #               verticalalignment='bottom', horizontalalignment = 'right', fontsize=20)
    

    
    mae, rmse, r2, r, pears_r, medae = error_measures(pred, pop20)
    fig.text(0.12, 0.67, ' MAE: {} \n MedAE: {} \n RMSE: {} \n R²: {}'.format(mae, medae, rmse, r2), fontsize = 40)

    
    fig.tight_layout()
    
    # save to file
    if config["save"] == True:
        # plt.savefig(save_path + 'scatter_density.png')
        # plt.savefig(fig_path + 'scatter_density.png')
        plt.savefig(proj_dir + 'data/test/figures/' + name + '_scatter_density.png')
    
        

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
    # errors_df.to_csv(save_path + 'error_measures_LMA.csv')
        
    return mae, rmse, r2, r, pears_r, medae



# run for all models    
all_models = ['linear_reg', 'random_forest_reg','ConvLSTM', 'BiConvLSTM', 'LSTM', 'BiLSTM'] #, 'multivariate_linear_reg', 'random_forest_reg'] #, 'BiConvLSTM' ] 
all_factors = ['pop'] #, 'static', 'pop']
all_modeln = ['02-20_3y'] #, '04-20_4y'] 
runs = ['run1', 'run2', 'run3', 'run4', 'run5']


# for m in all_models:
#     for f in all_factors:
#         for r in runs:
#             config['model'] = m
#             config['factors'] = f
#             config['run'] = r
#             main(config)


###############################################################################
# all scatter plots together
###############################################################################

# global reg    
# reg = True if config['model'] in ['linear_reg', 'multivariate_reg', 'random_forest_reg'] else False


path = proj_dir + 'data/test/'
gt = np.load(proj_dir + 'data/ori_data/input_all_unnormed.npy')
lima = np.load(proj_dir + 'data/ori_data/lima_ma.npy')
pop = gt[:,1,:,:] # all years   
pop20 = pop[-1,:,:]
pop20[lima == 0] = np.nan
    
params6 = 'lr0.0012_bs6_1l64_2lna/'
params2 = 'lr0.0012_bs2_1l64_2lna/'
paramsLSTM = 'lr0.0012_bs1_1l64_2lna/'

models = ['ConvLSTM_02-20_3y_all/' + params2 + 'run3/', 
            'ConvLSTM_02-20_3y_static/' + params2 + 'run5/',
            'ConvLSTM_02-20_3y_pop/' + params2 + 'run4/',
            'BiConvLSTM_02-20_3y_all/' + params6 + 'run2/', # bs6
            'BiConvLSTM_02-20_3y_static/' + params2 + 'run2/',    
            'BiConvLSTM_02-20_3y_pop/' + params2 + 'run2/',  
            'LSTM_02-20_3y_all/' + paramsLSTM + 'run3/',
            'LSTM_02-20_3y_static/' + paramsLSTM + 'run1/',
            'LSTM_02-20_3y_pop/' + paramsLSTM + 'run4/',
            'BiLSTM_02-20_3y_all/' + paramsLSTM + 'run5/',                       
            'BiLSTM_02-20_3y_static/' + paramsLSTM + 'run3/', 
            'BiLSTM_02-20_3y_pop/' + paramsLSTM + 'run2/',  
            'multivariate_reg_02-20_3y_all/',
            'multivariate_reg_02-20_3y_static/',
            'random_forest_reg_02-20_3y_all/',
            'random_forest_reg_02-20_3y_static/',
            'random_forest_reg_02-20_3y_pop/',
            'linear_reg_02-20_3y_pop/'
            ]

# all_models = pd.DataFrame(columns=['gt', 'model_n', 'feature'])

# i = 1
# t = len(models)
top = cm.get_cmap('Reds_r', 128)
bottom = cm.get_cmap('Greens', 128)
newcolors = np.vstack((top(np.append(np.linspace(0, 0.4, 70), np.linspace(0.4, 1, 30))),
                        bottom(np.append(np.linspace(0, 0.6, 30), np.linspace(0.6, 1, 70)))))
diffcmap = ListedColormap(newcolors, name='RdGn')
import matplotlib.ticker as ticker

for model in models:
    pred = np.load(path + model + 'pred_msk_eval_rescaled.npy')
    # mask lima region -> island will be removed
    # set negative predicted values to 0
    # pred[lima == 0] = np.nan # 0
    pred[pred < 0] = 0
    
    data = pd.DataFrame(columns=['gt', 'model_n', 'feature'])
    data['gt'] = pop20[lima==1]
    data['pred'] = pred[lima==1]
    
    model_n = model.split('/')[0]
    n_short = model_n.split('_')[0]
    if n_short == 'random':
        n_short = 'RF'
    elif n_short == 'multivariate':
        n_short = 'linear'
    
    name = n_short  + '_' + model_n.split('_')[-1]
    # data['model_n'] = n_short
    # data['feature'] = feature
    
    # values = np.vstack([data['pred'], data['gt']])
    # data['kernel'] 
    # ker = stats.gaussian_kde(values)(values)
    
    # all_models = pd.concat([all_models, data], ignore_index=True)
    
    # plt.subplot(t,t,i)
    density_scatter(pop20[lima==1], pred[lima==1], name = name)

    # diff20pred = pred - pop20    
    # fig, ax = plt.subplots(figsize = (8,8))
    # plt.imshow(diff20pred[:, 100:], cmap = diffcmap, vmin = -30, vmax = 30)
    # plt.title(None)
    # ax.xaxis.set_major_locator(ticker.NullLocator())
    # ax.yaxis.set_major_locator(ticker.NullLocator())
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # fig.text(0.2, 0.2, 'Model: ' + name, fontsize = 15)
    # plt.show()
    # fig.savefig(path + 'figures/' + name + '_pred_diff.png')

    # i += 1

# print(all_models)




# tips = sns.load_dataset("tips")
   
# g = sns.FacetGrid(all_models, row='feature', col='model_n', margin_titles=True)
# g.map_dataframe(sns.scatterplot, 'gt', 'pred', color='kernel', cmap='viridis', fit_reg=False, x_jitter=.1)
# for ax in g.axes_dict.values():
#     ax.axline((0, 0), slope=1, c='red', zorder=0)


