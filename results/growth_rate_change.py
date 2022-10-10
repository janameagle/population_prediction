# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 10:43:47 2022

@author: maie_ja
"""
proj_dir = "H:/Masterarbeit/population_prediction/"

import numpy as np


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
    seq_length = 5

interval = int(config['model_n'][-2])
lastyear = 20 - interval

save_path = proj_dir + 'data/test/{}_{}_{}/'.format(config['model'], config['model_n'], config['factors'])

if reg == False:
    save_path = save_path + 'lr{}_bs{}_1l{}_2l{}/'.format(config["lr"], config["batch_size"], config["l1"], config["l2"])
   
pred_path =  save_path + "pred_msk_eval_rescaled.npy"
gt_path = proj_dir + 'data/ori_data/input_all_unnormed.npy'

pred = np.load(pred_path)
gt = np.load(gt_path)


###############################################################################
# 
# lineplot: per district (mean/median) per intervall + predicted intervall(s)
# per subdistrict
# some examplary districts per model + per intervall
#
###############################################################################


###############################################################################
# 
# lineplot for testing: all pixels per intervall + predicted intervall
#
###############################################################################


###############################################################################
# 
# lineplot for testing: all pixels per intervall + predicted intervall
#
###############################################################################
