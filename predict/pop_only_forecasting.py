import numpy as np
import argparse
import os
import torch
from tqdm import tqdm
from torch.autograd import Variable
import cv2
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
import pandas as pd
from model.v_convlstm import ConvLSTM
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from sklearn import metrics
import matplotlib.pyplot as plt



proj_dir = "H:/Masterarbeit/population_prediction/"
# proj_dir = "C:/Users/jmaie/Documents/Masterarbeit/Code/population_prediction/"


def get_subsample_centroids(img, img_size=50):
    h_total = img.shape[-2]
    w_total = img.shape[-1]
    h_step = int(h_total // img_size * 1.5)
    w_step = int(w_total // img_size * 1.5)
    x_list = np.linspace(img_size//2, h_total-img_size//2, num = h_step)
    y_list = np.linspace(img_size//2, w_total -img_size//2, num= w_step)

    new_x_list = []
    new_y_list = []

    for i in x_list:
        for j in y_list:
            new_x_list.append(int(i))
            new_y_list.append(int(j))
    return new_x_list, new_y_list



def get_args():
    parser = argparse.ArgumentParser(description='Train ConvLSTM Models', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-epoch', '--epoch', default=15, type=int, dest='epoch')
    parser.add_argument('-lr', '--learn_rate', default=8e-4, type=float, dest='learn_rate')
    parser.add_argument('-f', '--n_features', default=4, type=int, dest='n_features')
    parser.add_argument('-b', '--batch_size', default=12, type=int, nargs='?', help='Batch size', dest='batch_size') #5
    parser.add_argument('-n', '--n_layer', default=3, type=int, dest='n_layer')
    parser.add_argument('-l', '--seq_len', default=4, type=int, dest='seq_len')
    parser.add_argument('-is', '--input_shape', default=(256, 256), type=tuple, dest='input_shape')
    return parser.parse_args()

n_years = 20
n_classes = 4
# define config
config = {
        "l1": 64,
        "l2": 'na',
        "lr": 0.0012,
        "batch_size": 6,
        "epochs": 41, # 50
        "model_n" : 'pop_only_01_20_1y_frc22'}




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    args = get_args()

    bias_status = True

    ori_data_dir = proj_dir + 'data/ori_data/pop_pred/input_all_' + str(n_years) + 'y_' + str(n_classes) + 'c_no_na_oh_norm.npy'

    
    ori_data = np.load(ori_data_dir)# .transpose((1, 0, 2, 3))
    processed_ori_data = ori_data
    #valid_input = processed_ori_data[[3,7,11,15], 1:, :, :] # 2004-2016, 4y interval, no lc unnormed
    # valid_input = processed_ori_data[[7,10,13,16], 1:, :, :] # 2008-2017 , 3y interval, no lc unnormed
    # valid_input = processed_ori_data[[11,13,15,17], 1:, :, :] # 2012-2018 , 2y interval, no lc unnormed
    # valid_input = processed_ori_data[[15,16,17,18], 1:, :, :] # 2016-2019 , 1y interval, no lc unnormed
    if config['model_n'] == 'pop_only_01_20_1y':
        valid_input = processed_ori_data[1:-1, 1, :, :] # years 2002-2020, 1y interval
        
    elif config['model_n'] == 'pop_only_01_20_4y_frc24':
        valid_input = processed_ori_data[[7,11,15,19], 1, :, :] # years 2008-2020, 4y interval
        
    elif config['model_n'] == 'pop_only_01_20_4y_frc28':
        valid_input = processed_ori_data[[11,15,19], 1, :, :] # years 2008-2020, 4y interval
        save_path = proj_dir + "data/test/{}/lr{}_bs{}_1l{}_2l{}/".format(config["model_n"], config["lr"], config["batch_size"], config["l1"], config["l2"])
        year24 = np.load(proj_dir + "data/test/pop_only_01_20_4y_frc24/lr{}_bs{}_1l{}_2l{}/pred_msk_eval_normed.npy".format(config["lr"], config["batch_size"], config["l1"], config["l2"]))
        valid_input = np.concatenate((valid_input, year24[np.newaxis,:,:]), axis=0)
        
    elif config['model_n'] == 'pop_only_01_20_4y_frc32':
        valid_input = processed_ori_data[[15,19], 1, :, :] # years 2008-2020, 4y interval
        year24 = np.load(proj_dir + "data/test/pop_only_01_20_4y_frc24/lr{}_bs{}_1l{}_2l{}/pred_msk_eval_normed.npy".format(config["lr"], config["batch_size"], config["l1"], config["l2"]))
        year28 = np.load(proj_dir + "data/test/pop_only_01_20_4y_frc28/lr{}_bs{}_1l{}_2l{}/pred_msk_eval_normed.npy".format(config["lr"], config["batch_size"], config["l1"], config["l2"]))
        valid_input = np.concatenate((valid_input, year24[np.newaxis,:,:], year28[np.newaxis,:,:]), axis=0)
        
    elif config['model_n'] == 'pop_only_01_20_1y_frc21':
        valid_input = processed_ori_data[2:, 1, :, :] # years 2008-2020, 4y interval
        
    elif config['model_n'] == 'pop_only_01_20_1y_frc22':
        valid_input = processed_ori_data[3:, 1, :, :] # years 2008-2020, 4y interval
        save_path = proj_dir + "data/test/{}/lr{}_bs{}_1l{}_2l{}/".format(config["model_n"], config["lr"], config["batch_size"], config["l1"], config["l2"])
        year21 = np.load(proj_dir + "data/test/pop_only_01_20_1y_frc21/lr{}_bs{}_1l{}_2l{}/pred_msk_eval_normed.npy".format(config["lr"], config["batch_size"], config["l1"], config["l2"]))
        valid_input = np.concatenate((valid_input, year21[np.newaxis,:,:]), axis=0)
        
    elif config['model_n'] == 'pop_only_01_20_1y_frc23':
        valid_input = processed_ori_data[4:, 1, :, :] # years 2008-2020, 4y interval
        year21 = np.load(proj_dir + "data/test/pop_only_01_20_1y_frc21/lr{}_bs{}_1l{}_2l{}/pred_msk_eval_normed.npy".format(config["lr"], config["batch_size"], config["l1"], config["l2"]))
        year22 = np.load(proj_dir + "data/test/pop_only_01_20_1y_frc22/lr{}_bs{}_1l{}_2l{}/pred_msk_eval_normed.npy".format(config["lr"], config["batch_size"], config["l1"], config["l2"]))
        valid_input = np.concatenate((valid_input, year21[np.newaxis,:,:], year22[np.newaxis,:,:]), axis=0)
        
    
    input_channel = 1 # 19

    df = pd.DataFrame()

    dir_checkpoint = proj_dir + "data/ckpts/pop_only_01_20_1y/lr{}_bs{}_1l{}_2l{}/CP_epoch{}.pth".format(config["lr"], config["batch_size"], config["l1"], config["l2"], config["epochs"]-1)
    
    x_list, y_list = get_subsample_centroids(valid_input, img_size=256)

    sub_img_list = []
    for x, y in zip(x_list, y_list):
        sub_img = valid_input[:, x - 128:x + 128, y - 128:y + 128]
        sub_img_list.append(sub_img)

    pred_img_list = []

    with torch.no_grad():
        for test_img in tqdm(sub_img_list):
            test_img = test_img[:, np.newaxis, :,:]
            test_img = Variable(torch.from_numpy(test_img.copy())).unsqueeze(0).to(device=device,
                                                                                   dtype=torch.float32)

            net = ConvLSTM(input_dim=input_channel,
                          hidden_dim=[64, 1], # args.n_features],
                          kernel_size=(3, 3), num_layers= 2 , # num_layers= args.n_layer,
                          batch_first=True, bias=bias_status, return_all_layers=False)

            net.to(device)
            net.load_state_dict(torch.load(dir_checkpoint))
          
            output_list = net(test_img) # all factors but lc

            pred_img = output_list[0].squeeze() # t, c, w, h
            pred_img = pred_img[-1,:,:] # take last year prediction
            pred_img_list.append(pred_img.cpu().numpy())
            
           

    pred_msk = np.zeros((valid_input.shape[-2], valid_input.shape[-1]))

    h = 0
    x_list, y_list = get_subsample_centroids(valid_input, img_size=256)
    for x, y in zip(x_list, y_list):
        if x == np.min(x_list) or x == np.max(x_list) or y == np.min(y_list) or y == np.max(y_list):
            pred_msk[x - 128:x + 128, y - 128:y + 128] = pred_img_list[h]
            h += 1
        else:
            pred_msk[x - 120:x + 120, y - 120:y + 120] = pred_img_list[h][8:248,8:248]
            h += 1

    plt.imshow(pred_msk)

    # rescale to actual pop values
    ori_unnormed = np.load(proj_dir + 'data/ori_data/pop_pred/input_all_' + str(n_years) + 'y_' + str(n_classes) + 'c_no_na_oh.npy')
    pop_unnormed = ori_unnormed[:, 1, :, :]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(pop_unnormed.reshape(-1, 1))
    pop = scaler.inverse_transform(pred_msk.reshape(-1,1)).reshape(pred_msk.shape[-2], pred_msk.shape[-1])
    

    save_path = proj_dir + "data/test/{}/lr{}_bs{}_1l{}_2l{}/".format(config["model_n"], config["lr"], config["batch_size"], config["l1"], config["l2"])

    os.makedirs(save_path, exist_ok=True)
    np.save(save_path + 'pred_msk_eval_normed.npy', pred_msk)
    np.save(save_path + 'pred_msk_eval_rescaled.npy', pop)
    plt.savefig(save_path + 'pred_msk_eval.png')
    

import tifffile
tifffile.imwrite(save_path + 'pred_msk_normed.tif', pred_msk)
tifffile.imwrite(save_path + 'pred_msk_rescaled.tif', pop)

