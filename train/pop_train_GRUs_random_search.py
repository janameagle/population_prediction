# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 12:42:32 2022

@author: jmaie
"""


from model.v_convlstm import ConvLSTM
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import logging
from torch.autograd import Variable
from torch.optim import lr_scheduler
import pandas as pd
from utilis.dataset import MyDataset
# from utilis.dataset import min_max_scale
from train.options import get_args
from utilis.weight_init import weight_init
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
import numpy as np
from livelossplot import PlotLosses # https://github.com/stared/livelossplot/blob/master/examples/pytorch.ipynb
from GPUtil import showUtilization as gpu_usage
import random
import matplotlib.pyplot as plt


print("initial usage")
gpu_usage()

proj_dir = "H:/Masterarbeit/population_prediction/"
# proj_dir = "C:/Users/jmaie/Documents/Masterarbeit/Code/population_prediction/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# def pre_prcessing(crop_img): # why?
#     crop_img_lulc = torch.from_numpy(crop_img[:, 0, :, :]) # was not converted to torch before, select lc
#     temp_list = []
#     for j in range(crop_img_lulc.shape[0]): # for each year?
#         temp = oh_code(crop_img_lulc[j], class_n=7) # array of binary mask per class
#         temp_list.append(temp[np.newaxis, :, :, :]) # store class masks per year in list
#     oh_crop_img_lulc = np.concatenate(temp_list, axis=0)
#     oh_crop_img = np.concatenate((oh_crop_img_lulc, crop_img[:, 1:, :, :]), axis=1)
#     return oh_crop_img

def evaluate(pred, gt):
    k_statistics = cohen_kappa_score(gt.astype(np.int64).flatten(), pred.astype(np.int64).flatten())
    acc_score = accuracy_score(gt.astype(np.int64).flatten(), pred.astype(np.int64).flatten())

    return k_statistics,acc_score

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

# def oh_code(a, class_n = 7):
#     oh_list = []
#     for i in range(class_n): # for each class
#         temp = torch.where(a == i, 1, 0) # binary mask per class
#         oh_list.append(temp) # store each class mask as list entry
#     return torch.stack(oh_list,0) #torch.stack(oh_list,1) # return array, not list

def get_valid_dataset(ori_data_dir):
    ori_data = np.load(ori_data_dir)#.transpose((1, 0, 2, 3))
    # scaled_data = torch.nn.functional.interpolate(torch.from_numpy(ori_data),
    #                                               scale_factor=(1 / 3, 1 / 3),
    #                                               recompute_scale_factor=True)
    # processed_ori_data = scaled_data.numpy()
    processed_ori_data = ori_data
    valid_input = processed_ori_data[-5:-1, :, :, :] # years 2016 - 2019
    gt = processed_ori_data[-1, 1, :, :] # last year, pop
    return valid_input, gt

def get_valid_record(valid_input, gt, net, device = device, factor_option = 'with_factors'): #torch.device('cuda')
    x_list, y_list = get_subsample_centroids(valid_input, img_size=256)
    sub_img_list = []
    for x, y in zip(x_list, y_list):
        sub_img = valid_input[:, :, x - 128:x + 128, y - 128:y + 128]
        sub_img_list.append(sub_img)

    pred_img_list = []
    with torch.no_grad():
        for test_img in sub_img_list:
            # test_img = pre_prcessing(test_img)
            # test_imag = Variable(...)
            test_img = Variable(torch.from_numpy(test_img.copy())).unsqueeze(0).to(device=device,
                                                                                   dtype=torch.float32)

            output_list = net(test_img[:, :, 1:, :, :]) # all except lc
            masks_pred = output_list[0].squeeze()
            # lin = nn.Linear(1,1)
            # pred = lin(masks_pred.permute(0,1,3,4,2)).permute(0,1,4,2,3)
            # pred.to(device)

            # pred_img = pred.squeeze()
            pred_img = masks_pred
            criterion = nn.MSELoss() # no crossentropyloss for regression
            loss = criterion(pred_img.float(), test_img[:,:,1,:,:].squeeze().float()) # for validation loss
           

            
            
            # pred_img = np.squeeze(np.argmax(pred_prob, axis=1)[-1:, :, :])
            pred_img_list.append(pred_img[-1,:,:].cpu().numpy())
    pred_msk = np.zeros((valid_input.shape[-2], valid_input.shape[-1]))

    h = 0

    x_list, y_list = get_subsample_centroids(valid_input, img_size=256)
    for x, y in zip(x_list, y_list):
        if x == np.min(x_list) or x == np.max(x_list) or y == np.min(y_list) or y == np.max(y_list):
            pred_msk[x - 128:x + 128, y - 128:y + 128] = pred_img_list[h]
            h += 1
        else:
            pred_msk[x - 120:x + 120, y - 120:y + 120] = pred_img_list[h][8:248, 8:248]
            h += 1

    k, acc = evaluate(gt, pred_msk)

    return k, acc, loss.item()




# from train_all script
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
args = get_args()

# pred_sequence_list = ['forecasting'] #'backcasting'                        # what is backcasting? why do it?

bias_status = True #False                                          # ?
beta = 0                                                           # ?

input_channel = 10                                            # 19 driving factors
factor = 'with_factors'
pred_sequence = 'forward'

# epochs = 15
# model_n = 'No_seed_convLSTM_no_na_random_search'

# net = ConvLSTM(input_dim=input_channel,
#                hidden_dim=[16, args.n_features], # hidden_dim = [32, 16, args.n_features]
#                kernel_size=(3, 3), num_layers = 2, # num_layers=args.n_layer,
#                batch_first=True, bias=bias_status, return_all_layers=False)
# net.to(device)

# train_ConvGRU_FullValid(net=net, device=device,
#                epochs=5, batch_size=args.batch_size, lr=args.learn_rate,
#                save_cp=True, save_csv=True, factor_option=factor,
#                pred_seq=pred_sequence, model_n=model_n)



def train_ConvGRU(config):
    liveloss = PlotLosses()
    dataset_dir = proj_dir + "data/" # "train_valid/{}/{}/".format(pred_seq,'dataset_1')
    train_dir = dataset_dir + "train/pop_pred_" + str(config['n_years']) + "y_" + str(config['n_classes'])+ "c_no_na_oh_norm/"
    train_data = MyDataset(imgs_dir = train_dir + 'input/',masks_dir = train_dir +'target/')
    train_loader = DataLoader(dataset = train_data, batch_size = config['batch_size'], shuffle=True, num_workers= 0)
    
    ori_data_dir = proj_dir + "data/ori_data/pop_pred/input_all_" + str(config['n_years']) + "y_" + str(config['n_classes'])+ "c_no_na_oh_norm.npy"
    valid_input, gt = get_valid_dataset(ori_data_dir)
    # valid_input = valid_input[:,2:,:,:] # no pop unnormed, no lc unnormed
    
    # change to config here
    net = ConvLSTM(input_dim = input_channel,
                   hidden_dim=[config['l1'], 1], #args.n_features], # hidden_dim = [32, 16, args.n_features]
                   kernel_size=(3, 3), num_layers = 2, # num_layers=args.n_layer,
                   batch_first=True, bias=bias_status, return_all_layers=False)
    net.to(device)
    
    optimizer = optim.Adam(net.parameters(), lr = config['lr'], betas = (0.9, 0.999))
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',factor=0.8, patience=10, verbose=True)
    criterion = nn.MSELoss() # no crossentropyloss for regression

    net.apply(weight_init)
    df = pd.DataFrame()
    
    for epoch in range(0, config["epochs"]):
        net.train()
        # epoch_loss = 0
        acc = 0
        train_record = {'train_loss': 0, 'train_acc': 0}

        for i, (imgs, true_masks) in enumerate(train_loader):
            imgs = imgs.to(device=device, dtype=torch.float32) # (b, t, c, w, h)
            # imgs = min_max_scale(imgs) # added to scale all factors but the lc
            imgs = Variable(imgs[:,-6:-2,:,:,:]) # 2015 - 2018

            true_masks = true_masks[:,-6:-2,:,:].to(device) # (b, t, w, h)
            

            # lulc classifer
            output_list = net(imgs[:, :, :, :, :]) # all factors but lc, 4 years
            # output_list = net(imgs)
            masks_pred = output_list[0].squeeze() # (b, t, c, w, h)
            #_, masks_pred_max = torch.max(masks_pred.data, 2) # what is dim 2? the classes?? [4, 6, 7, 256, 256]
            # loss = criterion(masks_pred.permute(0, 2, 1, 3, 4).float(), true_masks.float()) # 4 years, (b, c, t, w, h)
            # lin = nn.Linear(1,1)
            # pred = lin(masks_pred.permute(0,1,3,4,2)).permute(0,1,4,2,3)
            # pred.to(device)
            # loss = criterion(pred.squeeze().float(), true_masks.float()) # 4 years, (b, c, t, w, h), squeeze removes channel dim 1
            loss = criterion(masks_pred.float(), true_masks.float()) # 4 years, (b, c, t, w, h), squeeze removes channel dim 1

            # epoch_loss += loss.item()
            optimizer.zero_grad() # set the gradients to zero
            loss.backward()

            optimizer.step()

            # get acc
            # _, masks_pred_max = torch.max(masks_pred.data, 2)
            # pred_for_acc = pred[:,-1,0,:,:] # last year?
            pred_for_acc = masks_pred[:,-1,:,:] # last year?
            true_masks_for_acc = true_masks[:,-1,:,:] # [:,-1,:,:] # last year?

            corr = torch.sum(pred_for_acc == true_masks_for_acc.detach())
            tensor_size = pred_for_acc.size(0) * pred_for_acc.size(1) * pred_for_acc.size(2)
            acc += float(corr) / float(tensor_size)
            batch_acc = acc/(i+1)

            train_record['train_loss'] += loss.item()
            train_record['train_acc'] += batch_acc

            if i % 5 == 0:

                print('Epoch [{} / {}], batch: {}, train loss: {}, train acc: {}'.format(epoch+1,config["epochs"],i+1,
                                                                                         loss.item(),batch_acc))
        
        train_record['train_loss'] = train_record['train_loss'] / len(train_loader)
        train_record['train_acc'] = train_record['train_acc'] / len(train_loader)
        
        print(train_record)
        
        scheduler.step(batch_acc)
        # scheduler.step()
        # ===================================== Validation ====================================#
        with torch.no_grad():
            net.eval()

            val_record = {'val_kappa': 0, 'val_acc': 0, 'val_loss': 0}
            #k, acc, QA = get_valid_record(valid_input, gt, net, factor_option=factor_option)
            k, acc, ls = get_valid_record(valid_input, gt, net, factor_option='with_factors')

            val_record['val_kappa'] = k
            val_record['val_acc'] = acc
            #val_record['val_QA'] = QA
            val_record['val_loss'] = ls

            print(val_record)
         
            liveloss.update({
                'acc': train_record['train_acc'],
                'val_acc': val_record['val_acc'],
                'loss': train_record['train_loss'],
                'val_loss': val_record['val_loss']
                })
            liveloss.send()   

        print('---------------------------------------------------------------------------------------------------------')

        if config["save_cp"]:
            dir_checkpoint = proj_dir + "data/ckpts/pop_pred/{}/lr{}_bs{}/".format(config["model_n"], config["lr"], config["batch_size"])
            os.makedirs(dir_checkpoint, exist_ok=True)
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch}.pth')
            logging.info(f'Checkpoint {epoch} saved !')
        
        if config["save_csv"]:
            train_record.update(val_record)
            record_df = pd.DataFrame(train_record, index=[epoch])
            df = df.append(record_df)
            record_dir = proj_dir + 'data/record/pop_pred/{}/lr{}_bs{}_1l{}_2l{}/'.format(config["model_n"], config["lr"], config["batch_size"], config["l1"], config["l2"])
            os.makedirs(record_dir, exist_ok=True)
            df.to_csv(record_dir + '{}_lr{}_layer{}_bs{}_1l{}_2l{}.csv'.format(config["model_n"],config["lr"], args.n_layer, config["batch_size"], config["l1"], config["l2"]))




        
print("usage before main")
gpu_usage()



# define random choice of hyperparameters
config = {
        "l1": 2 ** np.random.randint(2, 8), # [4, 8, 16, 32, 64, 128, 256]
        "l2": 'na', # 2 ** np.random.randint(2, 8),
        "lr": np.random.uniform(0.01, 0.00001), # [0.1, 0.00001]
        "batch_size": random.choice([2, 4, 6, 8]),
        "epochs": 50,
        "model_n" : 'pop_No_seed_20y_4c_rand_srch_15-20',
        "save_cp" : True,
        "save_csv" : True,
        "n_years" : 6,
        "n_classes" : 4
    }




print(config)

# run with current set of random hyperparameters
import time
starttime = time.time()
train_ConvGRU(config)

time = time.time() - starttime
print(time)

# check cuda memory
# torch.cuda.current_device() 
# torch.cuda.get_device_name(0)
# torch.cuda.memory_allocated(0) 
# torch.cuda.memory_reserved(0)
# torch.cuda.memory_cached(0)
# torch.cuda.memory_cached(0)-torch.cuda.memory_allocated(0)









