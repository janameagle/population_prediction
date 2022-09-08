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
from sklearn import metrics
import csv


print("initial usage")
gpu_usage()

proj_dir = "H:/Masterarbeit/population_prediction/"
# proj_dir = "C:/Users/jmaie/Documents/Masterarbeit/Code/population_prediction/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


def evaluate(gt, pred):
    # mae = metrics.mean_absolute_error(gt, pred)
    rmse = metrics.mean_squared_error(gt, pred, squared = False)
            
    return rmse

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



def get_valid_dataset(ori_data_dir, model_name):
    ori_data = np.load(ori_data_dir)
    processed_ori_data = ori_data
    # valid_input = processed_ori_data[-5:, :, :, :] # years 2016 - 2019
    # valid_input = processed_ori_data[-5:, :, :, :] # years 2016 - 2019
    # gt = processed_ori_data[-1, 1, :, :] # last year, pop
    if model_name == 'pop_01-20_4y':
        valid_input = processed_ori_data[[3,7,11,15,19], :, :, :] # years 2004-2020, 4y interval
    
    elif model_name == 'pop_05-20_3y':
        valid_input = processed_ori_data[[7,10,13,16,19], :, :, :] # years 2008-2020, 3y interval
    
    elif model_name == 'pop_10-20_2y':
        valid_input = processed_ori_data[[11,13,15,17,19], :, :, :] # years 2012-2020, 2y interval
    
    elif model_name == 'pop_15-20_1y':
        valid_input = processed_ori_data[[15,16,17,18,19], :, :, :] # years 2016-2020, 1y interval
        
    elif model_name == 'pop_02-20_3y':
        valid_input = processed_ori_data[[4,7,10,13,16,19], :, :, :] # years 2005-2020, 3y interval
    
    elif model_name == 'pop_02-20_2y':
        valid_input = processed_ori_data[[3,5,7,9,11,13,15,17,19], :, :, :] # years 2004-2020, 2y interval
    
    elif model_name == 'pop_01_20_1y':
        valid_input = processed_ori_data[1:, :, :, :] # years 2002-2020, 1y interval
        
    elif model_name == 'pop_01_20_4y_static':
        valid_input = processed_ori_data[[3,7,11,15,19], [1,], :, :] # years 2002-2020, 1y interval
         
         
        
    gt = processed_ori_data[19, 1, :, :] # last year, pop
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
    
            test_img = Variable(torch.from_numpy(test_img.copy())).unsqueeze(0).to(device=device,
                                                                                   dtype=torch.float32)

            output_list = net(test_img[:, :-1, 1:, :, :]) # all except lc, except last year

            pred_img = output_list[0].squeeze()
            # lin = nn.Linear(1,1)
            # pred = lin(masks_pred.permute(0,1,3,4,2)).permute(0,1,4,2,3)
            
            pred_img = pred_img[-1,:,:] # take last year prediction
            criterion = nn.MSELoss() # no crossentropyloss for regression
            loss = criterion(pred_img.float(), test_img[:,-1,1,:,:].squeeze().float()) # for validation loss

            pred_img_list.append(pred_img.cpu().numpy())
   
    
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

    val_rmse = evaluate(gt, pred_msk)
    plt.imshow(pred_msk)

    return val_rmse, loss.item()


class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
   
                
early_stopping = EarlyStopping(tolerance=5, min_delta=10)               

print(device)
args = get_args()


bias_status = True                                         
beta = 0                                                          

input_channel = 10 # driving factors                                          
factor = 'with_factors'
pred_sequence = 'forward'


def train_ConvGRU(config):
    liveloss = PlotLosses()
    dataset_dir = proj_dir + "data/" # "train_valid/{}/{}/".format(pred_seq,'dataset_1')
    train_dir = dataset_dir + "train/pop_pred_" + str(config['n_years']) + "y_" + str(config['n_classes'])+ "c_no_na_oh_norm_buf/"
    train_data = MyDataset(imgs_dir = train_dir + 'input/', masks_dir = train_dir +'target/', model_name = config['model_n'])
    train_loader = DataLoader(dataset = train_data, batch_size = config['batch_size'], shuffle=True, num_workers= 0)
    
    ori_data_dir = proj_dir + "data/ori_data/pop_pred/input_all_" + str(config['n_years']) + "y_" + str(config['n_classes'])+ "c_no_na_oh_norm_buf.npy"
    valid_input, gt = get_valid_dataset(ori_data_dir, model_name = config['model_n'])
    
    
    
    net = ConvLSTM(input_dim = input_channel,
                   hidden_dim=[config['l1'], 1], #args.n_features], 
                   kernel_size=(3, 3), num_layers = 2, # num_layers=args.n_layer,
                   batch_first=True, bias=bias_status, return_all_layers=False)
    net.to(device)
    
    
    optimizer = optim.Adam(net.parameters(), lr = config['lr'], betas = (0.9, 0.999))
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.1, patience=10, verbose=True)
    criterion = nn.MSELoss() # no crossentropyloss for regression

    net.apply(weight_init)
    df = pd.DataFrame()
    
    
    
    for epoch in range(0, config["epochs"]):
        net.train()
        rmse = 0
        train_record = {'train_loss': 0, 'train_rmse': 0}

        for i, (imgs, true_masks) in enumerate(train_loader):
            imgs = imgs.to(device=device, dtype=torch.float32) # (b, t, c, w, h)

            # imgs = Variable(imgs[:,-6:-2,1:,:,:]) # b, t, c, w, h, 2015 - 2018, no lc
            # imgs = Variable(imgs) # b, t, c, w, h, 2015 - 2018, no lc

            # true_masks = true_masks[:,-5:-1,:,:].to(device, dtype=torch.float32) # (b, t, w, h)
            true_masks = true_masks.to(device, dtype=torch.float32) # (b, t, w, h)
            
            output_list = net(imgs) # all factors but lc, 4 years

            masks_pred = output_list[0].squeeze() # (b, t, w, h)
            masks_pred = masks_pred[:,-1,:,:]
            loss = criterion(masks_pred, true_masks[:,-1,:,:]) # 4 years, (b, c, t, w, h), squeeze removes channel dim 1

            optimizer.zero_grad() # set the gradients to zero
            loss.backward()

            optimizer.step()

            
            # get acc / error
            pred_for_acc = masks_pred.reshape(masks_pred.shape[0]*masks_pred.shape[-2]*masks_pred.shape[-1]).detach().numpy() # last year?
            true_masks_for_acc = true_masks[:,-1,:,:].reshape(true_masks.shape[0]*true_masks.shape[-2]*true_masks.shape[-1]).detach().numpy() # last year?

            
            # mae += metrics.mean_absolute_error(pred_for_acc, true_masks_for_acc)
            rmse += metrics.mean_squared_error(pred_for_acc, true_masks_for_acc, squared = False)
            # r2 = metrics.r2_score(pred_for_acc, true_masks_for_acc)

            batch_rmse = rmse/(i+1)

            train_record['train_loss'] += loss.item()
            train_record['train_rmse'] += batch_rmse

            if i % 5 == 0:

                print('Epoch [{} / {}], batch: {}, train loss: {}, train rmse: {}'.format(epoch+1,config["epochs"],i+1,
                                                                                         loss.item(), batch_rmse))
            
        
        
        train_record['train_loss'] = train_record['train_loss'] / len(train_loader)
        train_record['train_rmse'] = train_record['train_rmse'] / len(train_loader)
        
        print(train_record)
        
        scheduler.step(batch_rmse)
        # scheduler.step()
        # ===================================== Validation ====================================#
        with torch.no_grad():
            net.eval()

            val_record = {'val_loss': 0, 'val_rmse': 0}
            val_rmse, ls = get_valid_record(valid_input, gt, net, factor_option='with_factors')

            # val_record['val_kappa'] = k
            val_record['val_rmse'] = val_rmse
            #val_record['val_QA'] = QA
            val_record['val_loss'] = ls

            print(val_record)
         
            liveloss.update({
                'rmse': train_record['train_rmse'],
                'val_rmse': val_record['val_rmse'],
                'loss': train_record['train_loss'],
                'val_loss': val_record['val_loss']
                })
            liveloss.send()
            
            

        print('---------------------------------------------------------------------------------------------------------')


        if config["save_cp"]:
            dir_checkpoint = proj_dir + "data/ckpts/{}_buf/lr{}_bs{}_1l{}_2l{}/".format(config["model_n"], config["lr"], config["batch_size"], config["l1"], config["l2"])
            os.makedirs(dir_checkpoint, exist_ok=True)
            torch.save(net.state_dict(),
                        dir_checkpoint + f'CP_epoch{epoch}.pth')
            logging.info(f'Checkpoint {epoch} saved !')
        
        if config["save_csv"]:
            train_record.update(val_record)
            record_df = pd.DataFrame(train_record, index=[epoch])
            df = df.append(record_df)
            record_dir = proj_dir + 'data/record/{}_buf/lr{}_bs{}_1l{}_2l{}/'.format(config["model_n"], config["lr"], config["batch_size"], config["l1"], config["l2"])
            os.makedirs(record_dir, exist_ok=True)
            df.to_csv(record_dir + '{}_lr{}_bs{}_1l{}_2l{}.csv'.format(config["model_n"],config["lr"], config["batch_size"], config["l1"], config["l2"]))


        if epoch == 0:
            with open(record_dir + '/config.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, config.keys())
                writer.writeheader()
                writer.writerow(config)
                

        # Early stopping
        early_stopping(train_record['train_loss'], val_record['val_loss'])
        if early_stopping.early_stop:
            print("We are at epoch:", epoch)
            break


        
print("usage before main")
gpu_usage()



# define random choice of hyperparameters
config = {
        "l1": 64, #2 ** np.random.randint(2, 8), # [4, 8, 16, 32, 64, 128, 256] #64, # 
        "l2": 'na', #2 ** np.random.randint(2, 8), # 'na', # 
        "lr": 0.0012, #round(np.random.uniform(0.01, 0.00001), 4), # [0.1, 0.00001] # 0.0012, # 
        "batch_size": 6, #random.choice([2, 4, 6, 8]),
        "epochs": 50,
        "model_n" : 'pop_01_20_1y',
        "save_cp" : True,
        "save_csv" : True,
        "n_years" : 20,
        "n_classes" : 4
    }




print(config)



# run with current set of random hyperparameters
import time
starttime = time.time()
train_ConvGRU(config)
time1 = time.time() - starttime
print(str(time1/3600) + ' h')



# config['model_n'] = 'pop_15-20_1y'
# starttime2 = time.time()
# train_ConvGRU(config)
# time2 = time.time() - starttime2
# print(str(time2/3600) + ' h')


# config['model_n'] = 'pop_02-20_2y'
# starttime3 = time.time()
# train_ConvGRU(config)
# time3 = time.time() - starttime3
# print(str(time2/3600) + ' h')



# config['model_n'] = 'pop_01_20_1y'
# starttime4 = time.time()
# train_ConvGRU(config)
# time4 = time.time() - starttime4
# print(str(time4/3600) + ' h')