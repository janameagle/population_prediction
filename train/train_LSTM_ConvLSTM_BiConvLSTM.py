# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 12:42:32 2022

@author: jmaie
"""

"""
This script is for training the different neural networks.
The network architectures are stored in ../model/.
Outputs: stored checkpoints for each epoch,
        records about training and validation loss.
"""

from model.v_convlstm import ConvLSTM
from model.bi_convlstm import ConvBLSTM
# from model.v_convgru import ConvGRU
from model.v_lstm import MV_LSTM
# from model.v_gru import GRU
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
# from train.options import get_args
from utilis.weight_init import weight_init
import numpy as np
from livelossplot import PlotLosses # https://github.com/stared/livelossplot/blob/master/examples/pytorch.ipynb
# import matplotlib.pyplot as plt
from GPUtil import showUtilization as gpu_usage
from sklearn import metrics
import csv
import time


# define hyperparameters - or perform random search hyperparameter tuning
config = {
        "l1": 64, # ** np.random.randint(2, 8), # [4, 8, 16, 32, 64, 128, 256]
        "l2": 'na', # ** np.random.randint(2, 8), # 'na', # 
        "lr": 0.0012, # round(np.random.uniform(0.01, 0.00001), 4), # (0.1, 0.00001)
        "batch_size": 2, #6, #random.choice([2, 4, 6, 8]),
        "epochs": 50,
        "model_n" : '04-20_4y', # 02-20_3y, 02-20_2y, 01-20_1y
        "save_cp" : True,
        "save_csv" : True,
        "model": 'ConvLSTM', # 'ConvLSTM', 'LSTM', 'BiConvLSTM', 'ConvGRU'
        "factors" : 'pop', # 'all', 'static', 'pop'
        "run" : 'run1'
    }



conv = False if config['model'] in ['LSTM' , 'BiLSTM'] else True
proj_dir = 'D:/Masterarbeit/population_prediction/'


# check the gpu usage. training is much faster on gpu than cpu - but batch size has to be smaller
print("initial usage")
gpu_usage()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(device)


def evaluate(gt, pred):
    # mae = metrics.mean_absolute_error(gt, pred)
    rmse = metrics.mean_squared_error(gt, pred, squared = False)
    return rmse


# for tiling the study area for the validation
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


# validation is performed on the whole scene and tiled on demand
def get_valid_dataset(ori_data_dir, model_name):  
    ori_data = np.load(ori_data_dir)
    if model_name == '02-20_3y':
        valid_input = ori_data[[4,7,10,13,16,19], :, :, :] # years 2005-2020, 3y interval
    
    elif model_name == '04-20_4y':
        valid_input = ori_data[[7,11,15,19], :, :, :] # years 2008-2020, 4y interval
       
    elif model_name == '02-20_2y':
        valid_input = ori_data[[3,5,7,9,11,13,15,17,19], :, :, :] # years 2004-2020, 2y interval
        
    elif model_name == '01-20_1y':
        valid_input = ori_data[1:, :, :, :] # years 2002-2020, 1y interval
        

    
    if config['factors'] == 'all':
        valid_input = valid_input[:,1:,:,:]  # all input features except multiclass lc
    elif config['factors'] == 'static':
        valid_input = valid_input[:,[1,3,4,5,6],:,:]  # static input features: pop, slope, road dist, water dist, center dist
    elif config['factors'] == 'pop':
        valid_input = valid_input[:, 1, :, :] # population data only
        

    gt = ori_data[19, 1, :, :] # last year, population
    return valid_input, gt


# predict the target year with the trained model to perform the validation
def get_valid_record(valid_input, gt, net, factors, device = device):
    x_list, y_list = get_subsample_centroids(valid_input, img_size=256)
    sub_img_list = []
    for x, y in zip(x_list, y_list):
        if config['factors'] == 'pop':
            sub_img = valid_input[:, np.newaxis, x - 128:x + 128, y - 128:y + 128]
        else:
            sub_img = valid_input[:, :, x - 128:x + 128, y - 128:y + 128]
        sub_img_list.append(sub_img)

    pred_img_list = []
    with torch.no_grad():
        for test_img in sub_img_list:
            
            if conv == False: # LSTM and GRU
                test_img = test_img.reshape(test_img.shape[0], test_img.shape[1], test_img.shape[-2]*test_img.shape[-1]) # (t,c, w*h)
                test_img = np.moveaxis(test_img, 2, 0) # (w*h, t, c)
                test_img = torch.from_numpy(test_img.copy()).to(device=device, dtype=torch.float32) # (w*h, t, c)
                pred_img = net(test_img[:, :-1, :]) # all except last year
                # pred_img = pred_img[:, -1, :].squeeze()
                pred_img = pred_img[:, 0] # take last year prediction
                pred_img_list.append(pred_img.cpu().numpy().reshape(256, 256))
            
            
            else: # convLSTM
                test_img = Variable(torch.from_numpy(test_img.copy())).unsqueeze(0).to(device=device,
                                                                                   dtype=torch.float32)
                output_list = net(test_img[:, :-1, :, :, :]) # except last year
                pred_img = output_list[0].squeeze()
                pred_img = pred_img[-1,:,:] # take last year prediction
                # pred_img = output_list[:,:,-1,:].squeeze()
                # pred_img = output_list.squeeze()
                # criterion = nn.MSELoss()
                # loss = criterion(pred_img.float(), test_img[:,-1,0,:,:].squeeze().float()) # validation loss
                pred_img_list.append(pred_img.cpu().numpy()) #.reshape(256,256))
   
    
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
    # plt.imshow(pred_msk)

    return val_rmse #, loss.item()


# to stop the training, when train and val loss differ strongly. To avoid overfitting
class EarlyStopping():
    def __init__(self, tolerance=10, min_delta=0.01):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_rmse, validation_rmse):
        if abs(validation_rmse - train_rmse) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
   
            
              

def train_ConvGRU(config):
    if conv == False: # LSTM and GRU
        config['batch_size'] = 1
    
    liveloss = PlotLosses()
    train_dir = proj_dir + "data/train/"
    train_data = MyDataset(imgs_dir = train_dir + 'input/', masks_dir = train_dir +'target/', model_name = config['model_n'])
    train_loader = DataLoader(dataset = train_data, batch_size = config['batch_size'], shuffle=True, num_workers= 0)
    
    ori_data_dir = proj_dir + "data/ori_data/input_all.npy"
    valid_input, gt = get_valid_dataset(ori_data_dir, config['model_n'])
    
    
    if conv == False: # LSTM and GRU
        seq_length = valid_input.shape[0]-1
        

    
    input_channel = 10 if config['factors'] == 'all' else 5 if config['factors'] == 'static' else 1
     
    
    # define the model
    if config["model"] == 'ConvLSTM':       
        net = ConvLSTM(input_dim = input_channel,
                       hidden_dim= config['l1'], #[config['l1'], 1], 
                       kernel_size=(3,3), num_layers = 1, # (3,3), num_layers = 2, 
                       batch_first=True, return_all_layers=False)
    
    elif config["model"] == 'BiConvLSTM':
        net = ConvBLSTM(input_dim = input_channel,
                       hidden_dim=config['l1'],
                       kernel_size=3,
                       batch_first=True, return_all_layers=False)
    
    elif config["model"] == 'LSTM':
          net = MV_LSTM(n_features = input_channel,
                        seq_length = seq_length,
                        hidden_dim = config['l1'],
                        num_layers = 1,
                        batch_first = True,
                        bidirectional = False) 
    
    elif config["model"] == 'BiLSTM':
          net = MV_LSTM(n_features = input_channel,
                        seq_length = seq_length,
                        hidden_dim = config['l1'],
                        num_layers = 1,
                        batch_first = True,
                        bidirectional = True)
          
    # elif config["model"] == 'ConvGRU':
    #     net = ConvGRU(input_dim = input_channel,
    #                   hidden_dim = [config['l1'],1],
    #                   kernel_size=(3,3), 
    #                   num_layers = 2,
    #                   batch_first = True, return_all_layers=False)


    net.to(device)
    # or load pretrained model:
    # dir_checkpoint = proj_dir + 'data/ckpts/LSTM_02-20_3y_all/lr0.0012_bs1_1l64_2lna/run2/CP_epoch6.pth'
    # net.load_state_dict(torch.load(dir_checkpoint))
    
    optimizer = optim.Adam(net.parameters(), lr = config['lr'], betas = (0.9, 0.999))
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    criterion = nn.MSELoss()

    if conv == True:
        net.apply(weight_init)
    
    df = pd.DataFrame()
    
    
    for epoch in range(config['epochs']):
        net.train()
        rmse = 0
        train_record = {'train_rmse': 0} # {'train_loss': 0, 'train_rmse': 0}

        for i, (imgs, true_masks) in enumerate(train_loader):

            if config['factors'] == 'static':
                imgs = imgs[:,:,[0,2,3,4,5],:,:] # select static features only, lc is already removed
         
            if config['factors'] == 'pop':
             imgs = imgs[:,:,0,:,:] # select pop only, lc is already removed
             imgs = imgs[:,:,np.newaxis,:,:]

            if conv == False: # LSTM and GRU. # reshape to 1d
                imgs = imgs.squeeze(0) # (t,c,w,h)
                imgs = imgs.reshape(imgs.shape[0], imgs.shape[1], imgs.shape[-2]*imgs.shape[-1]) # (t,c, w*h)
                imgs = torch.moveaxis(imgs, 2, 0) # (w*h, t, c)
                true_masks = true_masks.squeeze()
                true_masks = true_masks.reshape(true_masks.shape[0], true_masks.shape[-2]*true_masks.shape[-1])
                true_masks = torch.moveaxis(true_masks,1,0) # (w*h, t)
                net.init_hidden(imgs.shape[0])
                

            imgs = imgs.to(device=device, dtype=torch.float32) # (b, t, c, w, h)
            true_masks = true_masks.to(device, dtype=torch.float32) # (b, t, w, h)
            
            
            if conv == False: 
                output = net(imgs)
                # output = output[:, -1, :]
                loss = criterion(output.view(-1), true_masks[:,-1]) 
                pred_for_acc = output.detach().cpu().numpy()
                true_masks_for_acc = true_masks[:,-1].detach().cpu().numpy()

                
            else:
                output = net(imgs)
                masks_pred = output.squeeze() # (b, dim, t, w, h), dim = 1

                # masks_pred = output_list[0].squeeze() # (b, t, w, h)
                # masks_pred = masks_pred[:,-1,:,:] # last year's prediction
                masks_pred = masks_pred[:,:,-1]
                # loss = criterion(masks_pred, true_masks[:,-1,:,:])
                t = true_masks[:,-1,:,:]
                t = t.reshape(t.shape[-2]*t.shape[-1], t.shape[0])
                loss = criterion(masks_pred, t)
            
            optimizer.zero_grad() # set the gradients to zero
            loss.backward()
            optimizer.step()

            # get error
            # if conv == True:
            #     pred_for_acc = masks_pred.reshape(masks_pred.shape[0]*masks_pred.shape[-2]*masks_pred.shape[-1]).detach().cpu().numpy()
            #     true_masks_for_acc = true_masks[:,-1,:,:].reshape(true_masks.shape[0]*true_masks.shape[-2]*true_masks.shape[-1]).detach().cpu().numpy()

            # mae += metrics.mean_absolute_error(pred_for_acc, true_masks_for_acc)
            # rmse += metrics.mean_squared_error(pred_for_acc, true_masks_for_acc, squared = False)
            rmse += metrics.mean_squared_error(masks_pred.detach().cpu().numpy(), t.detach().cpu().numpy(), squared = False)
            # r2 = metrics.r2_score(pred_for_acc, true_masks_for_acc)

            batch_rmse = rmse/(i+1)

            # train_record['train_loss'] += loss.item()
            train_record['train_rmse'] += batch_rmse

            if i % 80 == 0:
                print('Epoch [{} / {}], batch: {}, train loss: {}, train rmse: {}, lr: {}'.format(epoch+1,config["epochs"],i+1,
                                                                                         loss.item(), batch_rmse, optimizer.param_groups[0]['lr']))
            

        # train_record['train_loss'] = train_record['train_loss'] / len(train_loader)
        train_record['train_rmse'] = train_record['train_rmse'] / len(train_loader)
        
        print(train_record)
        
        # ===================================== Validation ====================================#
        with torch.no_grad():
            net.eval()

            val_record = {'val_rmse': 0} # {'val_loss': 0, 'val_rmse': 0}
            val_rmse = get_valid_record(valid_input, gt, net, factors = config['factors'])

            val_record['val_rmse'] = val_rmse
            # val_record['val_loss'] = ls
            scheduler.step(val_rmse)
            
            print(val_record)
            
            # print live train and val error for each epoch
            liveloss.update({
                'rmse': train_record['train_rmse'],
                'val_rmse': val_record['val_rmse'],
                # 'loss': train_record['train_loss'],
                # 'val_loss': val_record['val_loss']
                })
            liveloss.send()
            
            

        print('---------------------------------------------------------------------------------------------------------')

        save_name = '{}_{}_{}/lr{}_bs{}_1l{}_2l{}/{}/'.format(config["model"], config["model_n"], config["factors"], config["lr"], config["batch_size"], config["l1"], config["l2"], config["run"])        

        if config["save_cp"]:
            dir_checkpoint = proj_dir + "data/ckpts/" + save_name
            os.makedirs(dir_checkpoint, exist_ok=True)
            torch.save(net.state_dict(), dir_checkpoint + f'CP_epoch{epoch}.pth')
            logging.info(f'Checkpoint {epoch} saved !')
        
        if config["save_csv"]:
            train_record.update(val_record)
            record_df = pd.DataFrame(train_record, index=[epoch])
            df = df.append(record_df)
            record_dir = proj_dir + 'data/record/' + save_name
            os.makedirs(record_dir, exist_ok=True)
            df.to_csv(record_dir + '{}_{}_{}_lr{}_bs{}_1l{}_2l{}.csv'.format(config["model"], config["model_n"], config["factors"],config["lr"], config["batch_size"], config["l1"], config["l2"]))



        if epoch == 0: # save the config information once
            with open(record_dir + '/config.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, config.keys())
                writer.writeheader()
                writer.writerow(config)
                

        # Early stopping
        early_stopping(train_record['train_rmse'], val_record['val_rmse'])
        if early_stopping.early_stop:
            print("Early stopped. We are at epoch:", epoch)
            break

    return record_dir





# define early stopping settings
early_stopping = EarlyStopping(tolerance=10, min_delta=0.01) 


# train multiple models
all_models = ['ConvLSTM']   # ['LSTM', 'BiLSTM', 'ConvLSTM', 'BiConvLSTM']
all_factors = ['pop']       # ['all', 'static', 'pop']
all_modeln = ['02-20_3y']   # ['02-20_3y', '04-20_4y', '02-20_2y', '01-20_1y']
runs = ['run6']             # ['run1', 'run2', 'run3', 'run4', 'run5']


for m in all_models:
    for n in all_modeln:
        for r in runs:
            config['model'] = m
            config['model_n'] = n
            config['run'] = r
            print(config)
            starttime = time.time() # to track training time
            record_dir = train_ConvGRU(config)
            hours = (time.time() - starttime)/3600
            df = pd.DataFrame({'runtime' : [hours]})
            df.to_csv(record_dir + 'runtime.csv')
            early_stopping.counter = 0 # reset early stopping

