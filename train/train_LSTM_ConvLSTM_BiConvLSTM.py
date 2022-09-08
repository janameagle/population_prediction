# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 12:42:32 2022

@author: jmaie
"""


from model.v_convlstm import ConvLSTM
from model.bi_convlstm import ConvBLSTM
from model.v_lstm import MV_LSTM
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
import matplotlib.pyplot as plt
from sklearn import metrics
import csv
import time



proj_dir = "H:/Masterarbeit/population_prediction/"

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



def get_valid_dataset(ori_data_dir, model_name):  # input data for validation
    ori_data = np.load(ori_data_dir)
    processed_ori_data = ori_data
    if config['model_n'] == 'pop_02-20_3y':
        valid_input = processed_ori_data[[4,7,10,13,16,19], :, :, :] # years 2005-2020, 3y interval
        
    # elif model_name == 'pop_02-20_3y_static':
    #      valid_input = processed_ori_data[[4,7,10,13,16,19], :, :, :] # years 2005-2020, 3y interval   
         # valid_input = valid_input[:,[1,3,4,5,6],:,:]                 # static input features
        
    elif model_name == 'pop_01-20_4y':
        valid_input = processed_ori_data[[3,7,11,15,19], :, :, :] # years 2004-2020, 4y interval
        # valid_input = valid_input[:,[1,3,4,5,6],:,:]              # static input features
        
    elif model_name == 'pop_02-20_2y':
        valid_input = processed_ori_data[[3,5,7,9,11,13,15,17,19], :, :, :] # years 2004-2020, 2y interval
        # valid_input = valid_input[:,[1,3,4,5,6],:,:]                        # static input features
        
    elif model_name == 'pop_01_20_1y':
        valid_input = processed_ori_data[1:, :, :, :] # years 2002-2020, 1y interval
        # valid_input = valid_input[:,[1,3,4,5,6],:,:]  # static input features

        
    # elif model_name == 'pop_01-20_4y':
    #     valid_input = processed_ori_data[[3,7,11,15,19], :, :, :] # years 2004-2020, 4y interval
    
    # elif model_name == 'pop_05-20_3y':
    #     valid_input = processed_ori_data[[7,10,13,16,19], :, :, :] # years 2008-2020, 3y interval
    
    # elif model_name == 'pop_10-20_2y':
    #     valid_input = processed_ori_data[[11,13,15,17,19], :, :, :] # years 2012-2020, 2y interval
    
    # elif model_name == 'pop_15-20_1y':
    #     valid_input = processed_ori_data[[15,16,17,18,19], :, :, :] # years 2016-2020, 1y interval
        
    # elif model_name == 'pop_02-20_3y':
    #     valid_input = processed_ori_data[[4,7,10,13,16,19], :, :, :] # years 2005-2020, 3y interval
    
    # elif model_name == 'pop_02-20_2y':
    #     valid_input = processed_ori_data[[3,5,7,9,11,13,15,17,19], :, :, :] # years 2004-2020, 2y interval
    
    # elif model_name == 'pop_01_20_1y':
    #     valid_input = processed_ori_data[1:, :, :, :] # years 2002-2020, 1y interval
    
    
    if config['factors'] == 'all':
        valid_input = valid_input[:,1:,:,:]  # all input features except multiclass lc
    elif config['factors'] == 'static':
        valid_input = valid_input[:,[1,3,4,5,6],:,:]  # static input features
    elif config['factors'] == 'pop':
        valid_input = valid_input[:, 1, :, :] # population data only
        

    gt = processed_ori_data[19, 1, :, :] # last year, population
    return valid_input, gt


def get_valid_record(valid_input, gt, net, factors, device = device):
    x_list, y_list = get_subsample_centroids(valid_input, img_size=256)
    sub_img_list = []
    for x, y in zip(x_list, y_list):
        sub_img = valid_input[:, :, x - 128:x + 128, y - 128:y + 128]
        sub_img_list.append(sub_img)

    pred_img_list = []
    with torch.no_grad():
        for test_img in sub_img_list:
            
            if config["model"] == 'LSTM':
                test_img = test_img.reshape(test_img.shape[0], test_img.shape[1], test_img.shape[-2]*test_img.shape[-1]) # (t,c, w*h)
                test_img = np.moveaxis(test_img, 2, 0) # (w*h, t, c)
                test_img = torch.from_numpy(test_img.copy()).to(device=device, dtype=torch.float32) # (w*h, t, c)
                pred_img = net(test_img[:, :-1, :]) # all except last year
                pred_img = pred_img[:, 0] # take last year prediction
                pred_img_list.append(pred_img.cpu().numpy().reshape(256, 256))
            
            
            else:
                test_img = Variable(torch.from_numpy(test_img.copy())).unsqueeze(0).to(device=device,
                                                                                   dtype=torch.float32)
                output_list = net(test_img[:, :-1, :, :, :]) # except last year
                pred_img = output_list[0].squeeze()
                pred_img = pred_img[-1,:,:] # take last year prediction
                # criterion = nn.MSELoss()
                # loss = criterion(pred_img.float(), test_img[:,-1,0,:,:].squeeze().float()) # validation loss
                pred_img_list.append(pred_img.cpu().numpy())
   
    
    pred_msk = np.zeros((valid_input.shape[-2], valid_input.shape[-1]))

    h = 0
    x_list, y_list = get_subsample_centroids(valid_input, img_size=256)
    for x, y in zip(x_list, y_list):
        if x == np.min(x_list) or x == np.max(x_list) or y == np.min(y_list) or y == np.max(y_list):
            pred_msk[x - 128:x + 128, y - 128:y + 128] = pred_img_list[h]
            h += 1
        else:
            pred_msk[x - 106:x + 106, y - 106:y + 106] = pred_img_list[h][22:234, 22:234]
            h += 1

    val_rmse = evaluate(gt, pred_msk)
    # plt.imshow(pred_msk)

    return val_rmse #, loss.item()


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
    if config['model'] == 'LSTM':
        config['batch_size'] = 1
        seq_length = 5
    
    liveloss = PlotLosses()
    dataset_dir = proj_dir + "data/"
    train_dir = dataset_dir + "train/pop_pred_{}y_{}c_no_na_oh_norm_buf/".format(config['n_years'], config['n_classes'])
    train_data = MyDataset(imgs_dir = train_dir + 'input/', masks_dir = train_dir +'target/', model_name = config['model_n'])
    train_loader = DataLoader(dataset = train_data, batch_size = config['batch_size'], shuffle=True) #, num_workers= 0)
    
    ori_data_dir = proj_dir + "data/ori_data/pop_pred/input_all_{}y_{}c_no_na_oh_norm_buf.npy".format(config['n_years'],config['n_classes'])
    valid_input, gt = get_valid_dataset(ori_data_dir, config['model_n'])

    
    input_channel = 10 if config['factors'] == 'all' else 5 if config['factors'] == 'static' else 1
     
    
    if config["model"] == 'ConvLSTM':       
        net = ConvLSTM(input_dim = input_channel,
                       hidden_dim=[config['l1'], 1], 
                       kernel_size=(3, 3), num_layers = 2, 
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
                        bidirectional = False) # try true
        

    net.to(device)
    
    
    optimizer = optim.Adam(net.parameters(), lr = config['lr'], betas = (0.9, 0.999))
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    criterion = nn.MSELoss()

    net.apply(weight_init)
    df = pd.DataFrame()
    
    
    for epoch in range(0, config['epochs']):
        net.train()
        rmse = 0
        train_record = {'train_rmse': 0} # {'train_loss': 0, 'train_rmse': 0}

        for i, (imgs, true_masks) in enumerate(train_loader):

            if config['factors'] == 'static':
                imgs = imgs[:,:,[0,2,3,4,5],:,:] # select static features only, lc is already removed


            if config['model'] == 'LSTM': # reshape to 1d
                imgs = imgs.squeeze() # (t,c,w,h)
                imgs = imgs.reshape(imgs.shape[0], imgs.shape[1], imgs.shape[-2]*imgs.shape[-1]) # (t,c, w*h)
                imgs = torch.moveaxis(imgs, 2, 0) # (w*h, t, c)
                true_masks = true_masks.squeeze()
                true_masks = true_masks.reshape(true_masks.shape[0], true_masks.shape[-2]*true_masks.shape[-1])
                true_masks = torch.moveaxis(true_masks,1,0) # (w*h, t)
                net.init_hidden(imgs.shape[0])
                

            imgs = imgs.to(device=device, dtype=torch.float32) # (b, t, c, w, h)
            true_masks = true_masks.to(device, dtype=torch.float32) # (b, t, w, h)
            
            
            if config['model'] == 'LSTM':
                output = net(imgs)
                loss = criterion(output.view(-1), true_masks[:,-1]) 
                pred_for_acc = output.detach().numpy()
                true_masks_for_acc = true_masks[:,-1].detach().numpy()
                
                
            else:
                output_list = net(imgs)

                masks_pred = output_list[0].squeeze() # (b, t, w, h)
                masks_pred = masks_pred[:,-1,:,:] # last year's prediction
                loss = criterion(masks_pred, true_masks[:,-1,:,:])
            
            optimizer.zero_grad() # set the gradients to zero
            loss.backward()
            optimizer.step()

            # get error
            pred_for_acc = masks_pred.reshape(masks_pred.shape[0]*masks_pred.shape[-2]*masks_pred.shape[-1]).detach().numpy()
            true_masks_for_acc = true_masks[:,-1,:,:].reshape(true_masks.shape[0]*true_masks.shape[-2]*true_masks.shape[-1]).detach().numpy()

            
            # mae += metrics.mean_absolute_error(pred_for_acc, true_masks_for_acc)
            rmse += metrics.mean_squared_error(pred_for_acc, true_masks_for_acc, squared = False)
            # r2 = metrics.r2_score(pred_for_acc, true_masks_for_acc)

            batch_rmse = rmse/(i+1)

            # train_record['train_loss'] += loss.item()
            train_record['train_rmse'] += batch_rmse

            if i % 5 == 0:
                print('Epoch [{} / {}], batch: {}, train loss: {}, train rmse: {}, lr: {}'.format(epoch+1,config["epochs"],i+1,
                                                                                         loss.item(), batch_rmse, optimizer.param_groups[0]['lr']))
            

        # train_record['train_loss'] = train_record['train_loss'] / len(train_loader)
        train_record['train_rmse'] = train_record['train_rmse'] / len(train_loader)
        
        print(train_record)
        
        # scheduler.step(batch_rmse)
        # scheduler.step()
        # ===================================== Validation ====================================#
        with torch.no_grad():
            net.eval()

            val_record = {'val_rmse': 0} # {'val_loss': 0, 'val_rmse': 0}
            val_rmse = get_valid_record(valid_input, gt, net, factors = config['factors'])

            val_record['val_rmse'] = val_rmse
            # val_record['val_loss'] = ls
            scheduler.step(val_rmse)
            
            print(val_record)
         
            liveloss.update({
                'rmse': train_record['train_rmse'],
                'val_rmse': val_record['val_rmse'],
                # 'loss': train_record['train_loss'],
                # 'val_loss': val_record['val_loss']
                })
            liveloss.send()
            
            

        print('---------------------------------------------------------------------------------------------------------')

        save_name = '{}_{}_{}/lr{}_bs{}_1l{}_2l{}/'.format(config["model"], config["model_n"], config["factors"], config["lr"], config["batch_size"], config["l1"], config["l2"])        

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
            df.to_csv(record_dir + save_name + '.csv')


        if epoch == 0:
            with open(record_dir + '/config.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, config.keys())
                writer.writeheader()
                writer.writerow(config)
                

        # Early stopping
        early_stopping(train_record['train_rmse'], val_record['val_rmse'])
        if early_stopping.early_stop:
            print("We are at epoch:", epoch)
            break



# define hyperparameters
config = {
        "l1": 64, #2 ** np.random.randint(2, 8), # [4, 8, 16, 32, 64, 128, 256]
        "l2": 'na', #2 ** np.random.randint(2, 8), # 'na', # 
        "lr": 0.0012, # round(np.random.uniform(0.01, 0.00001), 4), # (0.1, 0.00001)
        "batch_size": 6, #random.choice([2, 4, 6, 8]),
        "epochs": 50,
        "model_n" : '02-20_3y',
        "save_cp" : True,
        "save_csv" : True,
        "n_years" : 20,
        "n_classes" : 4,
        "model": 'ConvLSTM', # 'ConvLSTM', 'LSTM', 'BiConvLSTM'
        "factors" : 'static' # 'all', 'static', 'pop'
    }




print(config)
early_stopping = EarlyStopping(tolerance=5, min_delta=0.01) 


# run with current set of random hyperparameters
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