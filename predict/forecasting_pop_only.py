import numpy as np
import os
import torch
from tqdm import tqdm
from torch.autograd import Variable
from model.v_lstm import MV_LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tifffile



proj_dir = "D:/Masterarbeit/population_prediction/"

# define config
config = {
        "l1": 64, 
        "l2": 'na', 
        "lr": 0.0012, 
        "batch_size": 2, 
        "epochs": 50, #50
        "model_n" : '02-20_2y',
        "save" : True,
        "model": 'BiLSTM', # 'ConvLSTM', 'LSTM', 'BiConvLSTM', ('linear_reg', 'multivariate_reg',' 'random_forest_reg')
        "factors" : 'pop', # 'all', 'static', 'pop'
        "run" : 'run5',
        "forecast": 28
    }


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



def main(*kwargs):
    conv = False if config['model'] in ['LSTM' , 'BiLSTM'] else True
    if conv == False: # LSTM and GRU
        config['batch_size'] = 1

        
        
    save_name = '{}_{}_{}/lr{}_bs{}_1l{}_2l{}/{}/'.format(config["model"], config["model_n"], config["factors"], config["lr"], config["batch_size"], config["l1"], config["l2"], config["run"])        

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ori_data_dir = proj_dir + 'data/ori_data/input_all.npy'
    ori_data = np.load(ori_data_dir)# .transpose((1, 0, 2, 3))
    
    if config['model_n'] == '02-20_3y':     
        if config['forecast'] == 23:
            valid_input = ori_data[[7,10,13,16,19], 1, :, :] # 2008 - 2020, 3y
        
        if config['forecast'] == 26:
            valid_input = ori_data[[10,13,16,19], 1, :, :] # 2011 - 2020, 3y
            y23 = np.load(proj_dir + 'data/test/' + save_name + 'frc23/pred_msk_eval_normed.npy')
            valid_input = np.concatenate((valid_input, y23[np.newaxis,:,:]), axis=0)
            
        if config['forecast'] == 29:
            valid_input = ori_data[[13,16,19], 1, :, :] # 2014 - 2020, 3y
            y23 = np.load(proj_dir + 'data/test/' + save_name + 'frc23/pred_msk_eval_normed.npy')
            y26 = np.load(proj_dir + 'data/test/' + save_name + 'frc26/pred_msk_eval_normed.npy')
            valid_input = np.concatenate((valid_input, y23[np.newaxis,:,:], y26[np.newaxis,:,:]), axis=0)
            
        if config['forecast'] == 32:
            valid_input = ori_data[[16,19], 1, :, :] # 2017 - 2020, 3y
            y23 = np.load(proj_dir + 'data/test/' + save_name + 'frc23/pred_msk_eval_normed.npy')
            y26 = np.load(proj_dir + 'data/test/' + save_name + 'frc26/pred_msk_eval_normed.npy')
            y29 = np.load(proj_dir + 'data/test/' + save_name + 'frc29/pred_msk_eval_normed.npy')
            valid_input = np.concatenate((valid_input, y23[np.newaxis,:,:], y26[np.newaxis,:,:], y29[np.newaxis,:,:]), axis=0)
            
        if config['forecast'] == 35:
            valid_input = ori_data[[19], 1, :, :] # 2020
            y23 = np.load(proj_dir + 'data/test/' + save_name + 'frc23/pred_msk_eval_normed.npy')
            y26 = np.load(proj_dir + 'data/test/' + save_name + 'frc26/pred_msk_eval_normed.npy')
            y29 = np.load(proj_dir + 'data/test/' + save_name + 'frc29/pred_msk_eval_normed.npy')
            y32 = np.load(proj_dir + 'data/test/' + save_name + 'frc32/pred_msk_eval_normed.npy')
            valid_input = np.concatenate((valid_input, y23[np.newaxis,:,:], y26[np.newaxis,:,:], y29[np.newaxis,:,:], y32[np.newaxis,:,:]), axis=0)
     
        
    if config['model_n'] == '02-20_2y':     
        if config['forecast'] == 22:
            valid_input = ori_data[[5,7,9,11,13,15,17,19], 1, :, :] 
        
        if config['forecast'] == 24:
            valid_input = ori_data[[7,9,11,13,15,17,19], 1, :, :] 
            y22 = np.load(proj_dir + 'data/test/' + save_name + 'frc22/pred_msk_eval_normed.npy')
            valid_input = np.concatenate((valid_input, y22[np.newaxis,:,:]), axis=0)
            
        if config['forecast'] == 26:
            valid_input = ori_data[[9,11,13,15,17,19], 1, :, :] 
            y22 = np.load(proj_dir + 'data/test/' + save_name + 'frc22/pred_msk_eval_normed.npy')
            y24 = np.load(proj_dir + 'data/test/' + save_name + 'frc24/pred_msk_eval_normed.npy')
            valid_input = np.concatenate((valid_input, y22[np.newaxis,:,:], y24[np.newaxis,:,:]), axis=0)
            
        if config['forecast'] == 28:
            valid_input = ori_data[[11,13,15,17,19], 1, :, :] 
            y22 = np.load(proj_dir + 'data/test/' + save_name + 'frc22/pred_msk_eval_normed.npy')
            y24 = np.load(proj_dir + 'data/test/' + save_name + 'frc24/pred_msk_eval_normed.npy')
            y26 = np.load(proj_dir + 'data/test/' + save_name + 'frc26/pred_msk_eval_normed.npy')
            valid_input = np.concatenate((valid_input, y22[np.newaxis,:,:], y24[np.newaxis,:,:], y26[np.newaxis,:,:]), axis=0)
            
        if config['forecast'] == 30:
            valid_input = ori_data[[13,15,17,19], 1, :, :] 
            y22 = np.load(proj_dir + 'data/test/' + save_name + 'frc22/pred_msk_eval_normed.npy')
            y24 = np.load(proj_dir + 'data/test/' + save_name + 'frc24/pred_msk_eval_normed.npy')
            y26 = np.load(proj_dir + 'data/test/' + save_name + 'frc26/pred_msk_eval_normed.npy')
            y28 = np.load(proj_dir + 'data/test/' + save_name + 'frc28/pred_msk_eval_normed.npy')
            valid_input = np.concatenate((valid_input, y22[np.newaxis,:,:], y24[np.newaxis,:,:], y26[np.newaxis,:,:], y28[np.newaxis,:,:]), axis=0)
        
    if conv == False: # LSTM and GRU
        seq_length = valid_input.shape[0]
        
    input_channel = 10 if config['factors'] == 'all' else 5 if config['factors'] == 'static' else 1
        

    dir_checkpoint = proj_dir + 'data/ckpts/' + save_name + 'CP_epoch{}.pth'.format(config["epochs"]-1)        
    
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
        for test_img in tqdm(sub_img_list):
            
            if config["model"] == 'BiLSTM':
                  net = MV_LSTM(n_features = input_channel,
                                seq_length = seq_length,
                                hidden_dim = config['l1'],
                                num_layers = 1,
                                batch_first = True,
                                bidirectional = True)
            
            net.to(device)
            net.load_state_dict(torch.load(dir_checkpoint))
            
            
            if conv == False: # LSTM and GRU
                #test_img = test_img.squeeze()
                test_img = test_img.reshape(test_img.shape[0], test_img.shape[1], test_img.shape[-2]*test_img.shape[-1]) # (t,c, w*h)
                test_img = torch.from_numpy(test_img.copy()).to(device=device, dtype=torch.float32) # (w*h, t, c)
                test_img = torch.moveaxis(test_img, 2, 0) # (w*h, t, c)
                net.init_hidden(test_img.shape[0])
                pred_img = net(test_img) 
                pred_img = pred_img[:, 0] # take last year prediction
                pred_img_list.append(pred_img.cpu().numpy().reshape(256, 256))
            
            
            else:
                test_img = Variable(torch.from_numpy(test_img.copy())).unsqueeze(0).to(device=device,
                                                                                       dtype=torch.float32)
                output_list = net(test_img) 
                pred_img = output_list[0].squeeze()
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
            pred_msk[x - 106:x + 106, y - 106:y + 106] = pred_img_list[h][22:234,22:234]
            h += 1

    plt.imshow(pred_msk)

    # rescale to actual pop values
    ori_unnormed = np.load(proj_dir + 'data/ori_data/input_all_unnormed.npy')
    pop_unnormed = ori_unnormed[:, 1, :, :]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(pop_unnormed.reshape(-1, 1))
    pop = scaler.inverse_transform(pred_msk.reshape(-1,1)).reshape(pred_msk.shape[-2], pred_msk.shape[-1])


    save_path = proj_dir + "data/test/" + save_name + 'frc' + str(config['forecast']) + '/' 
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path + 'pred_msk_eval_normed.npy', pred_msk)
    np.save(save_path + 'pred_msk_eval_rescaled.npy', pop)
    #cv2.imwrite(save_path + 'pred_msk_eval.png', pred_msk)
    plt.savefig(save_path + 'pred_msk_eval.png')
    
    
    tifffile.imwrite(save_path + 'pred_msk_normed.tif', pred_msk)
    tifffile.imwrite(save_path + 'pred_msk_rescaled.tif', pop)
    

main()

