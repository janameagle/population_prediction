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


proj_dir = "H:/Masterarbeit/population_prediction/"
# proj_dir = "C:/Users/jmaie/Documents/Masterarbeit/Code/population_prediction/"


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


def oh_code(a, class_n = 7):
    oh_list = []
    for i in range(class_n):
        temp = np.where(a == i, 1, 0)
        oh_list.append(temp)
    return np.array(oh_list)

def pre_prcessing(crop_img):
    crop_img_lulc = crop_img[:, 0, :, :]
    temp_list = []
    for j in range(crop_img_lulc.shape[0]):
        temp = oh_code(crop_img_lulc[j], class_n=7)
        temp_list.append(temp[np.newaxis, :, :, :])
    oh_crop_img_lulc = np.concatenate(temp_list, axis=0)
    oh_crop_img = np.concatenate((oh_crop_img_lulc, crop_img[:, 1:, :, :]), axis=1)
    return oh_crop_img

# def color_annotation(image):
#     color = np.ones([image.shape[0], image.shape[1], 3])
#     color[image == 0] = [0, 102, 0]  # shrub
#     color[image == 1] = [0, 255, 255]  # savanna
#     color[image == 2] = [0, 204, 0]  # grassland
#     color[image == 3] = [0, 128, 255]  # croplands
#     color[image == 4] = [0, 0, 255]  # urban
#     color[image == 5] = [128, 128, 128]  # barren
#     color[image == 6] = [255, 128, 0]  # water
#     return color

def color_annotation(image):
    color = np.ones([image.shape[0], image.shape[1], 3])
    color[image == 0] = [0, 102, 0]  # vegetation
    color[image == 1] =  [0, 0, 255]  # urban
    color[image == 2] = [128, 128, 128]  # barren
    color[image == 3] = [255, 128, 0]  # water
    return color


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

n_years = 6
n_classes = 4

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    args = get_args()

    bias_status = True

    ori_data_dir = proj_dir + 'data/ori_data/pop_pred/input_all_' + str(n_years) + 'y_' + str(n_classes) + 'c_no_na_oh_norm.npy'

    ori_data = np.load(ori_data_dir)# .transpose((1, 0, 2, 3))
    # scaled_data = torch.nn.functional.interpolate(torch.from_numpy(ori_data),
    #                                               scale_factor=(1 / 3, 1 / 3),
    #                                               recompute_scale_factor=True)
    # processed_ori_data = scaled_data.numpy()
    processed_ori_data = ori_data
    valid_input = processed_ori_data[-5:-1, 1:, :, :] # [1:5, :, :, :] = years 2000 - 2020, no lc unnormed
    gt = processed_ori_data[-1, 1, :, :] # last year, pop
    print('valid_sequence shape: ', valid_input.shape) # t,c,w,h; pop, ...

    input_channel = 10 # 19

    df = pd.DataFrame()
    valid_record = {'kappa': 0, 'acc': 0}

    #dir_checkpoint = './ckpts/forecasting/{}_{}/{}/CP_epoch100.pth'
    dir_checkpoint = proj_dir + "data/ckpts/pop_pred/pop_No_seed_20y_4c_rand_srch_15-20/lr0.0013723866073356884_bs4/CP_epoch49.pth"

    print(dir_checkpoint)
    x_list, y_list = get_subsample_centroids(valid_input, img_size=256)

    sub_img_list = []
    for x, y in zip(x_list, y_list):
        sub_img = valid_input[:, :, x - 128:x + 128, y - 128:y + 128]
        sub_img_list.append(sub_img)

    pred_img_list = []

    with torch.no_grad():
        for test_img in tqdm(sub_img_list):
            # test_img = pre_prcessing(test_img)
            test_img = Variable(torch.from_numpy(test_img.copy())).unsqueeze(0).to(device=device,
                                                                                   dtype=torch.float32)
            # test_img = min_max_scale(test_img) # added to scale all factors but the lc

            net = ConvLSTM(input_dim=input_channel,
                          hidden_dim=[64, 1], # args.n_features],
                          kernel_size=(3, 3), num_layers= 2 , # num_layers= args.n_layer,
                          batch_first=True, bias=bias_status, return_all_layers=False)

            net.to(device)
            net.load_state_dict(torch.load(dir_checkpoint))
            # net.eval() # added from Pytorch tutorial

            output_list = net(test_img[:,:,:,:,:]) # all factors but lc

            masks_pred = output_list[0].squeeze().view(args.seq_len, 256, 256) # t, c, w, h
            # pred_prob = torch.softmax(torch.squeeze(masks_pred),dim=1).data.cpu().numpy()
            # lin = nn.Linear(4,1).to(device)
            #masks_pred.to(device)
            #pred = lin(masks_pred.permute(0,2,3,1)).permute(0,3,1,2)
            #pred.to(device)
            # pred_img = masks_pred.squeeze()
            pred_img_list.append(masks_pred[-1,:,:].cpu().numpy())
            
            # pred_img = np.squeeze(np.argmax(pred_prob, axis=1)[-1:,:,:])
            # pred_img_list.append(pred_img)

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

    k, acc = evaluate(pred_msk, gt)
    print('kappa: ', k, 'acc: ', acc)

    # rescale to actual pop values
    ori_unnormed = np.load(proj_dir + 'data/ori_data/pop_pred/input_all_' + str(n_years) + 'y_' + str(n_classes) + 'c_no_na_oh.npy')
    pop_unnormed = ori_unnormed[:, 1, :, :]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(pop_unnormed.reshape(-1, 1))
    pop = scaler.inverse_transform(pred_msk.reshape(-1,1)).reshape(pred_msk.shape[-2], pred_msk.shape[-1])
    



    save_path = proj_dir + 'data/test/pop_pred/pop_No_seed_20y_4c_rand_srch_15-20/lr0.0013723866073356884_bs4/'#.format(pred_seq, model_n,factor_option)

    # save_path = proj_dir + 'data/test/forward/No_seed_convLSTM/No_seed_convLSTM_no_na_normed_clean_tiles/'#.format(pred_seq, model_n,factor_option)
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path + 'pred_msk_eval_normed.npy', pred_msk)
    np.save(save_path + 'pred_msk_eval_rescaled.npy', pop)
    cv2.imwrite(save_path + 'pred_msk_eval.png', pred_msk)
