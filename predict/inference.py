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
from Documents.Masterarbeit.Code.population_prediction.model.v_convlstm import ConvLSTM

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

def color_annotation(image):
    color = np.ones([image.shape[0], image.shape[1], 3])
    color[image == 0] = [0, 0, 0]  # bg
    color[image == 1] = [204, 104, 0]  # water
    color[image == 2] = [0, 153, 0]  # vegatation
    color[image == 3] = [0, 76, 153]  # soil
    color[image == 4] = [192, 192, 192]  # impervious
    color[image == 5] = [102, 0, 204]  # formal
    color[image == 6] = [0, 255, 255]  # informal
    return color

def get_args():
    parser = argparse.ArgumentParser(description='Train ConvLSTM Models', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--n_features', default=7, type=int, dest='n_features')
    parser.add_argument('-s', '--filter_size', default=5, type=int, dest='filter_size')
    parser.add_argument('-b', '--batch_size', default=1, type=int, nargs='?', help='Batch size', dest='batch_size')
    parser.add_argument('-c', '--input_channel', default=15, type=int, dest='input_channel') # [7, 15]
    parser.add_argument('-n', '--n_layer', default=3, type=int, dest='n_layer')
    parser.add_argument('-l', '--seq_len', default=4, type=int, dest='seq_len')
    parser.add_argument('-is', '--input_shape', default=(256, 256), type=tuple, dest='input_shape')
    return parser.parse_args()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    args = get_args()

    bias_status = True

    ori_data_dir = 'C:/Users/jmaie/Documents/Masterarbeit/Code/population_prediction/data/ori_data/lulc_pred/input_all_6y_6c_no_na.npy'

    ori_data = np.load(ori_data_dir)# .transpose((1, 0, 2, 3))
    # scaled_data = torch.nn.functional.interpolate(torch.from_numpy(ori_data),
    #                                               scale_factor=(1 / 3, 1 / 3),
    #                                               recompute_scale_factor=True)
    # processed_ori_data = scaled_data.numpy()
    processed_ori_data = ori_data
    valid_input = processed_ori_data[1:5, :, :, :] # [1:5, :, :, :] = years 2000 - 2020
    gt = processed_ori_data[-1, 0, :, :] # last year, land cover
    print('valid_sequence shape: ', valid_input.shape)

    input_channel = 6 # 19

    df = pd.DataFrame()
    valid_record = {'kappa': 0, 'acc': 0}

    #dir_checkpoint = './ckpts/forecasting/{}_{}/{}/CP_epoch100.pth'
    dir_checkpoint = "C:/Users/jmaie/Documents/Masterarbeit/Code/population_prediction/data/ckpts/forward/No_seed_convLSTM_no_na/with_factors/CP_epoch4.pth"
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

            net = ConvLSTM(input_dim=input_channel,
                          hidden_dim=[32, 16, args.n_features],
                          kernel_size=(3, 3), num_layers=args.n_layer,
                          batch_first=True, bias=bias_status, return_all_layers=False)

            net.to(device)
            net.load_state_dict(torch.load(dir_checkpoint))
            # net.eval() # added from Pytorch tutorial

            output_list = net(test_img[:,:,1:,::]) # 1: for all factors but lc

            masks_pred = output_list[0].view(args.seq_len, args.n_features, 256, 256)
            pred_prob = torch.softmax(torch.squeeze(masks_pred),dim=1).data.cpu().numpy()
            pred_img = np.squeeze(np.argmax(pred_prob, axis=1)[-1:,:,:])
            pred_img_list.append(pred_img)

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

    k, acc = evaluate(gt, pred_msk)
    print('kappa: ', k, 'acc: ', acc)

    save_path = 'C:/Users/jmaie/Documents/Masterarbeit/Code/population_prediction/data/test/forward/No_seed_convLSTM/lulc_6y_6c_no_na/'#.format(pred_seq, model_n,factor_option)
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path + 'pred_msk_eval.npy', pred_msk)
    cv2.imwrite(save_path + 'pred_msk_eval.png', color_annotation(pred_msk))
