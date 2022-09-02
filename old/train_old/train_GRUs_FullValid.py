from model.v_convgru import ConvGRU
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
from train.options import get_args
from utilis.weight_init import weight_init
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
import numpy as np

def pre_prcessing(crop_img):
    crop_img_lulc = crop_img[:, 0, :, :]
    temp_list = []
    for j in range(crop_img_lulc.shape[0]):
        temp = oh_code(crop_img_lulc[j], class_n=7)
        temp_list.append(temp[np.newaxis, :, :, :])
    oh_crop_img_lulc = np.concatenate(temp_list, axis=0)
    oh_crop_img = np.concatenate((oh_crop_img_lulc, crop_img[:, 1:, :, :]), axis=1)
    return oh_crop_img

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
        temp = torch.where(a == i, 1, 0)
        oh_list.append(temp)
    return torch.stack(oh_list,1)

def get_valid_dataset(ori_data_dir):
    ori_data = np.load(ori_data_dir).transpose((1, 0, 2, 3))
    scaled_data = torch.nn.functional.interpolate(torch.from_numpy(ori_data),
                                                  scale_factor=(1 / 3, 1 / 3),
                                                  recompute_scale_factor=True)
    processed_ori_data = scaled_data.numpy()
    valid_input = processed_ori_data[1:5, :, :, :]
    gt = processed_ori_data[-1, 0, :, :]
    return valid_input, gt

def get_valid_record(valid_input, gt, net, device = torch.device('cuda'), factor_option = 'with_factors'):
    x_list, y_list = get_subsample_centroids(valid_input, img_size=256)
    sub_img_list = []
    for x, y in zip(x_list, y_list):
        sub_img = valid_input[:, :, x - 128:x + 128, y - 128:y + 128]
        sub_img_list.append(sub_img)

    pred_img_list = []
    with torch.no_grad():
        for test_img in sub_img_list:
            test_img = pre_prcessing(test_img)
            test_img = Variable(torch.from_numpy(test_img.copy())).unsqueeze(0).to(device=device,
                                                                                   dtype=torch.float32)

            output_list = net(test_img[:, :, 1:, :, :])
            masks_pred = output_list[0]
            pred_prob = torch.softmax(torch.squeeze(masks_pred), dim=1).data.cpu().numpy()
            pred_img = np.squeeze(np.argmax(pred_prob, axis=1)[-1:, :, :])
            pred_img_list.append(pred_img)
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

    return k, acc


def train_ConvGRU_FullValid(net = ConvGRU, device = torch.device('cuda'),
                  epochs=5, batch_size=1,lr=0.1,
                  save_cp=True, save_csv=True, factor_option='with_factors',
                  pred_seq='forward', model_n='convgru'):

    args = get_args()
    dataset_dir = "../data/train_valid/{}/{}/".format(pred_seq,'dataset_1')
    train_dir = dataset_dir + "train/"
    train_data = MyDataset(imgs_dir = train_dir + 'input/',masks_dir = train_dir +'target/')
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers= 4)

    ori_data_dir = '../data/ori_data/211002_lulcmaps.npy'

    valid_input, gt = get_valid_dataset(ori_data_dir)

    optimizer = optim.Adam(net.parameters(), lr, (0.9, 0.999))
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',factor=0.8, patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss()
    df = pd.DataFrame()

    net.apply(weight_init)

    for epoch in range(0, epochs):

        net.train()
        epoch_loss = 0
        acc = 0
        train_record = {'train_loss': 0, 'train_acc': 0}

        for i, (imgs, true_masks) in enumerate(train_loader):
            imgs = imgs.to(device=device, dtype=torch.float32)
            imgs = Variable(imgs)

            true_masks = Variable(true_masks.to(device=device, dtype=torch.long))

            # lulc classifer
            output_list= net(imgs[:, :, 1:, :, :])
            # output_list = net(imgs)
            masks_pred = output_list[0]
            _, masks_pred_max = torch.max(masks_pred.data, 2)
            loss = criterion(masks_pred.permute(0, 2, 1, 3, 4), true_masks)

            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # get acc
            # _, masks_pred_max = torch.max(masks_pred.data, 2)
            pred_for_acc = masks_pred_max[:,-1,:,:]
            true_masks_for_acc = true_masks[:,-1,:,:]

            corr = torch.sum(pred_for_acc == true_masks_for_acc.detach())
            tensor_size = pred_for_acc.size(0) * pred_for_acc.size(1) * pred_for_acc.size(2)
            acc += float(corr) / float(tensor_size)
            batch_acc = acc/(i+1)

            train_record['train_loss'] += loss.item()
            train_record['train_acc'] += batch_acc

            if i % 5 == 0:
                print('Epoch [{} / {}], batch: {}, train loss: {}, train acc: {}'.format(epoch+1,epochs,i+1,
                                                                                         loss.item(),batch_acc))
        train_record['train_loss'] = train_record['train_loss'] / len(train_loader)
        train_record['train_acc'] = train_record['train_acc'] / len(train_loader)

        print(train_record)
        scheduler.step(batch_acc)
        # scheduler.step()
        # ===================================== Validation ====================================#
        with torch.no_grad():
            net.eval()

            val_record = {'val_kappa': 0, 'val_acc': 0, 'val_QA': 0}
            k, acc, QA = get_valid_record(valid_input, gt, net, factor_option=factor_option)

            val_record['val_kappa'] = k
            val_record['val_acc'] = acc
            val_record['val_QA'] = QA

            print(val_record)

        print('---------------------------------------------------------------------------------------------------------')

        if save_cp:
            dir_checkpoint = "../ckpts/{}/{}/{}/".format(pred_seq, model_n,factor_option)
            os.makedirs(dir_checkpoint, exist_ok=True)
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch}.pth')
            logging.info(f'Checkpoint {epoch} saved !')

        if save_csv:
            train_record.update(val_record)
            record_df = pd.DataFrame(train_record, index=[epoch])
            df = df.append(record_df)
            record_dir = '../record/{}/{}/{}/'.format(pred_seq,factor_option, model_n)
            os.makedirs(record_dir, exist_ok=True)
            df.to_csv(record_dir + '{}_lr{}_layer{}.csv'.format(model_n,args.learn_rate, args.n_layer))


