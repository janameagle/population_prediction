from transformer.ViT_ensemble import ensemble_ViT
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
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score

def pre_prcessing(crop_img): # crop_img is sub img of valid dataset
    crop_img_lulc = crop_img[:, 0, :, :]                                       # 0 = land cover?
    temp_list = []
    for j in range(crop_img_lulc.shape[0]): # for each year?
        temp = oh_code(crop_img_lulc[j], class_n=7) # sequence of tensors      # what does 'oh' stand for? 
        temp_list.append(temp[np.newaxis, :, :, :])
    oh_crop_img_lulc = np.concatenate(temp_list, axis=0) # Join a sequence of arrays along an existing axis
    oh_crop_img = np.concatenate((oh_crop_img_lulc, crop_img[:, 1:, :, :]), axis=1) # 1: refers to all other driving factors?
    return oh_crop_img                                                         # how does output look like? binary mask per each class, for each year, stacked together with driving factors

def evaluate(pred, gt):
    k_statistics = cohen_kappa_score(gt.astype(np.int64).flatten(), pred.astype(np.int64).flatten())
    acc_score = accuracy_score(gt.astype(np.int64).flatten(), pred.astype(np.int64).flatten())

    return k_statistics,acc_score

def get_subsample_centroids(img, img_size=50): # img_size will be overwritten (with 256) when used
    h_total = img.shape[-2] # height 1200                                      # function used only for valid dataset / gt dataset? not for training (split already)
    w_total = img.shape[-1] # width 1200

    h_step = int(h_total // img_size * 1.5)                                    # why *1.5?
    w_step = int(w_total // img_size * 1.5)

    x_list = np.linspace(img_size//2, h_total-img_size//2, num = h_step) # Evenly spaced numbers over interval
    y_list = np.linspace(img_size//2, w_total -img_size//2, num= w_step)

    new_x_list = []
    new_y_list = []

    for i in x_list: # new list for integers
        for j in y_list:
            new_x_list.append(int(i))
            new_y_list.append(int(j))
    return new_x_list, new_y_list



def oh_code(a, class_n = 7):                                                   # musst be changed bc of regression, not classification?
    oh_list = []
    for i in range(class_n):
        temp = torch.where(a == i, 1, 0) # binary mask per class
        oh_list.append(temp)             # list of binary mask per class
    return torch.stack(oh_list,1)        # sequence of tensors

def get_valid_dataset(ori_data_dir):
    ori_data = np.load(ori_data_dir).transpose((1, 0, 2, 3))                   # driving factors,year,width,height?
    # Down/up samples the input to the given scale_factor
    scaled_data = torch.nn.functional.interpolate(torch.from_numpy(ori_data),
                                                  scale_factor=(1 / 3, 1 / 3), # multiplier for spatial size, why?
                                                  recompute_scale_factor=True)
    processed_ori_data = scaled_data.numpy()
    valid_input = processed_ori_data[1:5, :, :, :]                             # 1:5 = years?
    gt = processed_ori_data[-1, 0, :, :]                                       # gt = ground truth? i.e. -1 last year 2020, 0 land cover
    return valid_input, gt


# for validation. validation dataset (whole images) are used and split
def get_valid_record(valid_input, gt, net, device = torch.device('cuda'), factor_option = 'with_factors'): # where is 'with factors' used? what is it?
    x_list, y_list = get_subsample_centroids(valid_input, img_size=256)        # validation tile size 256? training data is 128?
    sub_img_list = []
    for x, y in zip(x_list, y_list):
        sub_img = valid_input[:, :, x - 128:x + 128, y - 128:y + 128] # get subimage around centroid
        sub_img_list.append(sub_img)

    pred_img_list = []
    with torch.no_grad(): #disabled gradient calculation
        for test_img in sub_img_list:
            test_img = pre_prcessing(test_img)                                 # binary masks per class stacked with driving factors?
            test_img = Variable(torch.from_numpy(test_img.copy())).unsqueeze(0).to(device=device,
                                                                                   dtype=torch.float32) # create tensor, remove dimensions with 1
            output_list = net(test_img)                                        # is this the prediction of the ensemble_ViT?
            masks_pred = output_list[0]                                        # [0] = 'out' from ensemble_ViT
            pred_prob = torch.softmax(torch.squeeze(masks_pred), dim=0).data.cpu().numpy() # softmax for range in [0,1] = probability
            pred_img = np.argmax(pred_prob, axis=0)                            # why? Returns the indices of the maximum values along an axis
            pred_img_list.append(pred_img)
    pred_msk = np.zeros((valid_input.shape[-2], valid_input.shape[-1]))

    h = 0 # used to iteratively go through pred_img_list
    x_list, y_list = get_subsample_centroids(valid_input, img_size=256)
    for x, y in zip(x_list, y_list):
        if x == np.min(x_list) or x == np.max(x_list) or y == np.min(y_list) or y == np.max(y_list):
            pred_msk[x - 128:x + 128, y - 128:y + 128] = pred_img_list[h]
            h += 1
        else:
            pred_msk[x - 120:x + 120, y - 120:y + 120] = pred_img_list[h][8:248, 8:248] # why? what?
            h += 1

    k, acc = evaluate(gt, pred_msk)

    return k, acc



def train_ensemble_ViT_FullValid(net = ensemble_ViT, device = torch.device('cuda'),
                  epochs=5, batch_size=1,lr=0.1,
                  save_cp=True, save_csv=True, factor_option='with_factors',
                  pred_seq='forward', model_n='convgru', changemap = False):   # why/when pred_seq not forward?

    args = get_args()
    dataset_dir = "../data/train_valid/{}/{}/".format(pred_seq,'dataset_1')
    train_dir = dataset_dir + "train/"
    train_data = MyDataset(imgs_dir = train_dir + 'input/',masks_dir = train_dir +'target/')          # input: stacked driving factors, target: yearly lc?
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers= 4) # what is the output of Dataloader?

    ori_data_dir = '../data/ori_data/211002_update_lulcmaps_withFactors_updated.npy'                  # what factors?

    valid_input, gt = get_valid_dataset(ori_data_dir)                          # valid_input = driving factors, full image, years 2000-2015?, gt = land cover 2020?

    optimizer = optim.Adam(net.parameters(), lr, (0.9, 0.999)) # performs stochastic gradient descent to update network weights iterative based in training data
    # Reduce learning rate when a metric has stopped improving
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',factor=0.8, patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss()

    df = pd.DataFrame()

    net.apply(weight_init) # what does weight_init do? Apply weights

    for epoch in range(epochs):

        net.train()
        epoch_loss = 0
        acc = 0
        train_record = {'train_loss': 0, 'train_acc': 0}

        for i, (imgs, true_masks) in enumerate(train_loader):                  
            imgs = imgs.to(device=device, dtype=torch.float32)
            imgs = Variable(imgs)                                              # Variable wraps a PyTorch Tensor and records operations applied to it
            true_masks = Variable(true_masks.to(device=device, dtype=torch.long))

            # lulc classifer
            output_list= net(imgs)                                             # is that the prediction?
            masks_pred = output_list[0]

            # outputs in regression tasks, for example, are numbers
            loss = criterion(masks_pred, true_masks[:,-1,:,:]) # defined as CrossEntropyLoss(). calculate how accurate our model is by defining the difference between the estimated probability with our desired outcome.

            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # get acc
            _, masks_pred_max = torch.max(masks_pred.data, 1)
            pred_for_acc = masks_pred_max
            true_masks_for_acc = true_masks[:,-1,:,:]                          # lc 2020?

            corr = torch.sum(pred_for_acc == true_masks_for_acc.detach())      # why ==? what does detach() do?
            tensor_size = pred_for_acc.size(0) * pred_for_acc.size(1) * pred_for_acc.size(2)
            acc += float(corr) / float(tensor_size)
            batch_acc = acc/(i+1)

            train_record['train_loss'] += loss.item() # train_record carries loss and acc info
            train_record['train_acc'] += batch_acc

            if i % 5 == 0:
                print('Epoch [{} / {}], batch: {}, train loss: {}, train acc: {}'.format(epoch+1,epochs,i+1,
                                                                                         loss.item(),batch_acc))

        train_record['train_loss'] = train_record['train_loss'] / len(train_loader) # why do that?
        train_record['train_acc'] = train_record['train_acc'] / len(train_loader)

        print(train_record)
        scheduler.step(batch_acc)
        # scheduler.step()
        # ===================================== Validation ====================================#
        with torch.no_grad(): # Context-manager that disabled gradient calculation. saves computational power
            net.eval()

            val_record = {'val_kappa': 0, 'val_acc': 0}
            k, acc = get_valid_record(valid_input, gt, net, factor_option=factor_option) # valid_input from get_valid_dataset

            val_record['val_kappa'] = k
            val_record['val_acc'] = acc

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
            df.to_csv(record_dir + '{}_lr{}_layer{}.csv'.format(model_n, lr, args.n_layer))


