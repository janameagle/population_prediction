import numpy as np
import torch
from torch.utils.data import Dataset
import os
from sklearn.preprocessing import MinMaxScaler


class MyDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, augment = False):                  # what data are the images, and what the mask?
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = [os.path.join(imgs_dir, x) for x in os.listdir(imgs_dir)]   # direction of images
        self.msk_ids = [os.path.join(masks_dir, x) for x in os.listdir(masks_dir)]
        self.aug = augment

    def __len__(self): # number of images
        return len(self.ids)

    def __getitem__(self, index):
        img = np.load(self.ids[index])
        mask = np.load(self.msk_ids[index])

        if self.aug == True: # create random augmentation -> random subimage
            img_size = img.shape[-1] # which dimension?
            crop_size = 256
            w_random = np.random.randint(img_size - crop_size) # one random nr, why imgs-crops?
            h_random = np.random.randint(img_size - crop_size)
            img = img[:, :, w_random:w_random + crop_size, h_random:h_random + crop_size] # why + crop size?
            mask = mask[:, w_random:w_random + crop_size, h_random:h_random + crop_size]
        else:
            img_size = img.shape[-1]
            crop_size = 256
            w = (img_size - crop_size) // 2
            h = (img_size - crop_size) // 2
            img = img[:, :, w:w + crop_size, h:h + crop_size]
            mask = mask[:, w:w + crop_size, h:h + crop_size]

        img = torch.from_numpy(img.copy()) # creates tensor from numpy array
        mask = torch.from_numpy(mask.copy())

        return img, mask

def min_max_scale(data): # used where?
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = MinMaxScaler(feature_range=(0, 1))
    flat_data = data.reshape(-1, 1)                                            # what are the dimensions?
    scaler.fit(flat_data.cpu().numpy()) # compute min and max for later use
    new_data = scaler.transform(flat_data.cpu().numpy()) # compute new range
    if data.dim()==3:
        new_data = new_data.reshape(data.size(0),data.size(1),data.size(2))
    elif data.dim()==4:
        new_data = new_data.reshape(data.size(0),data.size(1),data.size(2),data.size(3))
    elif data.dim()==5:
        new_data = new_data.reshape(data.size(0),data.size(1),data.size(2),data.size(3),data.size(4))
    else:
        new_data = new_data

    return torch.from_numpy(new_data).to(device)






