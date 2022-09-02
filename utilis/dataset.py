import numpy as np
import torch
from torch.utils.data import Dataset
import os
from sklearn.preprocessing import MinMaxScaler
#import torchvision.transforms as transforms


# train_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomRotation(degrees = 180),
#     transforms.ToTensor()
# ])


class MyDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir,  model_name = 'none'):  # transform = train_transform,                
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = [os.path.join(imgs_dir, x) for x in os.listdir(imgs_dir)]   # direction of images
        self.msk_ids = [os.path.join(masks_dir, x) for x in os.listdir(masks_dir)]
        # self.transf = transform
        self.model_name = model_name

    def __len__(self): # number of images
        return len(self.ids)

    def __getitem__(self, index):
        if self.model_name == 'pop_01-20_4y':
            img = np.load(self.ids[index])[[0,3,7,11],:,:,:] # 2001-2012, 4y interval
            mask = np.load(self.msk_ids[index])[[3,7,11,15],:,:] # 2004-2016, 4y interval
        
        elif self.model_name == 'pop_05-20_3y':
            img = np.load(self.ids[index])[[4,7,10,13],1:,:,:] # 2005-2014, 3y interval
            mask = np.load(self.msk_ids[index])[[7,10,13,16],:,:] # 2008-2017, 3y interval
        
        elif self.model_name == 'pop_10-20_2y':
            img = np.load(self.ids[index])[[9,11,13,15],1:,:,:] # 2010-2016, 2y interval
            mask = np.load(self.msk_ids[index])[[11,13,15,17],:,:] # 2012-2018, 2y interval
        
        elif self.model_name == 'pop_15-20_1y':
            img = np.load(self.ids[index])[[14,15,16,17],1:,:,:] # 2015-2018, 1y interval
            mask = np.load(self.msk_ids[index])[[15,16,17,18],:,:] # 2016-2019, 1y interval
        
        elif self.model_name == 'pop_02-20_3y':
            img = np.load(self.ids[index])[[1,4,7,10,13],:,:,:] # 2002-2014, 3y interval
            mask = np.load(self.msk_ids[index])[[4,7,10,13,16],:,:] # 2005-2017, 3y interval
            
        elif self.model_name == 'pop_02-20_2y':
            img = np.load(self.ids[index])[[1,3,5,7,9,11,13,15],1:,:,:] # 2002-2016, 2y interval
            mask = np.load(self.msk_ids[index])[[3,5,7,9,11,13,15,17],:,:] # 2004-2018, 2y interval
            
        elif self.model_name == 'pop_01-20_1y':
            img = np.load(self.ids[index])[0:18,1:,:,:] # 2001-2018, 1y interval
            mask = np.load(self.msk_ids[index])[1:19,:,:] # 2002-2019, 1y interval
            
        elif self.model_name == 'pop_01-20_4y_LSTM':
            img = np.load(self.ids[index])[[0,3,7,11],1:,:,:] # 2001-2018, 1y interval
            mask = np.load(self.msk_ids[index])[[3,7,11,15],:,:] # 2002-2019, 1y interval
            
        elif self.model_name == 'pop_only_01-20_1y':
            img = np.load(self.ids[index])[0:18,1,:,:] # 2001-2018, 1y interval
            mask = np.load(self.msk_ids[index])[1:19,:,:] # 2002-2019, 1y interval
            
        elif self.model_name == 'pop_only_01-20_4y':
            img = np.load(self.ids[index])[[0,3,7,11],0,:,:] # 2001-2012, 4y interval
            mask = np.load(self.msk_ids[index])[[3,7,11,15],:,:] 
        
        elif self.model_name == 'pop_01-20_4y_static':
            img = np.load(self.ids[index])[[0,3,7,11],:,:,:] # 2001-2012, 4y interval
            mask = np.load(self.msk_ids[index])[[3,7,11,15],:,:]
            
        elif self.model_name == 'pop_02-20_3y_static':
            img = np.load(self.ids[index])[[1,4,7,10,13],:,:,:] # 2002-2014, 3y interval
            mask = np.load(self.msk_ids[index])[[4,7,10,13,16],:,:] # 2005-2017, 3y interval
            
        elif self.model_name == 'pop_02-20_3y_static_LSTM':
            img = np.load(self.ids[index])[[1,4,7,10,13],:,:,:] # 2002-2014, 3y interval
            mask = np.load(self.msk_ids[index])[[4,7,10,13,16],:,:] # 2005-2017, 3y interval
            
        
        # if self.transf is not None: # create random augmentation -> random subimage
        #     img = self.transf(img)

        #     img_size = img.shape[-1] # width or height
        #     crop_size = 25
        #     w_random = np.random.randint(img_size - crop_size) # one random nr, why imgs-crops?
        #     h_random = np.random.randint(img_size - crop_size)
        #     img = img[:, :, w_random:w_random + crop_size, h_random:h_random + crop_size] # why + crop size?
        #     mask = mask[:, w_random:w_random + crop_size, h_random:h_random + crop_size]
        # else:
        #     img_size = img.shape[-1]
        #     crop_size = 256
        #     w = (img_size - crop_size) // 2
        #     h = (img_size - crop_size) // 2
        #     img = img[:, :, w:w + crop_size, h:h + crop_size]
        #     mask = mask[:, w:w + crop_size, h:h + crop_size]

        img = torch.from_numpy(img.copy()) # creates tensor from numpy array
        mask = torch.from_numpy(mask.copy())

        return img, mask

# def min_max_scale(data): # used where?
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     flat_data = data.reshape(-1, 1)                                            # what are the dimensions?
#     scaler.fit(flat_data.cpu().numpy()) # compute min and max for later use
#     new_data = scaler.transform(flat_data.cpu().numpy()) # compute new range
#     if data.dim()==3:
#         new_data = new_data.reshape(data.size(0),data.size(1),data.size(2))
#     elif data.dim()==4:
#         new_data = new_data.reshape(data.size(0),data.size(1),data.size(2),data.size(3))
#     elif data.dim()==5:
#         new_data = new_data.reshape(data.size(0),data.size(1),data.size(2),data.size(3),data.size(4))
#     else:
#         new_data = new_data

#     return torch.from_numpy(new_data).to(device)



# def min_max_scale(img): # (b,t,c,w,h)
#     device = 'cpu'
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data_new = np.zeros(img.shape)
#     data_new[:,:,0,:,:] = img[:,:,0,:,:]

#     for i in range(1,img.shape[2]):
#         temp = img[:,:,i,:,:].reshape(-1,1)
#         scaler.fit(temp)
#         new_data = scaler.transform(temp)
#         new_data = new_data.reshape(img.shape[0], img.shape[1], img.shape[-2], img.shape[-1])
#         data_new[:,:,i,:,:] = new_data

#     return torch.from_numpy(np.float32(data_new)).to(device)


