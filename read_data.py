# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:39:02 2022

@author: jmaie
"""
# data_dir = "H:/Masterarbeit/Code/population_prediction/"
proj_dir = "H:/Masterarbeit/population_prediction/"
# proj_dir = "C:/Users/jmaie/Documents/Masterarbeit/Code/population_prediction/"




#from PIL import Image
#import numpy as np

#path = "C:/Users/jmaie/Documents/Masterarbeit/Land_cover/MODIS_yearly/land_cover_2001.tif"

#im = Image.open(path)

#arr = np.array(im)
#print(np.unique(arr))

#arr = arr[arr != 0] # 0 are nodata values, remove
#arr[arr == 6] = 7   # closed and open shrub
#arr[arr == 8] = 9   # woody savanna and savanna
#arr[arr == 11] = 17 # wetland and waterbodies

# arr[arr == 10] = ? # grassland
# arr[arr == 12] = ? # cropland

#print(np.unique(arr))

#im_tif = Image.fromarray(arr)
#im_tif.save("C:/Users/jmaie/Documents/Masterarbeit/Land_cover/MODIS_yearly_combined_classes/land_cover_2001_comb.tif")


import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
# import torch


def min_max_scale(img): # (t,c,w,h); channels: lc, pop, urb_dist, slope, streets_dist, water_dist, center_dist, class0mask, class1mask, class2mask, class3mask
    # device = 'cpu'
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_new = np.zeros(img.shape)
    data_new[:,0,:,:] = img[:,0,:,:] # lc not to be normalized

    for i in range(1, img.shape[1]): # for each feature except lc
        temp = img[:,i,:,:].reshape(-1,1)
        scaler.fit(temp)
        new_data = scaler.transform(temp)
        new_data = new_data.reshape(img.shape[0], img.shape[-2], img.shape[-1])
        data_new[:,i,:,:] = new_data

    return np.float32(data_new)



# dat_in = np.load('C:/Users/jmaie/Documents/Masterarbeit/Code/population_prediction/data/18_input.npy')
# dat_targ = np.load('C:/Users/jmaie/Documents/Masterarbeit/Code/population_prediction/data/18_target.npy')

# print(dat_in.shape)
# print(dat_targ.shape)



# im = io.imread('C:/Users/jmaie/Documents/Masterarbeit/Code/population_prediction/data/brick_2001.tif')
# plot single layer as raster
#plt.imshow(im[:,:,2])
#print(im[0,0,:])


# im_resh = im.reshape(7, 888, 888)
# print(im_resh.shape)
# plt.imshow(im_resh[2,:,:])

# im_tr = im.transpose(2,0,1)
# print(im_tr.shape)
# plt.imshow(im_resh[1,:,:])


# reshape to have 7 layers of 888x888 images each
#im_move = np.moveaxis(im, 2, 0)
#print(im_move.shape)
#plt.imshow(im_move[4,:,:])


# dat_multitemp = np.stack([a, b, c], axis=0)
#dat_multitemp = np.array([a, b, c])



# loop over years and stack all the data to retreive an array [20,7,888,888]:
seq = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
# seq = ['01', '04', '08', '12', '16', '20']
# seq = ['15', '16', '17', '18', '19', '20']
n_years = 20
n_classes = 4
dat_multitemp = np.zeros((n_years,7,888,888))

i=0
for y in seq:
   im = io.imread(proj_dir + 'data/yearly_no_na/brick_20' + y + '.tif')
   im_move = np.moveaxis(im, 2, 0)
   dat_multitemp[i,:,:,:] = im_move
   i += 1
   

print(dat_multitemp.shape) # [:,x,:,:] = (pop, urb_dist_r, lc_r, slope_r, streets_dist_r, water_dist_r, center_dist_r)
print(dat_multitemp)
plt.imshow(dat_multitemp[1,2,:,:]) 

# put lc on first place in second axis, for lulc prediction model
lc_multitemp = np.zeros((n_years,7,888,888))
lc_multitemp[:,0,:,:] = dat_multitemp[:,2,:,:]
lc_multitemp[:,1:3,:,:] = dat_multitemp[:,0:2,:,:]
lc_multitemp[:,3:7,:,:] = dat_multitemp[:,3:7,:,:]
print(lc_multitemp)
print(lc_multitemp.shape) # (lc, pop, urb_dist, slope, streets_dist, water_dist, center_dist)
plt.imshow(lc_multitemp[1,0,:,:]) 

# assign new class values
lcnew_multitemp = lc_multitemp
lcnew_multitemp[lc_multitemp == 7] = 0 # shrub
lcnew_multitemp[lc_multitemp == 9] = 0 # savanna
lcnew_multitemp[lc_multitemp == 10] = 0 # grassland
lcnew_multitemp[lc_multitemp == 12] = 0 # croplands
lcnew_multitemp[lc_multitemp == 13] = 1 # urban
lcnew_multitemp[lc_multitemp == 16] = 2 # barren
lcnew_multitemp[lc_multitemp == 17] = 3 # water
np.unique(lcnew_multitemp[:, 0, :, :]) # shape: (t, c, w, h)



# add class masks as input factors
import torch
def oh_code(a, class_n = n_classes):
    oh_list = []
    for i in range(class_n): # for each class
        temp = torch.where(a == i, 1, 0) # binary mask per class
        oh_list.append(temp) # store each class mask as list entry
    return torch.stack(oh_list,0) #torch.stack(oh_list,1) # return array, not list


crop_img_lulc = torch.from_numpy(lcnew_multitemp[:, 0, :, :]) # was not converted to torch before, select lc
temp_list = []
for j in range(crop_img_lulc.shape[0]): # for each year?
    temp = oh_code(crop_img_lulc[j], class_n=n_classes) # array of binary mask per class
    temp_list.append(temp[np.newaxis, :, :, :]) # store class masks per year in list
oh_crop_img_lulc = np.concatenate(temp_list, axis=0)
oh_crop_img = np.concatenate((crop_img_lulc[:,np.newaxis,:,:], lcnew_multitemp[:, 1:, :, :], oh_crop_img_lulc), axis=1)
# oh_crop_img with lc, pop, urb_dist, slope, streets_dist, water_dist, center_dist, class0mask, class1mask, class2mask, class3mask
oh_crop_img = np.float32(oh_crop_img) # for smaller storage size





# normalize values
# lc_multitemp = min_max_scale(lc_multitemp)
# save stacked multitemporal image as numpy data
np.save(proj_dir + 'data/ori_data/pop_pred/input_all_' + str(n_years) +'y_' + str(n_classes) +'c_no_na_oh.npy', oh_crop_img)



# slice the input data image
full_image = np.load(proj_dir + 'data/ori_data/pop_pred/input_all_' + str(n_years) +'y_' + str(n_classes) +'c_no_na_oh.npy')
# full_image = lcnew_multitemp
full_image = min_max_scale(full_image)
# pop_unnormed = full_image[:,1,:,:]
# full_image = np.concatenate((pop_unnormed[:,np.newaxis,:,:], full_image_norm), axis = 1) # t, c, w, h. Channels: pop unnormed, lc unnormed, pop normed, ...
np.save(proj_dir + 'data/ori_data/pop_pred/input_all_' + str(n_years) +'y_' + str(n_classes) +'c_no_na_oh_norm.npy', full_image) # lc unnormed, pop normed, ...
h_total = full_image.shape[-1]
w_total = full_image.shape[-2]
img_size = 256 # how big the tiles should be

    #h_step = int(h_total // img_size * 1.5)                                    # why *1.5?
    #w_step = int(w_total // img_size * 1.5)

x_list = np.linspace(img_size//2, h_total -(img_size//2), num = 14) # Return evenly spaced numbers over a specified interval
y_list = np.linspace(img_size//2, w_total -(img_size//2), num = 14)
new_x_list = []
new_y_list = []

for i in x_list: # new list for integers
    for j in y_list:
        new_x_list.append(int(i))
        new_y_list.append(int(j))


sub_img_list = []

for x, y in zip(new_x_list, new_y_list):
    sub_img = full_image[:, :, x - 128:x + 128, y - 128:y + 128] # get subimage around centroid
    sub_img_list.append(np.float32(sub_img))    
    
print(len(sub_img_list))
print(sub_img_list[1].shape)

dir_input = proj_dir + 'data/train/pop_pred_' + str(n_years) +'y_' + str(n_classes) +'c_no_na_oh_norm/input/'
dir_target = proj_dir + 'data/train/pop_pred_' + str(n_years) +'y_' + str(n_classes) +'c_no_na_oh_norm/target/'
os.makedirs(dir_input, exist_ok=True)
os.makedirs(dir_target, exist_ok=True)

# save all sub images separately
for i in range(len(sub_img_list)):
    np.save(dir_input + str(i) + '_input.npy', sub_img_list[i][-6:,1:,:,:]) # all except lc not normed
    np.save(dir_target + str(i) + '_target.npy', sub_img_list[i][-6:,1,:,:]) # pop normed



# inp = np.load(proj_dir + 'data/18_input.npy')
# targ = np.load(proj_dir + 'data/18_target.npy')

# inp1 = np.load(proj_dir + 'data/train/lulc_pred_6y_4c_no_na/input/20_input.npy')
# targ1 = np.load(proj_dir + 'data/train/lulc_pred_6y_4c_no_na/target/20_target.npy')




