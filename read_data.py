# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:39:02 2022

@author: jmaie
"""
# data_dir = "H:/Masterarbeit/Code/population_prediction/"
proj_dir = "H:/Masterarbeit/population_prediction/"
# proj_dir = "C:/Users/jmaie/Documents/Masterarbeit/Code/population_prediction/"


import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os


def min_max_scale(img): # (t,c,w,h); channels: lc, pop, urb_dist, slope, streets_dist, water_dist, center_dist, class0mask, class1mask, class2mask, class3mask
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_new = np.zeros(img.shape)
    data_new[:,0,:,:] = img[:,0,:,:] # lc not to be normalized

    for i in range(1, img.shape[1]): # for each feature except lc
        temp = img[:,i,:,:].reshape(-1,1) # combine all axis to one
        scaler.fit(temp)
        new_data = scaler.transform(temp)
        new_data = new_data.reshape(img.shape[0], img.shape[-2], img.shape[-1]) # reshape to raster data
        data_new[:,i,:,:] = new_data

    return np.float32(data_new)


# loop over years and stack all the data to retreive an array [20,7,888,888]:
seq = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
n_years = 20
n_classes = 4
dat_multitemp = np.zeros((n_years,7,888,888))

i=0
for y in seq:
   im = io.imread(proj_dir + 'data/yearly_no_na/brick_20' + y + '.tif') # pop, urb_dist_r, lc_r, slope_r, streets_dist_r, water_dist_r, center_dist_r
   im_move = np.moveaxis(im, 2, 0) # (w,h,c) -> (c, w, h)
   im_move[[1,2],:,:] = im_move[[2,1],:,:] #switch lc with urban dist
   im_move[[0,1],:,:] = im_move[[1,0],:,:] #switch lc with pop
   temp = im_move[3,:,:]
   temp[temp<0] = 0
   im_move[3,:,:] = temp
   dat_multitemp[i,:,:,:] = im_move # stack all yearly stacks together
   i += 1
   

print(dat_multitemp.shape) # [:,x,:,:] = (lc_r, pop, urb_dist_r, slope_r, streets_dist_r, water_dist_r, center_dist_r)
print(dat_multitemp)
plt.imshow(dat_multitemp[1,2,:,:]) 


# assign new class values, to have only 4 classes
lcnew_multitemp = dat_multitemp
lcnew_multitemp[dat_multitemp == 7] = 0 # shrub -> vegetation
lcnew_multitemp[dat_multitemp == 9] = 0 # savanna -> vegetation
lcnew_multitemp[dat_multitemp == 10] = 0 # grassland -> vegetation
lcnew_multitemp[dat_multitemp == 12] = 0 # croplands -> vegetation
lcnew_multitemp[dat_multitemp == 13] = 1 # urban
lcnew_multitemp[dat_multitemp == 16] = 2 # barren
lcnew_multitemp[dat_multitemp == 17] = 3 # water
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


#### set everything outsite lima region to 0 -> island will be removed from population
oh_crop_img_lima = oh_crop_img.copy()
lima = np.load(proj_dir + 'data/ori_data/Lima_region.npy')
for i in range(20):
    oh_crop_img_lima[i,1,:,:][lima==0] = 0 # pop
    # oh_crop_img_lima[i,3,400:600,:170] = 0 # slope
    # oh_crop_img_lima[i,7,:,:][lima==0] = 0 # class0
    # oh_crop_img_lima[i,8,:,:][lima==0] = 0 # class1
    # oh_crop_img_lima[i,9,:,:][lima==0] = 0 # class2
    # oh_crop_img_lima[i,10,400:600,:170] = 1 # water
        

# save unnormed data
np.save(proj_dir + 'data/ori_data/input_all_unnormed.npy', oh_crop_img_lima)


# full_image = np.load(proj_dir + 'data/ori_data/input_all_unnormed.npy')
full_image = oh_crop_img_lima

# normalize values
full_image = min_max_scale(full_image) # lc as first channel (t,c,w,h)
np.save(proj_dir + 'data/ori_data/input_all.npy', full_image) # lc unnormed, pop normed, ...

# slice the input data
h_total = full_image.shape[-1]
w_total = full_image.shape[-2]
img_size = 256 # how big the tiles should be

# h_step = int(h_total // img_size * 1.5)                                    # why *1.5?
# w_step = int(w_total // img_size * 1.5)

x_list = np.linspace(img_size//2, h_total -(img_size//2), num = 10) # Return evenly spaced numbers over a specified interval
y_list = np.linspace(img_size//2, w_total -(img_size//2), num = 10) # num = w_step, num = 14
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

dir_input = proj_dir + 'data/train/input_less_tiles/'
dir_target = proj_dir + 'data/train/target_less_tiles/'
os.makedirs(dir_input, exist_ok=True)
os.makedirs(dir_target, exist_ok=True)

# save all sub images separately
for i in range(len(sub_img_list)):
    np.save(dir_input + str(i) + '_input.npy', sub_img_list[i][:,1:,:,:]) # all except lc not normed
    np.save(dir_target + str(i) + '_target.npy', sub_img_list[i][:,1,:,:]) # pop normed



# for i in range(16):
#     inp = np.load(proj_dir + 'data/train/input_less_tiles/'+ str(i) +'_input.npy')
#     plt.imshow(inp[0,0,:,:])
#     plt.show()

# targ = np.load(proj_dir + 'data/train/target_less_tiles/10_target.npy')


# inp1 = np.load(proj_dir + 'data/train/lulc_pred_6y_4c_no_na/input/20_input.npy')
# targ1 = np.load(proj_dir + 'data/train/lulc_pred_6y_4c_no_na/target/20_target.npy')




