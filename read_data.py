# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:39:02 2022

@author: jmaie
"""
proj_dir = "H:/Masterarbeit/Code/population_prediction/"
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
# seq = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
seq = ['01', '04', '08', '12', '16', '20']
dat_multitemp = np.zeros((6,7,888,888))

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
lc_multitemp = np.zeros((6,7,888,888))
lc_multitemp[:,0,:,:] = dat_multitemp[:,2,:,:]
lc_multitemp[:,1:3,:,:] = dat_multitemp[:,0:2,:,:]
lc_multitemp[:,3:7,:,:] = dat_multitemp[:,3:7,:,:]
print(lc_multitemp)
print(lc_multitemp.shape) # (lc, pop, urb_dist, slope, streets_dist, water_dist, center_dist)
plt.imshow(lc_multitemp[1,0,:,:]) 

# assign new class values
lcnew_multitemp = lc_multitemp
lcnew_multitemp[lc_multitemp == 7] = 0 # shrub
lcnew_multitemp[lc_multitemp == 9] = 1 # savanna
lcnew_multitemp[lc_multitemp == 10] = 2 # grassland
lcnew_multitemp[lc_multitemp == 12] = 3 # croplands
lcnew_multitemp[lc_multitemp == 13] = 4 # urban
lcnew_multitemp[lc_multitemp == 16] = 5 # barren
lcnew_multitemp[lc_multitemp == 17] = 6 # water
np.unique(lcnew_multitemp[:, 0, :, :])

# save stacked multitemporal image as numpy data
np.save(proj_dir + 'data/ori_data/lulc_pred/input_all_6y_6c_no_na.npy', lc_multitemp)



# slice the input data image

full_image = lcnew_multitemp
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
    sub_img_list.append(sub_img)    
    
print(len(sub_img_list))
print(sub_img_list[1].shape)

# save all sub images separately
for i in range(len(sub_img_list)):
    np.save(proj_dir + 'data/train/lulc_pred_6y_6c_no_na/input/'+ str(i) + '_input.npy', sub_img_list[1][:,:,:,:])
    np.save(proj_dir + 'data/train/lulc_pred_6y_6c_no_na/target/'+ str(i) + '_target.npy', sub_img_list[1][:,0,:,:])











