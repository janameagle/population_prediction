# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:30:50 2022

@author: maie_ja
"""

import numpy as np
import matplotlib.pyplot as plt

pred_dir = 'H:/Masterarbeit/population_prediction/data/test/No_seed_convLSTM_20y_4c_no_na_oh_norm_random_search_15-20/lr0.006973090246172593_bs4/pred_msk_eval.npy'
pred = np.load(pred_dir)

print(pred)
print(pred.shape)

ori_data_dir = 'H:/Masterarbeit/population_prediction/data/ori_data/lulc_pred/input_all_20y_4c_no_na_oh_norm.npy'
ori_data = np.load(ori_data_dir)

print(ori_data.shape)

lc19 = ori_data[-2,0,:,:]
lc20 = ori_data[-1,0,:,:]

plt.imshow(lc19)
plt.imshow(lc20)

# check accuracy to 2019 and 2020 layer
# overall accuracy
total_cells = lc19.shape[-1] * lc19.shape[-2]

pred_19_sync = pred[pred == lc19]
pred_19_acc = pred_19_sync.shape[0] / total_cells
print(pred_19_acc)

pred_20_sync = pred[pred == lc20]
pred_20_acc = pred_20_sync.shape[0] / total_cells
print(pred_20_acc)

# accuracy of urban layer
lc19_urb = lc19[lc19 == 1].shape[0]

pred_19_sync_urb = pred[np.where((pred == 1) & (pred == lc19))] # urban in pred and in lc20
pred_19_acc_urb = pred_19_sync_urb.shape[0] / lc19_urb
print(pred_19_acc_urb)


lc20_urb = lc20[lc20 == 1].shape[0]

pred_20_sync_urb = pred[np.where((pred == 1) & (pred == lc20))] # urban in pred and in lc20
pred_20_acc_urb = pred_20_sync_urb.shape[0] / lc20_urb
print(pred_20_acc_urb)




# changes between lc19 and lc20
change = np.zeros((888,888))
change[lc20 != lc19] = lc20[lc20 != lc19]
print(change.shape)
plt.imshow(change)
newurb = np.zeros((888,888))
newurb[change == 1] = 1
plt.imshow(newurb)
newurban = change[change == 1]
len(newurban)


urban = np.zeros((888,888))
urban[pred == 1] = 1
urban[pred != 1] = 2
correct_urban = urban[urban == newurban]
len(correct_urban)



# change check for all classes
change = np.zeros((888,888))
change += 99
change[lc20 != lc19] = lc20[lc20 != lc19] # values that are new in lc20

n_classes = 4
total = np.zeros((n_classes, 4))
for i in range(n_classes):
    new_class = np.zeros((888,888))
    new_class[change == i] = 1 # binary mask where this class is new
    new_count = len(new_class[new_class == 1])
    
    pred_class = np.zeros((888,888))
    pred_class[pred == i] = 1
    pred_class[pred != i] = 99 # binary mask where this class was predicted - 99 instead of 0
    
    correct_class_pred_count = len(pred_class[pred_class == new_class]) # number of pixels where this class was predicted correctly
    
    total[i,0] = new_count
    total[i,1] = correct_class_pred_count
    total[i,2] = (correct_class_pred_count / new_count) * 100
    

total = total.astype(int)
print(total)

# create barplot
barWidth = 0.25
# plt.bar(np.arange(n_classes), total[:,0], width = barWidth)
# plt.bar(np.arange(n_classes), total[:,1], width = barWidth)
plt.bar(np.arange(n_classes), total[:,2], width = barWidth)
plt.ylabel('Percentage correct prediction')
plt.title('Prediction of new class pixels')
plt.xticks([r for r in range(n_classes)],
           ['Veg', 'Urban', 'Bare', 'Water'])

plt.legend()
plt.show()




constant = np.zeros((888,888))
constant += 99
constant[lc20 == lc19] = lc20[lc20 == lc19] # values that are new in lc20

for i in range(n_classes):
    same_class = np.zeros((888,888))
    same_class[constant == i] = 1 # binary mask where this class is new
    new_count = len(same_class[same_class == 1])
    
    pred_class = np.zeros((888,888))
    pred_class[pred == i] = 1
    pred_class[pred != i] = 99 # binary mask where this class was predicted - 99 instead of 0
    
    correct_class_pred_count = len(pred_class[pred_class == same_class]) # number of pixels where this class was predicted correctly
    
    total[i,0] = new_count
    total[i,1] = correct_class_pred_count
    total[i,2] = (correct_class_pred_count / new_count) * 100
    total[i,3] = 100 - total[i,2]   

total = total.astype(int)
print(total)


# create barplot
barWidth = 0.25
plt.bar(np.arange(n_classes), total[:,0], width = barWidth)
plt.bar(np.arange(n_classes), total[:,1], width = barWidth)
# plt.bar(np.arange(n_classes), total[:,2], width = barWidth)
# plt.bar(np.arange(n_classes), total[:,3], width = barWidth)
plt.ylabel('Correct prediction')
plt.title('Prediction of same class pixels')
plt.xticks([r for r in range(n_classes)],
           ['Veg', 'Urban', 'Bare', 'Water'])

plt.legend()
plt.show()
















