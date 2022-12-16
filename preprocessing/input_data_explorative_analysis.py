# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:39:02 2022

@author: jmaie
"""

"""
explorative analysis of the input data.
"""


proj_dir = "H:/Masterarbeit/population_prediction/"


import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib import cm


# read predicted data
pred_path = proj_dir + 'data/test/pop_pred/pop_No_seed_20y_4c_rand_srch_15-20/lr0.00145_bs2/'#.format(pred_seq, model_n,factor_option)
pred = np.load(pred_path + 'pred_msk_eval_rescaled.npy')



# loop over years and stack all the data to retreive an array [20,7,888,888]:
seq = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
# seq = ['01', '04', '08', '12', '16', '20']
n_years = 20
n_classes = 7
dat_multitemp = np.zeros((n_years,7,888,888))

i=0
y = '01'
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



# differences between pred and gt
pop = lc_multitemp[:,1,:,:]
pop16 = pop[15,:,:]
pop20 = pop[19,:,:]

diff16pred = pred - pop16
diff20pred = pred - pop20



# define colormaps
cmap = cm.bwr
discrete = cm.tab20c
popcmap = cm.OrRd


# plot the differences to see prediction accuracy
fig, axs = plt.subplots(figsize = (15,8))
outer_grid = fig.add_gridspec(4, 4, wspace=0, hspace=0)

ax1 = plt.subplot(241)
p16 = ax1.imshow(pop16[100:750, 0:750], cmap = popcmap, vmin = 0, vmax = 400)
ax1.set_title("Pop 2016")
ax1.set_axis_off()
#fig.colorbar(p16)


ax2 = plt.subplot(242)
p20 = ax2.imshow(pop20[100:750, 0:750], cmap = popcmap, vmin = 0, vmax = 400)
ax2.set_title("Pop 2020")
ax2.set_axis_off()
#fig.colorbar(p20, ax = ax2)

ax3 = plt.subplot(245)
diff16 = ax3.imshow(diff16pred[100:750, 0:750], cmap = cmap, vmin = -40, vmax = 40)
ax3.set_title("Diff pred - 2016")
ax3.set_axis_off()
#fig.colorbar(diff16, ax = ax3)


ax4 = plt.subplot(246)
diff20 = ax4.imshow(diff20pred[100:750, 0:750], cmap = cmap, vmin = -40, vmax = 40)
ax4.set_title("Diff pred - 2020")
ax4.set_axis_off()
fig.colorbar(diff20, ax = [ax3, ax4], location = 'bottom')


ax5 = plt.subplot(122)
pr = ax5.imshow(pred[100:750, 0:750], cmap = popcmap, vmin = 0, vmax = 400)
ax5.set_title("Prediction of 2020")
ax5.set_axis_off()
fig.colorbar(pr, ax = ax5)


plt.show()




# # population changes over years
# change0120 = pop[-1,:,:] - pop[0,:,:]
# change0104 = pop[3,:,:] - pop[0,:,:]
# change0408 = pop[7,:,:] - pop[3,:,:]
# change0812 = pop[11,:,:] - pop[7,:,:]
# change1216 = pop[15,:,:] - pop[11,:,:]
# change1620 = pop[19,:,:] - pop[15,:,:]



# plt.imshow(change0120)
# change0120.min()
# change0120.max()
# p1 = np.percentile(change0120, 1)
# p99 = np.percentile(change0120, 99)




# # plot the change 
# fig, ax = plt.subplots(1, figsize = (10,8))
# ch = plt.imshow(change0120[100:750,0:750], cmap = discrete, vmin = -100, vmax = 100)
# fig.colorbar(ch)
# ax.set(title="Change of population between 2001 and 2020")
# ax.set_axis_off()
# plt.show()



# # plot the change 
# fig, ax = plt.subplots(2,3, figsize = (12,8), constrained_layout = True)
# im = ax[0, 0].imshow(change0104[100:750,0:750], cmap = cmap, vmin = -50, vmax = 50)
# ax[0, 0].set_title("Change 01 - 04")
# ax[0, 0].set_axis_off()

# ax[0, 1].imshow(change0408[100:750,0:750], cmap = cmap, vmin = -50, vmax = 50)
# ax[0, 1].set_title("Change 04 - 08")
# ax[0, 1].set_axis_off()

# ax[0, 2].imshow(change0812[100:750,0:750], cmap = cmap, vmin = -50, vmax = 50)
# ax[0, 2].set_title("Change 08 - 12")
# ax[0, 2].set_axis_off()

# ax[1, 0].imshow(change1216[100:750,0:750], cmap = cmap, vmin = -50, vmax = 50)
# ax[1, 0].set_title("Change 12 - 16")
# ax[1, 0].set_axis_off()

# ax[1, 1].imshow(change1620[100:750,0:750], cmap = cmap, vmin = -50, vmax = 50)
# ax[1, 1].set_title("Change 16 - 20")
# ax[1, 1].set_axis_off()

# total = ax[1, 2].imshow(change0120[100:750,0:750], cmap = cmap, vmin = -100, vmax = 100)
# ax[1, 2].set_title("Change 01 - 20", fontweight ="bold")
# ax[1, 2].set_axis_off()
# fig.colorbar(total)



# color_bar = fig.add_axes([0, 0.15, 0.01, 0.7])
# fig.colorbar(im, cax = color_bar)
# ax[0,0].set_axis_off()
# plt.show()






# # plot the change 
# fig, ax = plt.subplots(1,3, figsize = (15,5), constrained_layout = True)
# first = ax[0].imshow(pop[0,100:750,0:750], cmap = popcmap, vmin = 0, vmax = 600)
# ax[0].set_title("Pop 2001")
# ax[0].set_axis_off()
# fig.colorbar(first, ax = ax[0])

# last = ax[1].imshow(pop[-1,100:750,0:750], cmap = popcmap, vmin = 0, vmax = 600)
# ax[1].set_title("Pop 2020")
# ax[1].set_axis_off()
# fig.colorbar(last, ax = ax[1])

# change = ax[2].imshow(change0120[100:750,0:750], cmap = cmap, vmin = -100, vmax = 100)
# ax[2].set_title("Change 01 - 20", fontweight ="bold")
# ax[2].set_axis_off()
# fig.colorbar(change, ax = ax[2])

# plt.show()





# # plot the change 
# plt.subplots(figsize = (12,8))
# ax1 = plt.subplot(221)
# first = ax1.imshow(pop[0,100:750,0:750], cmap = popcmap, vmin = 0, vmax = 200)
# ax1.set_title("Pop 2001")
# ax1.set_axis_off()


# ax2 = plt.subplot(223)
# last = ax2.imshow(pop[-1,100:750,0:750], cmap = popcmap, vmin = 0, vmax = 200)
# ax2.set_title("Pop 2020")
# ax2.set_axis_off()
# fig.colorbar(first, ax = [ax1, ax2])

# ax3 = plt.subplot(122)
# change = ax3.imshow(change0120[100:750,0:750], cmap = cmap, vmin = -100, vmax = 100)
# ax3.set_title("Change 01 - 20")
# ax3.set_axis_off()
# fig.colorbar(change, ax = ax3)

# plt.show()

