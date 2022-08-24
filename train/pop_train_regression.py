# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 12:42:32 2022

@author: jmaie
"""


from torch.utils.data import DataLoader
import pandas as pd
from utilis.dataset import MyDataset
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import csv
import os
import statistics
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


proj_dir = "H:/Masterarbeit/population_prediction/"
# proj_dir = "C:/Users/jmaie/Documents/Masterarbeit/Code/population_prediction/"

# define random choice of hyperparameters
config = {
        "l1": 64, # 2 ** np.random.randint(2, 8), # [4, 8, 16, 32, 64, 128, 256]
        "l2": 'na', # 2 ** np.random.randint(2, 8),
        "lr": 0.0012, # round(np.random.uniform(0.01, 0.00001), 4), # [0.1, 0.00001]
        "batch_size": 6, #random.choice([2, 4, 6, 8]),
        "epochs": 50,
        "model_n" : 'linear_regression_01-16_',
        "save_cp" : True,
        "save_csv" : True,
        "n_years" : 20,
        "n_classes" : 4
    }



ori_data_dir = proj_dir + "data/ori_data/pop_pred/input_all_" + str(config['n_years']) + "y_" + str(config['n_classes'])+ "c_no_na_oh_norm.npy"
ori_data = np.load(ori_data_dir)


### just pop as input for regression
pop = ori_data[:,1,:,:] # all years
pop = pop.reshape(pop.shape[0],-1)
print(pop.shape)
p = pop.transpose(1,0)
print(p.shape)

### multivariate linear regression
# dat = ori_data[:,1:,:,:] # all years, all factors except multiclass layer
# dat = dat.reshape(dat.shape[0], dat.shape[1], -1)
# p = dat.transpose(2,0,1)
# print(p.shape)


# subset = pop[:,pop[0,:] > 0]
# p75 = np.percentile(subset, 99)
# subset = subset[:,subset[0,:] > p75]

# pixel = subset[:,:]
# pixel = np.insert(pixel, 0, range(20), axis = 1)
# train = pixel[0:17, :]
# test = pixel[16:20, :]




###############################################################################
# linear regression per pixel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

rmse_list = []
pred_img_1D = np.zeros((p.shape[0]))


# loop through pixels
for i in tqdm(range(0, p.shape[0])):
    one = p[i,:16]
    y = one#.reshape(-1,1)
    X = np.arange(16).reshape(-1,1)
    #X = one[:,1:]
    #Xnew = np.append(X, Xyears, axis = 1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    ##### linear regression
    reg = LinearRegression()
    reg.fit(X, y)
    interc = reg.intercept_
    coef = reg.coef_
    
    # make predictions:
    def calc(slope, intercept, year):
        return slope*year+intercept
    
    score = calc(coef, interc, 20)
    pred_img_1D[i] = score # save prediction for 2020
    
    
    
    
    # ###### polynomial regression
    # poly = PolynomialFeatures(degree=2, include_bias = False)
    # X_poly = poly.fit_transform(X) # x.reshape(-1,1)
    # # poly.fit(X_poly, y)
    # poly_reg = LinearRegression()
    # poly_reg.fit(X_poly, y)
    
    # y_predicted = poly_reg.predict(X_poly)
    
    # interc = poly_reg.intercept_
    # coef = poly_reg.coef_
    
    # # make predictions:
    # def calc2d(coef, intercept, year):
    #     return coef[0]*year + coef[1]*year**2 + intercept
    
    # score = calc2d(coef, interc, 20)
    # pred_img_1D[i] = score # save prediction for 2020
    
    
    
    # y_pred = reg.predict(X_test) # predict all to get error measures
    
    #mae = mean_absolute_error(y_test, y_pred)
    #mse = mean_squared_error(y_test, y_pred)
    # rmse = mean_squared_error(y_test, y_pred, squared = False)
    
    # rmse_list.append(rmse)


rmse = mean_squared_error(p[:,-1], pred_img_1D, squared = False)
#rmse_mean = statistics.mean(rmse_list)
pred_img = pred_img_1D.reshape(888,888)

plt.imshow(pred_img)
plt.title('prediction of 2020')
plt.show()

print('rmse '+ str(rmse))



# rescale to actual pop values
ori_unnormed = np.load(proj_dir + 'data/ori_data/pop_pred/input_all_20y_4c_no_na_oh.npy')
pop_unnormed = ori_unnormed[:, 1, :, :]
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(pop_unnormed.reshape(-1, 1))
pred_img_rescaled = scaler.inverse_transform(pred_img.reshape(-1,1)).reshape(pred_img.shape[-2], pred_img.shape[-1])



save_path = proj_dir + "data/test/" + config['model_n'] + "/"
os.makedirs(save_path, exist_ok=True)
np.save(save_path + 'pred_msk_eval_normed.npy', pred_img)
np.save(save_path + 'pred_msk_eval_rescaled.npy', pred_img_rescaled)


import tifffile
tifffile.imwrite(save_path + 'pred_msk_normed.tif', pred_img)
tifffile.imwrite(save_path + 'pred_msk_rescaled.tif', pred_img_rescaled)


###############################################################################

# plt.plot(train[:,0], train[:,1:], color = "black")
# plt.plot(test[:,0], test[:,1:], color = "red")
# plt.ylabel('pop')
# plt.xlabel('year')
# plt.xticks(rotation=45)
# plt.title("Train/Test split for pop data")
# plt.show()


# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.arima.model import ARIMA

# ARIMAmodel = ARIMA(train[:,1], order = (2,0,4))
# ARIMAmodel = ARIMAmodel.fit()

# # generate the predictions

# y_pred = ARMIAmodel.get_forecast(4)
# y_pred_df = pd.DataFrame(y_pred.conf_int(alpha = 0.05))
# y_pred_df["Predictions"] = ARMIAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
# y_pred_df.index = test[:,0]
# y_pred_out = y_pred_df["Predictions"] 

# plt.plot(train[:,0], train[:,1:], color = "black")
# plt.plot(test[:,0], test[:,1:], color = "red")
# plt.plot(y_pred_out, color='green', label = 'Predictions')
# plt.legend()




# ###############################################################################
# # multiple models
# from tqdm import tqdm
# from sklearn import metrics
# import statistics

# all_pred = np.zeros((4, pixel.shape[1]))
# rmse_list = []

# for i in tqdm(range(3000, pixel.shape[1])):
    
#     df_subset = pixel[:,i] # one column = one pixel
    
#     ARIMAmodel = ARIMA(df_subset, order = (1,0,0)) # p, q, d
#     ARMIAmodel = ARIMAmodel.fit()
    
#     y_pred = ARIMAmodel.get_forecast(4)
#     y_pred_df = pd.DataFrame(y_pred.conf_int(alpha = 0.05))
#     y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
#     y_pred_df.index = test[:,0]
#     y_pred_out = y_pred_df["Predictions"] 

#     all_pred[:, i] = y_pred_out
#     rmse = metrics.mean_squared_error(test[:,i], y_pred_out, squared = False)
#     rmse_list.append(rmse)

# all_predictions = np.insert(all_pred, 0, test[:,0], axis = 1)

# plt.plot(train[:,0], train[:,1:], color = "black", lw=0.1)
# plt.plot(test[:,0], test[:,1:], color = "red", lw=0.1)
# plt.plot(all_predictions[:,0], all_predictions[:,1:], color='green', label = 'Predictions', lw=0.1)

# rmse_mean = statistics.mean(rmse_list)
# print('rmse_mean: ' + str(rmse_mean))


# # check if the series is stationary. if p value >= the series is non-stationary, d value > 0
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# result = adfuller(pixel[:,10])
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])

# df = pd.DataFrame(pixel[:,10])

# # Original Series
# fig, axes = plt.subplots(3, 2, sharex=True)
# axes[0, 0].plot(df); axes[0, 0].set_title('Original Series')
# plot_acf(df, ax=axes[0, 1])

# # 1st Differencing
# axes[1, 0].plot(df.diff()); axes[1, 0].set_title('1st Order Differencing')
# plot_acf(df.diff().dropna(), ax=axes[1, 1])

# # 2nd Differencing
# axes[2, 0].plot(df.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
# plot_acf(df.diff().diff().dropna(), ax=axes[2, 1])

# plt.show()


# # PACF plot of 1st differenced series
# plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

# fig, axes = plt.subplots(1, 2, sharex=True)
# axes[0].plot(df.diff()); axes[0].set_title('1st Differencing')
# axes[1].set(ylim=(0, 0.5))
# plot_pacf(df.diff(), ax=axes[1], lags = 9)

# plt.show()


