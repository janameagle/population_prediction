# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 12:42:32 2022

@author: jmaie
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error



proj_dir = "H:/Masterarbeit/population_prediction/"
# proj_dir = "C:/Users/jmaie/Documents/Masterarbeit/Code/population_prediction/"

# define random choice of hyperparameters
config = {
        "l1": 64, # 2 ** np.random.randint(2, 8), # [4, 8, 16, 32, 64, 128, 256]
        "l2": 'na', # 2 ** np.random.randint(2, 8),
        "lr": 0.0012, # round(np.random.uniform(0.01, 0.00001), 4), # [0.1, 0.00001]
        "batch_size": 6, #random.choice([2, 4, 6, 8]),
        "epochs": 50,
        "model_n" : 'multilinear_regression_01-16_static',
        "save_cp" : True,
        "save_csv" : True,
        "n_years" : 20,
        "n_classes" : 4
    }



ori_data_dir = proj_dir + "data/ori_data/pop_pred/input_all_" + str(config['n_years']) + "y_" + str(config['n_classes'])+ "c_no_na_oh_norm_buf.npy"
ori_data = np.load(ori_data_dir)



###############################################################################

def linear_reg(ori_data):
    pop = ori_data[:,1,:,:] # all years
    pop = pop.reshape(pop.shape[0],-1)
    p = pop.transpose(1,0)

    
    rmse_list = []
    pred_img_1D = np.zeros((p.shape[0]))
    
    # loop through pixels
    for i in tqdm(range(0, p.shape[0])):
        one = p[i,:16]
        y = one #.reshape(-1,1)
        X = np.arange(16).reshape(-1,1)
        #X = one[:,1:]
        #Xnew = np.append(X, Xyears, axis = 1)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        
        reg = LinearRegression()
        reg.fit(X, y)
        interc = reg.intercept_
        coef = reg.coef_
        
        # make predictions:
        def calc(slope, intercept, year):
            return slope*year+intercept
        
        score = calc(coef, interc, 20)
        pred_img_1D[i] = score # save prediction for 2020

        return pred_img_1D

def 2d_reg(ori_data):
    pop = ori_data[:,1,:,:] # all years
    pop = pop.reshape(pop.shape[0],-1)
    p = pop.transpose(1,0)
    
    
    rmse_list = []
    pred_img_1D = np.zeros((p.shape[0]))
    
    # loop through pixels
    for i in tqdm(range(0, p.shape[0])):
        one = p[i,:16]
        y = one #.reshape(-1,1)
        X = np.arange(16).reshape(-1,1)
        #X = one[:,1:]
        #Xnew = np.append(X, Xyears, axis = 1)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        

        ###### polynomial regression
        poly = PolynomialFeatures(degree=2, include_bias = False)
        X_poly = poly.fit_transform(X) # x.reshape(-1,1)
        # poly.fit(X_poly, y)
        poly_reg = LinearRegression()
        poly_reg.fit(X_poly, y)
        
        y_predicted = poly_reg.predict(X_poly)
        
        interc = poly_reg.intercept_
        coef = poly_reg.coef_
        
        # make predictions:
        def calc2d(coef, intercept, year):
            return coef[0]*year + coef[1]*year**2 + intercept
        
        score = calc2d(coef, interc, 20)
        pred_img_1D[i] = score # save prediction for 2020
        
        return pred_img_1D
    
    
def exp_reg(ori_data):
    pop = ori_data[:,1,:,:] # all years
    pop = pop.reshape(pop.shape[0],-1)
    p = pop.transpose(1,0)
    
    rmse_list = []
    pred_img_1D = np.zeros((p.shape[0]))
    
    # loop through pixels
    for i in tqdm(range(0, p.shape[0])):
        one = p[i,:16]
        y = one #.reshape(-1,1)
        X = np.arange(16).reshape(-1,1)
        #X = one[:,1:]
        #Xnew = np.append(X, Xyears, axis = 1)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        

        X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        np.seterr(divide = 'ignore') 
        a, b = np.polyfit(X, np.log(y), 1) # 1 is the degree of the polynomial
        # y = a*e^{bx} where y = one, x = X
        
        # make predictions:
        def calcexp(a, b, year):
            return math.exp(a)*math.exp(b*year)
        
        score = calcexp(a, b, 20)
        pred_img_1D[i] = score # save prediction for 2020
        
        plt.scatter(X,y)
        plt.plot(X, np.exp(a)*np.exp(b*X))
        plt.show()
        
        y_pred = reg.predict(X_test) # predict all to get error measures
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared = False)
        
        rmse_list.append(rmse)
        
        return pred_img_1D



################ multivariate regression

def multivariate_reg(ori_data):
    dat = ori_data[:,[1,3,4,5,6],:,:] # all years, static features
    dat = dat.reshape(dat.shape[0], dat.shape[1], -1)
    p = dat.transpose(2,0,1)

    
    x = np.zeros((p.shape[0], p.shape[-1]-1 + 16)) # nr of pixels, nr of input years + nr of features
    x[:,:16] = p[:,:16,0] # per pixel yearly values 01-16
    x[:,16:20] = p[:,0,1:] # per pixel static features (same every year, so middle shape can be exchanged)
    y = p[:,-1,0] # per pixel year 20 value as gt
    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)
    
    mlr = LinearRegression()  
    mlr.fit(x_train, y_train)
    
    print("Intercept: ", mlr.intercept_)
    print("Coefficients:")
    list(zip(x, mlr.coef_))
    
    #Prediction of test set
    y_pred_mlr= mlr.predict(x_test)
    
    # rmse = mean_squared_error(y_test, y_pred_mlr)
    
    pred_img_1D = mlr.predict(x)
    
    return pred_img_1D

#################

# pred_img_1D = linear_reg(ori_data)
# pred_img_1D = 2d_reg(ori_data)
pred_img_1D = multivariate_reg(ori_data)


# gt = p[~np.isnan(pred_img_1D), -1]
# pred = pred_img_1D[~np.isnan(pred_img_1D)]
rmse = mean_squared_error(p[:,-1], pred_img_1D, squared = False)
#rmse_mean = statistics.mean(rmse_list)
pred_img = pred_img_1D.reshape(ori_data.shape[-2], ori_data.shape[-1])


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


