# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 12:42:32 2022

@author: jmaie
"""

"""
In this script, the simple linear regression, multivariate linear regression 
and random forest regression are trained to be used as baseline models.
Their results will be compared to the neural network results.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import tifffile


proj_dir = "H:/Masterarbeit/population_prediction/"

# define parameters
config = {
        "model_n" : '02-17_3y',
        "save_cp" : True,
        "save_csv" : True,
        "factors" : 'static', # 'all', 'static', 'pop'
        "model" :'random_forest_reg' # 'linear_reg', 'multivariate_reg', 'random_forest_reg'
    }


interval = int(config['model_n'][-2])
ori_data_dir = proj_dir + "data/ori_data/input_all.npy"
ori_data = np.load(ori_data_dir)

pop20 = ori_data[-1,1,:,:]

###############################################################################


################ simple linear regression

# pixel wise linear regression with population as single input
def linear_reg(ori_data):
    p = ori_data[:,1,:,:].reshape(ori_data.shape[0],-1).transpose(1,0) # (t,c,w,h) -> (w*h, t), population only

    pred_img_1D = np.zeros((p.shape[0]))
    
    # loop through pixels
    for i in tqdm(range(0, p.shape[0])):
        y = p[i,:-1] # one pixel, last year pop
        X = np.arange(p.shape[1]-1).reshape(-1,1) # nr of years
        
        reg = LinearRegression()
        reg.fit(X, y)
        interc = reg.intercept_
        coef = reg.coef_
        
        # make predictions:
        def calc(slope, intercept, year):
            return slope*year+intercept
        
        score = calc(coef, interc, p.shape[1]) # nr years incl 20. (to predict year 20)
        pred_img_1D[i] = score # save prediction for 2020

    return pred_img_1D






################ multivariate regression

def multivariate_reg(ori_data):
    p = ori_data.reshape(ori_data.shape[0], ori_data.shape[1], -1).transpose(2,0,1) # (t, c, w, h) -> (w*h, t, c)

    years = p.shape[1] - 2 # (last two years for prediction)    
    
    if config["factors"] == 'all':
        channels = p.shape[-1] - 2 # without population, lc per years
        x = np.zeros((p.shape[0], channels + years*3)) # nr of pixels, nr of features + input years
        x[:,:years] = p[:,:years,1] # per pixel pop values of 3y intervall years
        x[:,years:years + channels] = p[:,0,2:] # per pixel static features (same every year, so middle shape can be exchanged) 
        for i in range(years):
            x[:,years+channels+i] = p[:,i,0]  # lc per year
            x[:,years*2+channels+i] = p[:,i,2]  # distance to urban extent per year
        feature_names = ['02','05','08','11','14', 'dist_ext', 'slope', 'roads', 'water', 'center', 'lc02', 'lc05', 'lc08', 'lc11', 'lc14', 'ext02', 'ext05', 'ext08', 'ext11', 'ext14'] #'cl_bare', 'cl_urban', 'cl_grass', 'cl_water']
        y = p[:,-2,1] # per pixel second last year value as train gt


    if config["factors"] == 'static':
        channels = p.shape[-1] - 1 # without population
        x = np.zeros((p.shape[0], channels + years)) # nr of pixels, nr of features + input years
        x[:,:years] = p[:,:years,0] # per pixel pop values of 3y intervall years
        x[:,years:years + channels] = p[:,0,1:] # per pixel static features (same every year, so middle shape can be exchanged) 
        feature_names = ['02','05','08','11','14', 'slope', 'roads', 'water', 'center'] # for plotting
        y = p[:,-2,0] # per pixel second last year value as train gt

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)
    
    reg = LinearRegression()  
    # reg.fit(x_train, y_train)
    reg.fit(x, y)

           
    
    #Prediction of test set
    # y_pred_mlr= reg.predict(x_test)
    # rmse = metrics.mean_squared_error(y_test, y_pred_mlr)
    # print(rmse)
    
    # predict 2020
    if config["factors"] == 'all':
        val_data = np.zeros((p.shape[0], channels + years*3)) # nr of pixels, nr of features + input years
        val_data[:,:years] = p[:,1:years+1,1] # per pixel pop values of 3y intervall years
        val_data[:,years:years + channels] = p[:,0,2:] # per pixel static features (same every year, so middle shape can be exchanged) 
        for i in range(years):
            val_data[:,years+channels+i] = p[:,i+1,0]  # lc per year
            val_data[:,years*2+channels+i] = p[:,i+1,2]  # distance to urban extent per year
        
    if config["factors"] == 'static':
        val_data = np.zeros((p.shape[0], channels + years)) # nr of pixels, nr of input years + nr of features
        val_data[:,:years] = p[:,1:years+1,0] # per pixel pop values of 3y intervall years
        val_data[:,years:] = p[:,0,1:] # per pixel static features
   
    pred_img_1D = reg.predict(val_data)
    
    return pred_img_1D, reg





########################################

def random_forest_reg(ori_data):
    p = ori_data.reshape(ori_data.shape[0], ori_data.shape[1], -1).transpose(2,0,1) # (t, c, w, h) -> (w*h, t, c)

    years = p.shape[1] - 2 # (last two years for prediction)    
    
    if config["factors"] == 'all':
        channels = p.shape[-1] - 2 # without population, lc per years
        x = np.zeros((p.shape[0], channels + years*3)) # nr of pixels, nr of features + input years
        x[:,:years] = p[:,:years,1] # per pixel pop values of 3y intervall years
        x[:,years:years + channels] = p[:,0,2:] # per pixel static features (same every year, so middle shape can be exchanged) 
        for i in range(years):
            x[:,years+channels+i] = p[:,i,0]  # lc per year
            x[:,years*2+channels+i] = p[:,i,2]  # distance to urban extent per year
        feature_names = ['02','05','08','11','14', 'dist_ext', 'slope', 'roads', 'water', 'center', 'lc02', 'lc05', 'lc08', 'lc11', 'lc14', 'ext02', 'ext05', 'ext08', 'ext11', 'ext14'] #'cl_bare', 'cl_urban', 'cl_grass', 'cl_water']
        y = p[:,-2,1] # per pixel second last year value as train gt


    if config["factors"] == 'static':
        channels = p.shape[-1] - 1 # without population
        x = np.zeros((p.shape[0], channels + years)) # nr of pixels, nr of features + input years
        x[:,:years] = p[:,:years,0] # per pixel pop values of 3y intervall years
        x[:,years:years + channels] = p[:,0,1:] # per pixel static features (same every year, so middle shape can be exchanged) 
        feature_names = ['02','05','08','11','14', 'slope', 'roads', 'water', 'center']
        y = p[:,-2,0] # per pixel second last year value as train gt


    if config["factors"] == 'pop':
        x = p[:,:years,0] # per pixel pop values of 3y intervall years
        feature_names = ['02','05','08','11','14']
        y = p[:,-2,0] # per pixel second last year value as train gt
        
       # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)
    
    
    # Initializing the Random Forest Regression model with 250 trees
    reg = RandomForestRegressor(n_estimators = 250, random_state = 0, verbose = 1)

    # Fitting the Random Forest Regression model to the data
    # reg.fit(x_train, y_train)
    reg.fit(x, y)
    
    
    # check feature importances
    print(reg.feature_importances_)
    plt.barh(feature_names, reg.feature_importances_)
    plt.show()    
    
    
    #Prediction of test set
    # y_pred_mlr= reg.predict(x_test)
    # rmse = metrics.mean_squared_error(y_test, y_pred_mlr)
    # print(rmse)
    
    
    # predict 2020
    if config["factors"] == 'all':
        val_data = np.zeros((p.shape[0], channels + years*3)) # nr of pixels, nr of features + input years
        val_data[:,:years] = p[:,1:years+1,1] # per pixel pop values of 3y intervall years
        val_data[:,years:years + channels] = p[:,0,2:] # per pixel static features (same every year, so middle shape can be exchanged) 
        for i in range(years):
            val_data[:,years+channels+i] = p[:,i+1,0]  # lc per year
            val_data[:,years*2+channels+i] = p[:,i+1,2]  # distance to urban extent per year
        
    if config["factors"] == 'static':
        val_data = np.zeros((p.shape[0], channels + years)) # nr of pixels, nr of input years + nr of features
        val_data[:,:years] = p[:,1:years+1,0] # per pixel pop values of 3y intervall years
        val_data[:,years:] = p[:,0,1:] # per pixel static features
        
        
    if config["factors"] == 'pop':
        val_data = p[:,1:years+1,0] # per pixel pop values of 3y intervall years

   
    pred_img_1D = reg.predict(val_data)
    
    return pred_img_1D, reg



###############################################################################
  
# indices for input years depending on interval
input_years = list(reversed(range(20, 1, -interval))) # list of years with interval
input_years = [value -1 for value in input_years] # according index to refer to the year

# select relevant input factors
input_data = ori_data[:,:-4,:,:] if config["factors"] == 'all' else ori_data[:,[1,3,4,5,6],:,:] # if all, no oh-lc layers
# select relevant input years
input_data = input_data[input_years,:,:,:]


if config["model"] == 'linear_reg':
    pred = linear_reg(input_data)
elif config["model"] == 'multivariate_reg':
    pred, model = multivariate_reg(input_data)
elif config["model"] == 'random_forest_reg':
    pred, model = random_forest_reg(input_data)


pred_img = pred.reshape(ori_data.shape[-2], ori_data.shape[-1])

mae = round(metrics.mean_absolute_error(pred_img, pop20), 3)
rmse = round(metrics.mean_squared_error(pred_img, pop20, squared = False), 3)
r2 = round(metrics.r2_score(pred_img, pop20), 3)



# plot prediction and print errors
plt.imshow(pred_img)
plt.title('prediction of 2020')
plt.show()

print('mae '+ str(mae))
print('rmse '+ str(rmse))
print('r2 '+ str(r2))



# rescale to actual pop values
ori_unnormed = np.load(proj_dir + 'data/ori_data/input_all_unnormed.npy')
pop_unnormed = ori_unnormed[:, 1, :, :]
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(pop_unnormed.reshape(-1, 1))
pred_img_rescaled = scaler.inverse_transform(pred_img.reshape(-1,1)).reshape(pred_img.shape[-2], pred_img.shape[-1])


# save prediction
save_path = proj_dir + "data/test/{}_{}_{}/".format(config['model'], config['model_n'], config['factors'])
os.makedirs(save_path, exist_ok=True)
np.save(save_path + 'pred_msk_eval_normed.npy', pred_img)
np.save(save_path + 'pred_msk_eval_rescaled.npy', pred_img_rescaled)


tifffile.imwrite(save_path + 'pred_msk_normed.tif', pred_img)
tifffile.imwrite(save_path + 'pred_msk_rescaled.tif', pred_img_rescaled)


