# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 10:26:05 2022

@author: maie_ja
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from paretoset import paretoset


proj_dir = "D:/Masterarbeit/population_prediction/"

path = proj_dir + 'data/test/'

params6 = 'lr0.0012_bs6_1l64_2lna/'
params2 = 'lr0.0012_bs2_1l64_2lna/'
paramsLSTM = 'lr0.0012_bs1_1l64_2lna/'

models = ['ConvLSTM_02-20_3y_all/' + params6 + 'run1/', # bs6
          'ConvLSTM_02-20_3y_all/' + params6 + 'run2/', # bs6
          'ConvLSTM_02-20_3y_all/' + params2 + 'run3/', 
          # 'ConvLSTM_02-20_3y_all/' + params2 + 'run4/',
          # 'ConvLSTM_02-20_3y_all/' + params2 + 'run5/',
          'ConvLSTM_02-20_3y_static/' + params6 + 'run1/', # bs6
          'ConvLSTM_02-20_3y_static/' + params6 + 'run2/', # bs6
          'ConvLSTM_02-20_3y_static/' + params6 + 'run3/', # bs6
          # 'ConvLSTM_02-20_3y_static/' + params2 + 'run4/',
          # 'ConvLSTM_02-20_3y_static/' + params2 + 'run5/',
          'ConvLSTM_02-20_3y_pop/' + params6 + 'run1/', # bs6
          'ConvLSTM_02-20_3y_pop/' + params6 + 'run2/', # bs6
          'ConvLSTM_02-20_3y_pop/' + params2 + 'run3/',                      
          # 'ConvLSTM_02-20_3y_pop/' + params2 + 'run4/',
          # 'ConvLSTM_02-20_3y_pop/' + params2 + 'run5/',
          'BiConvLSTM_02-20_3y_all/' + params6 + 'run1/', # bs6
          'BiConvLSTM_02-20_3y_all/' + params6 + 'run2/', # bs6
          'BiConvLSTM_02-20_3y_all/' + params6 + 'run3/', # bs6
          # 'BiConvLSTM_02-20_3y_all/' + params2 + 'run4/',
          # 'BiConvLSTM_02-20_3y_all/' + params2 + 'run5/',
          'BiConvLSTM_02-20_3y_static/' + params6 + 'run1/', # bs6
          'BiConvLSTM_02-20_3y_static/' + params2 + 'run2/',    
          'BiConvLSTM_02-20_3y_static/' + params2 + 'run3/',   
          # 'BiConvLSTM_02-20_3y_static/' + params2 + 'run4/',    
          # 'BiConvLSTM_02-20_3y_static/' + params2 + 'run5/',
          'BiConvLSTM_02-20_3y_pop/' + params6 + 'run1/', # bs6       
          'BiConvLSTM_02-20_3y_pop/' + params2 + 'run2/',  
          'BiConvLSTM_02-20_3y_pop/' + params2 + 'run3/',      
          # 'BiConvLSTM_02-20_3y_pop/' + params2 + 'run4/',
          # 'BiConvLSTM_02-20_3y_pop/' + params2 + 'run5/',
          'LSTM_02-20_3y_all/' + paramsLSTM + 'run1/',
          'LSTM_02-20_3y_all/' + paramsLSTM + 'run2/',
          'LSTM_02-20_3y_all/' + paramsLSTM + 'run3/',
          'LSTM_02-20_3y_all/' + paramsLSTM + 'run4/',
          'LSTM_02-20_3y_all/' + paramsLSTM + 'run5/',
          'LSTM_02-20_3y_static/' + paramsLSTM + 'run1/',
          'LSTM_02-20_3y_static/' + paramsLSTM + 'run2/', 
          'LSTM_02-20_3y_static/' + paramsLSTM + 'run3/',
          'LSTM_02-20_3y_static/' + paramsLSTM + 'run4/',   
          'LSTM_02-20_3y_static/' + paramsLSTM + 'run5/',
          'LSTM_02-20_3y_pop/' + paramsLSTM + 'run1/',
          'LSTM_02-20_3y_pop/' + paramsLSTM + 'run2/',
          'LSTM_02-20_3y_pop/' + paramsLSTM + 'run3/',
          'LSTM_02-20_3y_pop/' + paramsLSTM + 'run4/',
          'LSTM_02-20_3y_pop/' + paramsLSTM + 'run5/',
          'BiLSTM_02-20_3y_all/' + paramsLSTM + 'run1/', 
          'BiLSTM_02-20_3y_all/' + paramsLSTM + 'run2/',      
          'BiLSTM_02-20_3y_all/' + paramsLSTM + 'run3/',     
          'BiLSTM_02-20_3y_all/' + paramsLSTM + 'run4/',     
          'BiLSTM_02-20_3y_all/' + paramsLSTM + 'run5/',                       
          'BiLSTM_02-20_3y_static/' + paramsLSTM + 'run1/',   
          'BiLSTM_02-20_3y_static/' + paramsLSTM + 'run2/', 
          'BiLSTM_02-20_3y_static/' + paramsLSTM + 'run3/', 
          'BiLSTM_02-20_3y_static/' + paramsLSTM + 'run4/', 
          'BiLSTM_02-20_3y_static/' + paramsLSTM + 'run5/', 
          'BiLSTM_02-20_3y_pop/' + paramsLSTM + 'run1/', 
          'BiLSTM_02-20_3y_pop/' + paramsLSTM + 'run2/',  
          'BiLSTM_02-20_3y_pop/' + paramsLSTM + 'run3/', 
          'BiLSTM_02-20_3y_pop/' + paramsLSTM + 'run4/', 
          'BiLSTM_02-20_3y_pop/' + paramsLSTM + 'run5/',             
          'multivariate_reg_02-20_3y_all/',
          'multivariate_reg_02-20_3y_static/',
          'random_forest_reg_02-20_3y_all/',
          'random_forest_reg_02-20_3y_static/',
          'random_forest_reg_02-20_3y_pop/',
          'linear_reg_02-20_3y_pop/']

errors = pd.DataFrame(columns=['model_n', 'mae', 'rmse', 'r2', 'r', 'medae'])

for model in models:
    data = pd.read_csv(path + model + 'error_measures_LMA.csv', index_col = 'measure', usecols = ['measure', 'value'])
    # print(data)
    data = data.transpose()
    model_n = model.split('/')[0]
    data['model_n_short'] = model_n.split('_')[0]
    data['feat'] = model_n.split('_')[-1]
    data['model_n'] = data['model_n_short'] + '_' + data['feat']
    errors = pd.concat([errors, data])


# rvalues = errors.loc[:,['model_n', 'r', 'r2']]

errors = errors.drop(['pears_r'], axis = 1)
errors = errors.drop(['r'], axis = 1)
# errors = errors.drop(['r2'], axis = 1)
# errors = errors.set_index('model_n')
# print(errors)
# print(errors['rmse'])


# ax1 = errors.plot.scatter(x='mae',
#                       y='rmse',
#                       c='r2',
#                       colorbar = 'viridis')


# plt.plot(errors)

####################################
# scatterplot

errors.index.name = 'model_n'

# for model_n,row in errors.iterrows():
#   plt.scatter(row['mae'], row['rmse'], label=model_n,  s=100)

# plt.xlabel('MAE')
# plt.ylabel('RMSE')
# plt.legend(loc=(1.04, 0))
# plt.show()


import seaborn as sns

fig,ax = plt.subplots()
sns.scatterplot(data=errors, hue='model_n', x='medae', y='rmse', palette= 'Set2', s=200) #'Spectral'
plt.legend(loc=(1.04, 0))
plt.show()


#######################################
# find pareto optimal sets

# rmse_medae = errors[['rmse', 'medae']].astype(float)
# mask = paretoset(rmse_medae, sense=['min', 'min'])
# paretoset_errors = rmse_medae[mask]

mask = paretoset(errors[['mae','rmse', 'medae', 'r2']].astype(float), sense=['min','min', 'min', 'max'])
paretoset_errors = errors[mask]

fig,ax = plt.subplots()
sns.scatterplot(data=errors, x='medae', y='mae', color='blue', s=200) #'Spectral'
sns.scatterplot(data=paretoset_errors, hue='model_n', x='medae', y='mae', s=200) #'Spectral'
plt.title('Best models according to pareto optimization')
plt.legend(title = None)
plt.show()



######################################
# color per feature set
fig,ax = plt.subplots()
sns.scatterplot(data=errors, x='mae', y='rmse', hue='feat', s=100) 
plt.legend(loc=(1.04, 0))
plt.show()


melt = errors.melt(id_vars=['model_n', 'feat'], value_vars=['mae', 'medae', 'rmse'])
fig,ax = plt.subplots()
sns.boxplot(data=melt, x='value', y='variable', hue='feat')
plt.ylabel('error measure')
plt.title('Error values per feature set')
plt.legend(title='Features')
# sns.boxplot(data=melt, x='value', y='feat')
plt.show()


######################################
# find best runs for each model, from rmse
models = errors.model_n.unique()
best_models = pd.DataFrame()
for model in models:
    temp = errors[errors.model_n == model]
    # best rmse
    min_ = temp[temp.rmse == temp.rmse.min()]
    # mask = paretoset(temp[['mae','rmse', 'medae', 'r2']].astype(float), sense=['min','min', 'min', 'max'])
    # temp_errors = temp[mask]
    best_models = pd.concat([best_models, min_])

print(best_models)

####################################
# barplots
errors = best_models.drop(['r2', 'feat', 'model_n_short'], axis = 1)
errors_m = errors.sort_values('rmse', ascending=False)
errors_m = errors_m.melt(id_vars='model_n', var_name= 'error', value_name = 'value')
errors_m.loc[(errors_m.value < 0), 'value']= -1 
ax = sns.barplot(data = errors_m, y = 'model_n', x = 'value', hue = 'error')
plt.title('Error values per model')
plt.legend(title = 'Error measure')
plt.ylabel(None)
plt.show()


rvalues_m = best_models[['model_n', 'r2']].sort_values('r2', ascending=True)
rvalues_m = rvalues_m.melt(id_vars='model_n', var_name= 'error', value_name = 'value')
rvalues_m.loc[(rvalues_m.value < 0), 'value']= -1 
ax = sns.barplot(data = rvalues_m, y = 'model_n', x = 'value', hue = 'error')
plt.title('R2 values per model')
plt.legend(title = 'Error measure')
plt.ylabel(None)
ax.set_xlim(0.985,1)
plt.show()


