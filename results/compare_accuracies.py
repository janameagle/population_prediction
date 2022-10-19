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


proj_dir = "H:/Masterarbeit/population_prediction/"

path = proj_dir + 'data/test/'

params = 'lr0.0012_bs6_1l64_2lna/'
paramsLSTM = 'lr0.0012_bs1_1l64_2lna/'

models = ['ConvLSTM_02-20_3y_all/' + params,
          # 'ConvLSTM_02-20_3y_static/' + params,
          'ConvLSTM_02-20_3y_pop/' + params,
          # 'ConvLSTM_02-20_3y_all_fclayer/' + params,
          'BiConvLSTM_02-20_3y_all/' + params,
          'BiConvLSTM_02-20_3y_static/' + params,
          'BiConvLSTM_02-20_3y_pop/' + params,
          'LSTM_02-20_3y_all/' + paramsLSTM,
          'LSTM_02-20_3y_static/' + paramsLSTM,
          'LSTM_02-20_3y_bi_all/' + paramsLSTM,
          'LSTM_02-20_3y_bi_static/' + paramsLSTM,
          'LSTM_02-20_3y_bi_pop/' + paramsLSTM,
          'LSTM_02-20_3y_pop/' + paramsLSTM,
          # 'GRU_02-20_3y_all/' + paramsLSTM,
          # 'ConvGRU_02-20_3y_all/' + params,
          'multivariate_reg_02-20_3y_all/',
          'multivariate_reg_02-20_3y_static/',
          'random_forest_reg_02-20_3y_all/',
          'random_forest_reg_02-20_3y_static/',
          'random_forest_reg_02-20_3y_pop/',
          'linear_reg_02-20_3y_pop/']

errors = pd.DataFrame(columns=['model_n', 'mae', 'rmse', 'r2', 'r', 'medae'])

for model in models:
    data = pd.read_csv(path + model + 'error_measures_LMA.csv')
    
    # print(data)
    data = data.set_index('measure')
    data = data.loc[:, 'value']
    model_n = model.split('/')[0]
    data['model_n'] = model_n
    data = data.transpose()
    errors = errors.append(data)

rvalues = errors.loc[:,['model_n', 'r', 'r2']]

errors = errors.drop(['pears_r'], axis = 1)
errors = errors.drop(['r'], axis = 1)
errors = errors.drop(['r2'], axis = 1)
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

rvalues.index.name = 'model_n'
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

rmse_medae = errors[['rmse', 'medae']].astype(float)
mask = paretoset(rmse_medae, sense=['min', 'min'])
paretoset_errors = rmse_medae[mask]

mask = paretoset(errors[['mae','rmse', 'medae']].astype(float), sense=['min','min', 'min'])
paretoset_errors = errors[mask]

fig,ax = plt.subplots()
sns.scatterplot(data=errors, x='medae', y='rmse', color='blue', s=200) #'Spectral'
sns.scatterplot(data=paretoset_errors, hue='model_n', x='medae', y='rmse', s=200) #'Spectral'
plt.legend(loc=(1.04, 0))
plt.show()



####################################
# barplot

errors_m = errors.sort_values('medae', ascending=False)
errors_m = errors_m.melt(id_vars='model_n', var_name= 'error', value_name = 'value')
errors_m.loc[(errors_m.value < 0), 'value']= -1 
ax = sns.barplot(data = errors_m, y = 'model_n', x = 'value', hue = 'error')
plt.show()


rvalues_m = rvalues.sort_values('r', ascending=True)
rvalues_m = rvalues_m.melt(id_vars='model_n', var_name= 'error', value_name = 'value')
rvalues_m.loc[(rvalues_m.value < 0), 'value']= -1 
ax = sns.barplot(data = rvalues_m, y = 'model_n', x = 'value', hue = 'error')
ax.set_xlim(0.985,1)
plt.show()

