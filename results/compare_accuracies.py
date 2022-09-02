# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 10:26:05 2022

@author: maie_ja
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


proj_dir = "H:/Masterarbeit/population_prediction/"

path = proj_dir + 'data/test/'

params = 'lr0.0012_bs6_1l64_2lna/'

models = [#'pop_01-20_4y/' + params,
          #'pop_02-20_3y/' + params,
          #'pop_05-20_3y/' + params,
          #'pop_10-20_2y/' + params,
          #'pop_02-20_2y/' + params,
          'pop_02-20_3y_buf/' + params,
          #'pop_01-20_4y_buf/' + params,
          'pop_only_01-20_4y_buf/' + params,
          'pop_only_01_20_1y/' + params,
          #'pop_01_20_4y_LSTM/' + params,
          'pop_02-20_3y_static_buf/' + params,
          'pop_02-20_3y_static_buf_bi/' + params,
          'pop_02-20_3y_static_LSTM_buf/' + params,
          'linear_regression_01-16_buf/']#,
          #'comp_regression_lin_2d_01-16/',
          #'comp_regression_lin_2d_3d_01-16/']

errors = pd.DataFrame(columns=['model_n', 'mae', 'rmse', 'r2', 'ssim'])

for model in models:
    if os.path.exists(path + model + 'error_measures_buf.csv'):
        data = pd.read_csv(path + model + 'error_measures_buf.csv')
    else:
        data = pd.read_csv(path + model + 'error_measures.csv')
    print(data)
    data = data.set_index('measure')
    data = data.loc[:, 'value']
    model_n = model.split('/')[0]
    data['model_n'] = model_n
    data = data.transpose()
    errors = errors.append(data)


errors = errors.set_index('model_n')
print(errors)
print(errors['rmse'])


# ax1 = errors.plot.scatter(x='mae',
#                       y='rmse',
#                       c='r2',
#                       colorbar = 'viridis')


# plt.plot(errors)


errors.index.name = 'model_n'

for model_n,row in errors.iterrows():
  plt.scatter(row['mae'], row['rmse'], label=model_n,  s=100)

plt.xlabel('MAE')
plt.ylabel('RMSE')
plt.legend(loc=(1.04, 0))
plt.show()


import seaborn as sns

fig,ax = plt.subplots()
sns.scatterplot(data=errors, hue='model_n', x='ssim', y='rmse', palette= 'Set2', s=200) #'Spectral'
plt.legend(loc=(1.04, 0))
plt.xlim(0.8, 1)
plt.show()