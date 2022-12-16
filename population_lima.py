# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:33:47 2022

@author: maie_ja
"""

"""
plotting the population change of Lima and Peru over the years.
"""

proj_dir = "D:/Masterarbeit/population_prediction/"

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

# colors
myset2 = ['#0051a2', '#f4777f']
myset3 = ['#0051a2', '#ffd44f', '#93003a']

# import data
data = pd.read_csv(proj_dir + 'data/populationlima.csv', sep=';')
print(data)
data.plot()

# years = list(data.year[:-1].astype(str)).append('2020-estimation')

# change matplotlib fontsize globally
plt.rcParams['font.size'] = 32

# multiple line plot
fig,ax = plt.subplots(figsize=(24,10))
plt.plot('year', 'Peru', data=data[data['year'] <= 2017], color='#0051a2', linewidth=2)
plt.plot('year', 'Peru', data=data[data['year'] >= 2017], color='#0051a2', linewidth=2, linestyle = '--', label='_Hidden')
plt.plot('year','Lima Metropolitan Area',data=data[data['year'] <= 2017] ,color='#f4777f', linewidth=2)
plt.plot('year','Lima Metropolitan Area',data=data[data['year'] >= 2017] ,color='#f4777f', linewidth=2, linestyle = '--', label='_Hidden')
plt.title('Population change Peru and Lima')
plt.legend(title = None)
plt.ylabel('Population')
plt.xlabel(None)
ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda y, pos: '{:,.0f}'.format(y/1000000)+'M'))
plt.show()