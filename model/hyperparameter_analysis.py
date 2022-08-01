# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 16:31:30 2022

@author: maie_ja
"""

import os
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import plotly.express as px



proj_dir = "H:/Masterarbeit/population_prediction/"
# proj_dir = "C:/Users/jmaie/Documents/Masterarbeit/Code/population_prediction/"

config_path = proj_dir + 'data/record/pop_pred/eval/'


folders = os.walk(config_path)

print(folders)



filelist = []

for root, dirs, files in os.walk(config_path):
	for file in files:
        #append the file name to the list
		filelist.append(os.path.join(root,file).replace('\\', '/'))



configlist = []
dflist = []
arrlist = []
for file in filelist:
    if file.endswith("config.csv"):
        df_config = read_csv(file)
        configlist.append(df_config)
    elif file.endswith(".csv"):
        df = read_csv(file)
        data = df.values
        dflist.append(df)
        arrlist.append(data)
        

myrange = [*range(len(configlist))]
# summarize the infos per model run
trainings = pd.DataFrame(index = myrange, columns = ["l1", "l2", "lr", "bs", "train_mae", "val_mae"])

for i in range( len(configlist)):
    trainings.loc[i, "l1"] = configlist[i].loc[0, "l1"]
    trainings.loc[i, "l2"] = configlist[i].loc[0, "l2"]
    trainings.loc[i, "lr"] = configlist[i].loc[0, "lr"]
    trainings.loc[i, "batch_size"] = configlist[i].loc[0, "batch_size"]
    trainings.loc[i, "train_mae"] = dflist[i]["train_mae"].iloc[-1]
    trainings.loc[i, "val_mae"] = dflist[i]["val_mae"].iloc[-1]
    
    
# create real NaN values
trainings["l2"] = pd.to_numeric(trainings["l2"], errors = 'coerce')
    
# plt.plot(trainings) 
    
    
    
###############################################################################
# radar chart    
###############################################################################
import plotly.graph_objects as go
import plotly.io as pio  
pio.renderers.default='browser'  
fig = go.Figure()
for i in range(len(configlist)):
    fig.add_trace(go.Scatterpolar(
        r = trainings.loc[i], 
        theta = trainings.columns,
        name = i))
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 5]
    )),
  showlegend=False
)

fig.show()

 
 