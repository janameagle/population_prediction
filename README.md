# population_prediction
Population prediction and multihazard exposure assessment

This repository was created for my Master Thesis with the title 'Anticipating a Risky Future: Neural Networks for Predictive Exposure Analysis.

Multiple Long Short-Term Memory deep learning networks are trained with multitemporal WorldPop population grids and ancillary geospatial data.

The resulting forecasted future population distributions are used together with an earthquake and tsunami model to quantify the affected population. 


These are the processing steps and according folders:
- preprocessing : normalizing, masking, resampling, stacking of input features and creation of training tiles
- train: train different deep learning models 
          - model: architecture of the neural networks
          - utilis: additional settings for the networks
- predict: use the trained networks to predict future time steps
- results: analyse and validate the predictions results
- hazard: analyze the earthquake and tsunami model for the exposure assessment
