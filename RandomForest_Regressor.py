import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


#same dataset use for polynomial and svr
dataset  = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Documents\Machine Learning programs\emp_sal.csv')

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Random forest Algorithm

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=20, random_state=43)
rf_reg.fit(x, y)

rf_pred = rf_reg.predict([[6.5]])
print(rf_pred)

