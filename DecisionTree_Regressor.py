import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


#same dataset use for polynomial and svr
dataset  = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Documents\Machine Learning programs\emp_sal.csv')

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Decision Tree

from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor(criterion='poisson',max_depth=3)    #criterion='absolute_error',max_depth=5,random_State = 0,splitter = 'best'
dt_reg.fit(x,y)

dt_predict = dt_reg.predict([[6.5]])
print(dt_predict)

