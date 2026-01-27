import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


#same dataset use for polynomial and svr
dataset  = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Documents\Machine Learning programs\emp_sal.csv')

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


#SVR model
from sklearn.svm import SVR
regressor  = SVR(kernel = 'poly',degree=4,gamma='auto', C = 5.0)  # regressor is model | SVR is algorithm
regressor.fit(x, y)

y_pred_svr = regressor.predict([[6.5]])
print(y_pred_svr)


# KNN  Model
from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor(n_neighbors=4,weights='distance',algorithm='brute')

knn_reg.fit(x, y)

y_pred_knn = knn_reg.predict([[6.5]])
print(y_pred_knn)

