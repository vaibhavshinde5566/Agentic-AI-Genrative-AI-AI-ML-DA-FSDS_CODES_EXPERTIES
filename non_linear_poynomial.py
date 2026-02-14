import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dataset  = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Documents\Machine Learning programs\emp_sal.csv')

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#linear model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

#linear vis
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x),color = 'blue')
plt.title('linear regression graph')
plt.xlabel('plosition level')
plt.ylabel('salary')
plt.show()
#is os not good model
# this is for predictyion is predict 33ctc th
lin_model_pred = lin_reg.predict([[6.5]])  #6.5 is a salary to predict future 
print(lin_model_pred)


# will be using polynomial
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)  #by default it gives 2 degree means 2 square

x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)


lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

#
plt.scatter(x, y, color = 'red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color = 'blue')
plt.title('Truth of bluff deg 5')
plt.xlabel('position level')
plt.ylabel('salary')

lin_model_pred = lin_reg.predict([[6.5]])   #this is linear model that is not correct prediction it give 6.5 salary feature predict 33 lakh
lin_model_pred

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))   # this is polynomial prediction thia is predict correct 6.5 is salry predict next is 18 lakh
print(poly_model_pred)