import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Documents\Machine Learning programs\car-mpg.csv')

from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso ,Ridge

from sklearn.metrics import r2_score


data = data.drop(['car_name'],axis = 1)
data['origin'] = data['origin'].replace({1:'america',2:'europe',3:'asia'})
data = pd.get_dummies(data,columns=['origin'],dtype=int)
data = data.replace('?',np.nan)
data = data.apply(lambda x:x.fillna(x.median()),axis = 0)                                   



data = data.apply(pd.to_numeric,error = 'ignore')
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].apply(lambda x:x.fillna(x.median()))

x = data.drop(['mpg'],axis = 1)   #independent
y = data[['mpg']]                 #dependent

X_s = preprocessing.scale(x)
x_s = pd.DataFrame(X_s, columns=X.columns)

Y_s = preprocessing.scale(Y)
Y_s = pd.DataFrame(Y_s, columns=Y.columns)

data.shape

X_train, X_test, y_train,y_test = train_test_split(X_s, y_s, test_size = 0.30, random_state = 1)
X_train.shape

#simple linear model

#Fit simple linear model and find coefficients
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

for idx, col_name in enumerate(X_train.columns):
    print('The coefficient for {} is {}'.format(col_name, regression_model.coef_[0][idx]))
    
intercept = regression_model.intercept_[0]
print('The intercept is {}'.format(intercept))


#alpha factor here is lambda (penalty term) which helps to reduce the magnitude of coeff

ridge_model = Ridge(alpha = 0.3)
ridge_model.fit(X_train, y_train)

print('Ridge model coef: {}'.format(ridge_model.coef_))
#As the data has 10 columns hence 10 coefficients appear here    



#alpha factor here is lambda (penalty term) which helps to reduce the magnitude of coeff

lasso_model = Lasso(alpha = 0.1)
lasso_model.fit(X_train, y_train)

print('Lasso model coef: {}'.format(lasso_model.coef_))
#As the data has 10 columns hence 10 coefficients appear here   