import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Documents\Machine Learning programs\Churn_Modelling.csv')

x = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

#encoding categorical data
#label encoding the 'gender' column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])

# one hot encoding the 'geography' column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)


from sklearn.model_selection import train_test_split
x_train ,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2 , random_state=0)

from xgboost import XGBClassifier
classifier = XGBClassifier(random_state = 0)
classifier.fit(x_train,y_train)


y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

bias = classifier.score(x_train, y_train)
print(bias)

variance = classifier.score(x_test, y_test)
print(variance)


# Applying k fold cross validation 
from sklearn.model_selection import cross_val_score
accusaries = cross_val_score(estimator = classifier, X = x_train, y = y_train , cv=5 )
print("accuracy{:.2f} %".format(accusaries.mean()*100))

      