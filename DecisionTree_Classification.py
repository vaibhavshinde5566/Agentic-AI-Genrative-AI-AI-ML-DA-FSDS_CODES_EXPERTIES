import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Documents\Machine Learning programs\logit classification.csv')

x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split
x_train ,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2 , random_state=0)

#=Ftutre scaling ======================  tree algorithm does not require future scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=0)
dc.fit(x_train, y_train)

y_pred = dc.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)           # ac = 95

bias = dc.score(x_train, y_train)
print(bias)

variance = dc.score(x_test, y_test)
print(variance)