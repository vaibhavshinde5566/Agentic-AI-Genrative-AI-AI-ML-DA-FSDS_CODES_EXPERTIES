import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Documents\Machine Learning programs\logit classification.csv')

x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train ,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2 , random_state=0)

 #================ Standard scaller
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#========================= Normalizer
# it is for "multinomialNB
# from sklearn.preprocessing import Normalizer
# nl = Normalizer()
# x_train = nl.fit_transform(x_train)
# x_test = nl.transform(x_test)


#=============== BernoulliNb
# from sklearn.naive_bayes import BernoulliNB
# nv = BernoulliNB()
# nv.fit(x_train, y_train)
# y_pred  = nv.predict(x_test)

#==============Guasian navive baise
#guasian naive baise does not required standaed 
from sklearn.naive_bayes import GaussianNB
gnv = GaussianNB()
gnv.fit(x_train, y_train)
y_pred  = gnv.predict(x_test)

#======================

# from sklearn.naive_bayes import MultinomialNB
# mnv = MultinomialNB()
# mnv.fit(x_train, y_train)
# y_pred  = mnv.predict(x_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix( y_test , y_pred)
print(cm)

# it is for accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)      # accuracy is 0.95 it is best accuracy