# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 08:45:55 2018

@author: Connor Higgins
"""
import numpy as np
import pandas as pd

# Loading Imputed Dataset
df = pd.read_csv(r"C:\Users\Connor Higgins\Documents\Work\XXXXX\XXX.csv",
                index_col='Unnamed: 0')

#Splitting the data into training and testing portions
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

#Splitting each portion into X value and y target variables
X_train = train_set.drop(['Target'], axis=1)
y_train = train_set['Target']

X_test = test_set.drop(['Target'], axis=1)
y_test = test_set['Target']

#Applying the BaggingClassifier Algorithm
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=2000, #num of decision trees (2000 was used based on optimization)
    max_samples=500, bootstrap=True, #500 training instances (again chosen for optimization (~590 instances in total))
    n_jobs=-1)
bag_clf.fit(X_train, y_train) #Fitting the data to the model
y_predb = bag_clf.predict(X_test) #Creating a predicted value based on the X_test values

print('Bagging Accuracy:', accuracy_score(y_test, y_predb)) # Overall accuracy of the model 
