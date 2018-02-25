# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 08:45:22 2018

@author: Connor Higgins
"""
import numpy as np
import pandas as pd

# Loading Imputed Dataset
df = pd.read_csv(r"C:\Users\Connor Higgins\Documents\Work\TEC_Mechanical\TraneML5000NUM.csv",
                index_col='Unnamed: 0')

#Splitting the data into training and testing portions: test set is 0 when classifying new units
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0, random_state=42)

#Splitting each portion into X value and y target variables
X_train = train_set.drop(['Target'], axis=1)
y_train = train_set['Target']

X_test = test_set.drop(['Target'], axis=1)
y_test = test_set['Target']

#Applying the BaggingClassifier Algorithm
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=2000, #num of decision trees (2000 was used based on optimization)
    max_samples=500, bootstrap=True, #500 training instances (again chosen for optimization (~590 instances in total))
    n_jobs=-1)
bag_clf.fit(X_train, y_train) #Fitting the data to the model

def HVAC_Unit_Predictor(Tonnage, Amps_NP, AVG_Amps, Voltage_NP, AVG_Volts, IM_HP, IM_Amps_NP, IM_Actual, Calc_TD):
    
    # Inputing unit data into an array
    unit_predictor = np.array([Tonnage, Amps_NP, AVG_Amps, Voltage_NP, AVG_Volts, IM_HP, IM_Amps_NP, IM_Actual, Calc_TD])
    
    # Fitting unit array to model to determine status and probability of status
    y_unit_predictor = bag_clf.predict(unit_predictor.reshape(1,-1))
    y_unit_proba = bag_clf.predict_proba(unit_predictor.reshape(1, -1))
    
    # Status classifying
    status = ''
    if y_unit_predictor == 1:
        status = 'Broken'
    else:
        status = 'Working'
    
    # Assigning probabilities to variables
    working_proba = y_unit_proba[:, 0]
    broken_proba = y_unit_proba[:, 1]
    
    # Setting up a readable response based on status and working/broken probability
    prediction = "This unit is predicted as {}, ".format(status)
    percent_broken = "with a {}% chance that it's Broken, ".format(broken_proba[0] *100)
    percent_working = "and a {}% chance that it's Working.".format(working_proba[0]*100)
    
    # Saving information such that it can be retrieved
    key = input('Please provide either an order number, date, building location, and/or some other identifier: ')
    result = y_unit_predictor[0]
    specific_unit_info = pd.DataFrame(data = np.array([key, result, Tonnage, Amps_NP, AVG_Amps, Voltage_NP,
                                                       AVG_Volts, IM_HP, IM_Amps_NP, IM_Actual, Calc_TD]).reshape(1,11),
                                      columns = ['Key', 'Result', 'Tonnage', 'Amps_NP', 'AVG_Amps', 'Voltage_NP',
                                                 'AVG_Volts', 'IM_HP', 'IM_Amps_NP', 'IM_Actual', 'Calc_TD'])
    # Read-Out Info
    print(prediction + percent_broken + percent_working)

    # Data Result (inputted to data_frame)
    return(specific_unit_info)
