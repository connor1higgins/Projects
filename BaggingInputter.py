# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 09:07:23 2018

@author: Connor Higgins
"""

#HVAC_Unit_Predictor(Tonnage, Amps_NP, AVG_Amps, Voltage_NP, AVG_Volts, 
#                    IM_HP, IM_Amps_NP, IM_Actual, Calc_TD)

#Function will ask for a identifier of some sort, something to keep track of the unit

import pandas as pd
import numpy as np
df_new_units = pd.DataFrame(data = np.zeros((1, 11)),
                            columns=['Key', 'Result', 'Tonnage', 'Amps_NP',
                                     'AVG_Amps', 'Voltage_NP', 'AVG_Volts',
                                     'IM_HP', 'IM_Amps_NP', 'IM_Actual', 'Calc_TD'])
df_new_units = df_new_units.replace(0, np.nan)

#Appending data to new dataframe
#df_list = df_new_units.append(HVAC_Unit_Predictor(7.5, 12, 20.7, 230, 243, 2, 6.3, 4.3, 24), ignore_index=True)

#Appending data to existing list
#df_list = df_list.append(HVAC_Unit_Predictor(7.5, 12, 21.7, 235, 260, 4, 6.3, 4.3, 24), ignore_index=True)


# Saving this as a csv file. This could then be loaded with the original training set and subsequently combined
# This would then give us a larger dataset that is consistently growing with new training info,
# to be used to increase overall classification accuracy. 

#df_list.to_csv(r"C:\Users\Connor Higgins\Documents\Work\TEC_Mechanical\TraneMLTest.csv")