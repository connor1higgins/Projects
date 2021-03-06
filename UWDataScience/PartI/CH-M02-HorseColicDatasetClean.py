# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 18:46:26 2018

@author: Connor Higgins

"""

""" Project I completed for a UW Data Science Course. 
      Takes an unclean dataset from the uci machine learning archives
      and returns a clean, imputed, one-hot encoded .csv file ready for model fitting"""

### Setup ###
## import statements
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

## loading dataset
# url: url for horse-colic.data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/'\
      'horse-colic.data'
# horsedf: using pd.read_csv to load horse-colic.data, 
#          values separated by 1 or more spaces, no header included
horsedf = pd.read_csv(url, sep='\s+', header=None)

## applying column names using information from horse-colic.names
#        names slightly changed for callability
horsedf.columns = ['Surgery', # 1=  yes, 2 = no
        'Age', # 1 = Adult, 2 = Young (< 6 months)
        'Hospital_Number', # numeric id assigned to horse
        'Rectal_Temperature', # in degrees celcius
        'Pulse', # in beats per minute
        'Respiratory_Rate', # in breaths per minute
        'Temperature_of_Extremities', # 1 = Normal, 2 = Warm, 3 = Cool, 4 = Cold 
        'Peripheral_Pulse', # 1 = Normal, 2 = Increased, 3 = Reduced, 4 = Absent
        'Mucous_Membranes', # 1 = Normal, 2 = Bright, 3 = Pale, 4 = Pcyan, 5 = Red, 6 = Dcyan 
        'Capillary_Refill_Time', # 1 = (< 3 seconds), 2 = (>= 3 seconds)
        'Pain', # 1 = None, 2 = Depressed, 3 = Mild, 4 = Severe, 5 = Cont.Severe
        'Peristalsis', # 1 = Hypermotile, 2 = Normal, 3 = Hypomotile, 4 = Absent
        'Abdominal_Distension', # 1 = None, 2 = Slight, 3 = Moderate, 4 = Severe
        'Nasogastric_Tube', # amount of gas: 1 = None, 2 = Slight, 3 = Significant
        'Nasogastric_Reflux', # amount of reflux: 1 = None, 2 = > 1 liter, 3 = < 1 liter
        'Nasogastric_Reflux_PH', # in pH units
        'Feces', # 1 = Normal, 2 = Increased, 3 = Decreased, 4 = Absent
        'Abdomen', # 1 = Normal, 2 = Other, 3 = Firm Feces,  4 = Dist.S.Int., 5 = Dist.L.Int.
        'Packed_Cell_Volume', # in percent of cells in blood by volume
        'Total_Protein',  # in g/dL
        'Abdominocentesis_Appearance', # 1 = Clear, 2 = Cloudy, 3 = Serosanguinous
        'Abdominocentesis_Total_Protein', # in g/dL
        'Outcome', # 1 = Lived, 2 = Died, 3 = Euthanized
        'Surgical_Lesion', # 1 = Yes, 2 = No
        'Lesion_1', # Lesion Code for Surgical Lesion 1 (if applicable) 
        'Lesion_2', # Lesion Code for Surgical Lesion 2 (if applicable)
        'Lesion_3', # Lesion Code for Surgical Lesion 3 (if applicable)
        'cp_data'  ] # pathology data present: 1 = Yes, 2 = No

### Numeric Data ###
# numcols: all columns containing numeric data
numcols = horsedf[['Rectal_Temperature', 'Pulse',
                   'Respiratory_Rate', 'Nasogastric_Reflux_PH',
                   'Packed_Cell_Volume', 'Total_Protein', 
                   'Abdominocentesis_Total_Protein']]

## finding any missing values that are mislabeled, replacing with np.NaN
numcols = numcols.replace('?', np.NaN)

## casting numeric columns to float
numcols = numcols.astype(float)

## fixing misentered values in column 'Total_Protein'
# highprotein: values in column 'Total_Protein' that are greater than 15
Replace = numcols.loc[:, 'Total_Protein'] > 15
# assigning a tenth of the original value to where highprotein is true
numcols.loc[Replace, 'Total_Protein'] = numcols.loc[Replace, 'Total_Protein'] * 0.10
           
## removing outliers from column 'Rectal_Temperature'
Replace = (numcols.loc[:,'Rectal_Temperature'] 
            >= (np.mean(numcols.loc[:,'Rectal_Temperature']) 
                + 2 * np.std(numcols.loc[:,'Rectal_Temperature'])))\
        | (numcols.loc[:,'Rectal_Temperature'] 
            <= (np.mean(numcols.loc[:,'Rectal_Temperature'])
                - 2 * np.std(numcols.loc[:,'Rectal_Temperature'])))
# converting outliers to np.nan
numcols.loc[Replace, 'Rectal_Temperature'] = np.nan       

## removing outliers from column 'Pulse'
#Pulseoutliers: boolean array of all 'Pulse' outliers
Replace = (numcols.loc[:,'Pulse'] >= (np.mean(numcols.loc[:,'Pulse']) 
                                  + 2 * np.std(numcols.loc[:,'Pulse'])))\
        | (numcols.loc[:,'Pulse'] <= (np.mean(numcols.loc[:,'Pulse'])
                                  - 2 * np.std(numcols.loc[:,'Pulse'])))
# converting outliers to np.nan
numcols.loc[Replace, 'Pulse'] = np.nan          
           
## removing outliers from column 'Respiratory_Rate'
#RespRateoutliers: boolean array of all 'Respiratory_Rate' outliers
Replace = (numcols.loc[:,'Respiratory_Rate'] 
           >= (np.mean(numcols.loc[:,'Respiratory_Rate']) 
               + 2 * np.std(numcols.loc[:,'Respiratory_Rate'])))\
        | (numcols.loc[:,'Respiratory_Rate'] 
           <= (np.mean(numcols.loc[:,'Respiratory_Rate'])
               - 2 * np.std(numcols.loc[:,'Respiratory_Rate'])))
# converting outliers to np.nan
numcols.loc[Replace, 'Respiratory_Rate'] = np.nan

## removing outliers from column 'Total_Protein'
Replace = (numcols.loc[:,'Total_Protein'] 
           >= (np.mean(numcols.loc[:,'Total_Protein']) 
               + 2 * np.std(numcols.loc[:,'Total_Protein'])))\
        | (numcols.loc[:,'Total_Protein']
           <= (np.mean(numcols.loc[:,'Total_Protein'])
               - 2 * np.std(numcols.loc[:,'Total_Protein'])))
# converting outliers to np.nan
numcols.loc[Replace, 'Total_Protein'] = np.nan    

# dropping columns 'Nasogastric_Reflux_PH' and 'Abdominocentesis_Total_Protein'
# from numcols, as they are missing 82.3% and 66.0% of their data respectively
numcols = numcols.drop(['Nasogastric_Reflux_PH',
                        'Abdominocentesis_Total_Protein'], axis=1)

## imputing missing data for remaining numeric columns using the median
for i in numcols.columns:
    Median = np.nanmedian(numcols.loc[:, i])
    HasNan = np.isnan(numcols.loc[:, i])
    numcols.loc[HasNan, i] = Median

## creating non-normalized version of numcols: numcols_nonnormed
numcols_nonnormed = numcols.copy()

## normalizing all data using sklearn
for i in numcols.columns:
    X = numcols.loc[:, i].values.reshape(-1, 1)
    stdscale = StandardScaler().fit(X)
    numcols.loc[:, i] = stdscale.transform(X)

### Categorical Data ###
## catcols: all columns containing categorical data
catcols = horsedf[['Surgery', 'Age', 'Hospital_Number',
                   'Temperature_of_Extremities', 'Peripheral_Pulse',
                   'Mucous_Membranes', 'Capillary_Refill_Time', 
                   'Pain', 'Peristalsis', 'Abdominal_Distension',
                   'Nasogastric_Tube', 'Nasogastric_Reflux', 
                   'Feces', 'Abdomen', 'Abdominocentesis_Appearance', 
                   'Outcome', 'Surgical_Lesion', 'Lesion_1',
                   'Lesion_2', 'Lesion_3', 'cp_data']]

## Dropping columns 'Hospital_Number' and 'cp_data', as the former a unique id
# that isn't connected to a health metric and the latter is not significant
# sine pathology data is not included or collected for these cases. 
catcols = catcols.drop(['Hospital_Number', 'cp_data'], axis=1)

## Dropping 'Lesion_2' and 'Lesion_3' columns as they are missing 96.7% and 99.6%
# of their data respectively
catcols = catcols.drop(['Lesion_2', 'Lesion_3'], axis=1)

## Casting categorical columns to str
catcols = catcols.astype(str)

## decoding catcol columns and imputing missing values
# 'Surgery' column
Replace = catcols.loc[:, 'Surgery'] == '1'
catcols.loc[Replace, 'Surgery'] = 'Yes'
Replace = (catcols.loc[:, 'Surgery'] == '2') \
        | (catcols.loc[:, 'Surgery'] == '?')
catcols.loc[Replace, 'Surgery'] = 'No'
# 'Age' column
Replace = catcols.loc[:, 'Age'] == '1'
catcols.loc[Replace, 'Age'] = 'Adult'
Replace = catcols.loc[:, 'Age'] == '9'
catcols.loc[Replace, 'Age'] = 'Young'
# 'Temperature_of_Extremities' column
Replace = (catcols.loc[:, 'Temperature_of_Extremities'] == '1') \
        | (catcols.loc[:, 'Temperature_of_Extremities'] == '?')
catcols.loc[Replace, 'Temperature_of_Extremities'] = 'Normal'
Replace = catcols.loc[:, 'Temperature_of_Extremities'] == '2'
catcols.loc[Replace, 'Temperature_of_Extremities'] = 'Warm'
Replace = catcols.loc[:, 'Temperature_of_Extremities'] == '3'
catcols.loc[Replace, 'Temperature_of_Extremities'] = 'Cool'
Replace = catcols.loc[:, 'Temperature_of_Extremities'] == '4'
catcols.loc[Replace, 'Temperature_of_Extremities'] = 'Cold'
# 'Peripheral_Pulse' column
Replace = (catcols.loc[:, 'Peripheral_Pulse'] == '1') \
        | (catcols.loc[:, 'Peripheral_Pulse'] == '?')
catcols.loc[Replace, 'Peripheral_Pulse'] = 'Normal'
Replace = catcols.loc[:, 'Peripheral_Pulse'] == '2'
catcols.loc[Replace, 'Peripheral_Pulse'] = 'Increased'
Replace = catcols.loc[:, 'Peripheral_Pulse'] == '3'
catcols.loc[Replace, 'Peripheral_Pulse'] = 'Reduced'
Replace = catcols.loc[:, 'Peripheral_Pulse'] == '4'
catcols.loc[Replace, 'Peripheral_Pulse'] = 'Absent' 
# 'Mucous_Membranes' column
Replace = (catcols.loc[:, 'Mucous_Membranes'] == '1') \
        | (catcols.loc[:, 'Mucous_Membranes'] == '?')
catcols.loc[Replace, 'Mucous_Membranes'] = 'Normal'
Replace = catcols.loc[:, 'Mucous_Membranes'] == '2'
catcols.loc[Replace, 'Mucous_Membranes'] = 'Bright'
Replace = catcols.loc[:, 'Mucous_Membranes'] == '3'
catcols.loc[Replace, 'Mucous_Membranes'] = 'Pale'
Replace = catcols.loc[:, 'Mucous_Membranes'] == '4'
catcols.loc[Replace, 'Mucous_Membranes'] = 'Pale_Cyan'
Replace = catcols.loc[:, 'Mucous_Membranes'] == '5'
catcols.loc[Replace, 'Mucous_Membranes'] = 'Red'
Replace = catcols.loc[:, 'Mucous_Membranes'] == '6'
catcols.loc[Replace, 'Mucous_Membranes'] = 'Dark_Cyan'
# 'Capillary_Refill_Time' column
Replace = (catcols.loc[:, 'Capillary_Refill_Time'] == '1') \
        | (catcols.loc[:, 'Capillary_Refill_Time'] == '?')
catcols.loc[Replace, 'Capillary_Refill_Time'] = '<3sec'
Replace = (catcols.loc[:, 'Capillary_Refill_Time'] == '2') \
        | (catcols.loc[:, 'Capillary_Refill_Time'] == '3')
catcols.loc[Replace, 'Capillary_Refill_Time'] = '>=3sec'
# 'Pain' column
Replace = (catcols.loc[:, 'Pain'] == '1') \
        | (catcols.loc[:, 'Pain'] == '?')
catcols.loc[Replace, 'Pain'] = 'None'
Replace = catcols.loc[:, 'Pain'] == '2'
catcols.loc[Replace, 'Pain'] = 'Depressed'
Replace = catcols.loc[:, 'Pain'] == '3'
catcols.loc[Replace, 'Pain'] = 'Mild'
Replace = catcols.loc[:, 'Pain'] == '4'
catcols.loc[Replace, 'Pain'] = 'Intermittent_Severe'
Replace = catcols.loc[:, 'Pain'] == '5'
catcols.loc[Replace, 'Pain'] = 'Continuous_Severe'
# 'Peristalsis' column
Replace = catcols.loc[:, 'Peristalsis'] == '1'
catcols.loc[Replace, 'Peristalsis'] = 'Hypermotile'
Replace = (catcols.loc[:, 'Peristalsis'] == '2') \
        | (catcols.loc[:, 'Peristalsis'] == '?')
catcols.loc[Replace, 'Peristalsis'] = 'Normal'
Replace = catcols.loc[:, 'Peristalsis'] == '3'
catcols.loc[Replace, 'Peristalsis'] = 'Hypomotile'
Replace = catcols.loc[:, 'Peristalsis'] == '4'
catcols.loc[Replace, 'Peristalsis'] = 'Absent'
# 'Abdominal_Distension' column
Replace = (catcols.loc[:, 'Abdominal_Distension'] == '1') \
        | (catcols.loc[:, 'Abdominal_Distension'] == '?')
catcols.loc[Replace, 'Abdominal_Distension'] = 'None'
Replace = catcols.loc[:, 'Abdominal_Distension'] == '2'
catcols.loc[Replace, 'Abdominal_Distension'] = 'Slight'
Replace = catcols.loc[:, 'Abdominal_Distension'] == '3'
catcols.loc[Replace, 'Abdominal_Distension'] = 'Moderate'
Replace = catcols.loc[:, 'Abdominal_Distension'] == '4'
catcols.loc[Replace, 'Abdominal_Distension'] = 'Severe'
# 'Nasogastric_Tube' column
Replace = (catcols.loc[:, 'Nasogastric_Tube'] == '1') \
        | (catcols.loc[:, 'Nasogastric_Tube'] == '?')
catcols.loc[Replace, 'Nasogastric_Tube'] = 'No_Gas'
Replace = catcols.loc[:, 'Nasogastric_Tube'] == '2'
catcols.loc[Replace, 'Nasogastric_Tube'] = 'Slight_Gas'
Replace = catcols.loc[:, 'Nasogastric_Tube'] == '3'
catcols.loc[Replace, 'Nasogastric_Tube'] = 'Significant_Gas'
# 'Nasogastric_Reflux' column
Replace = (catcols.loc[:, 'Nasogastric_Reflux'] == '1') \
        | (catcols.loc[:, 'Nasogastric_Reflux'] == '?')
catcols.loc[Replace, 'Nasogastric_Reflux'] = 'None'
Replace = catcols.loc[:, 'Nasogastric_Reflux'] == '2'
catcols.loc[Replace, 'Nasogastric_Reflux'] = '>=1liter'
Replace = catcols.loc[:, 'Nasogastric_Reflux'] == '3'
catcols.loc[Replace, 'Nasogastric_Reflux'] = '<1liter'
# 'Feces' column
Replace = (catcols.loc[:, 'Feces'] == '1') \
        | (catcols.loc[:, 'Feces'] == '?')
catcols.loc[Replace, 'Feces'] = 'Normal'
Replace = catcols.loc[:, 'Feces'] == '2'
catcols.loc[Replace, 'Feces'] = 'Increased'
Replace = catcols.loc[:, 'Feces'] == '3'
catcols.loc[Replace, 'Feces'] = 'Decreased'
Replace = catcols.loc[:, 'Feces'] == '4'
catcols.loc[Replace, 'Feces'] = 'Absent'
# 'Abdomen' column
Replace = (catcols.loc[:, 'Abdomen'] == '1') \
        | (catcols.loc[:, 'Abdomen'] == '?')
catcols.loc[Replace, 'Abdomen'] = 'Normal'
Replace = catcols.loc[:, 'Abdomen'] == '2'
catcols.loc[Replace, 'Abdomen'] = 'Abnormal'
Replace = catcols.loc[:, 'Abdomen'] == '3'
catcols.loc[Replace, 'Abdomen'] = 'Firm_Feces'
Replace = catcols.loc[:, 'Abdomen'] == '4'
catcols.loc[Replace, 'Abdomen'] = 'Distended_Small_Intestine'
Replace = catcols.loc[:, 'Abdomen'] == '5'
catcols.loc[Replace, 'Abdomen'] = 'Distended_Large_Intestine'
# 'Abdominocentesis_Appearance' column
Replace = (catcols.loc[:, 'Abdominocentesis_Appearance'] == '1') \
        | (catcols.loc[:, 'Abdominocentesis_Appearance'] == '?')
catcols.loc[Replace, 'Abdominocentesis_Appearance'] = 'Clear'
Replace = catcols.loc[:, 'Abdominocentesis_Appearance'] == '2'
catcols.loc[Replace, 'Abdominocentesis_Appearance'] = 'Cloudy'
Replace = catcols.loc[:, 'Abdominocentesis_Appearance'] == '3'
catcols.loc[Replace, 'Abdominocentesis_Appearance'] = 'Serosanguinous'
# 'Outcome' column
Replace = (catcols.loc[:, 'Outcome'] == '1') \
        | (catcols.loc[:, 'Outcome'] == '?')
catcols.loc[Replace, 'Outcome'] = 'Lived'
Replace = catcols.loc[:, 'Outcome'] == '2'
catcols.loc[Replace, 'Outcome'] = 'Died'
Replace = catcols.loc[:, 'Outcome'] == '3'
catcols.loc[Replace, 'Outcome'] = 'Euthanized'
# 'Surgical_Lesion' column
Replace = catcols.loc[:, 'Surgical_Lesion'] == '1'
catcols.loc[Replace, 'Surgical_Lesion'] = 'Yes'
Replace = catcols.loc[:, 'Surgical_Lesion'] == '2'
catcols.loc[Replace, 'Surgical_Lesion'] = 'No'

## Decoding catcols column 'Lesion_1'
# separating codes into temporary 3, 4, and 5 digit variant columns
for i in range(len(catcols)):
    if len(catcols.loc[i, 'Lesion_1']) == 3:
        catcols.loc[i, 'ThreeNumCodes'] = catcols.loc[i, 'Lesion_1']
    elif len(catcols.loc[i, 'Lesion_1']) == 4:
        catcols.loc[i, 'FourNumCodes'] = catcols.loc[i, 'Lesion_1']
    elif len(catcols.loc[i, 'Lesion_1']) == 5:
        catcols.loc[i, 'FiveNumCodes'] = catcols.loc[i, 'Lesion_1']
catcols = catcols.replace(np.NaN, '?')
# creating column 'Lesion_Site' from 1st number of lesion code
for i in range(len(catcols)):
    if catcols.loc[:, 'Lesion_1'][i] == '0':
        catcols.loc[i, 'Lesion_Site'] = 'None'
    elif catcols.loc[:, 'FourNumCodes'][i][0] == '1':
        catcols.loc[i, 'Lesion_Site'] = 'Gastric'
    elif (catcols.loc[:, 'FourNumCodes'][i][0] == '2') \
    | (catcols.loc[:, 'FiveNumCodes'][i][0] == '2'):
        catcols.loc[i, 'Lesion_Site'] = 'Small_Intestine'
    elif (catcols.loc[:, 'FourNumCodes'][i][0] == '3') \
    | (catcols.loc[:, 'FiveNumCodes'][i][0] == '3') \
    | (catcols.loc[:, 'ThreeNumCodes'][i][0] == '3'):
        catcols.loc[i, 'Lesion_Site'] = 'Large_Intestine'
    elif (catcols.loc[:, 'FourNumCodes'][i][0] == '4') \
    | (catcols.loc[:, 'FiveNumCodes'][i][0] == '4') \
    | (catcols.loc[:, 'ThreeNumCodes'][i][0] == '4'):
        catcols.loc[i, 'Lesion_Site'] = 'Large_Colon_and_Cecum'
    elif catcols.loc[:, 'FourNumCodes'][i][0] == '5':
        catcols.loc[i, 'Lesion_Site'] = 'Cecum'
    elif catcols.loc[:, 'FourNumCodes'][i][0] == '6':
        catcols.loc[i, 'Lesion_Site'] = 'Transverse_Colon'
    elif catcols.loc[:, 'FourNumCodes'][i][0] == '7':
        catcols.loc[i, 'Lesion_Site'] = 'Rectum'
    elif catcols.loc[:, 'FourNumCodes'][i][0] == '8':
        catcols.loc[i, 'Lesion_Site'] = 'Uterus'
    elif catcols.loc[:, 'FourNumCodes'][i][0] == '9':
        catcols.loc[i, 'Lesion_Site'] = 'Bladder'
    elif (catcols.loc[:, 'FiveNumCodes'][i][0:2] == '11') \
    | (catcols.loc[:, 'FiveNumCodes'][i][0:2] == '12'):
        catcols.loc[i, 'Lesion_Site'] = 'All_Intestines'
# creating column 'Lesion_Type' from 2nd number of lesion code
for i in range(len(catcols)): # ThreeNumCodes and No Lesions
    if catcols.loc[:, 'Lesion_1'][i] == '0':
        catcols.loc[i, 'Lesion_Type'] = 'None'
    elif catcols.loc[:, 'ThreeNumCodes'][i] != '?':
        catcols.loc[i, 'Lesion_Type'] = 'Simple'
for i in range(len(catcols)): #FourNumCodes
    if catcols.loc[:, 'FourNumCodes'][i] == '?':
        pass
    elif (catcols.loc[:, 'FourNumCodes'][i][1] == '0') \
    | (catcols.loc[:, 'FourNumCodes'][i][1] == '1'):
        catcols.loc[i, 'Lesion_Type'] = 'Simple'
    elif catcols.loc[:, 'FourNumCodes'][i][1] == '2':
        catcols.loc[i, 'Lesion_Type'] = 'Strangulation'
    elif catcols.loc[:, 'FourNumCodes'][i][1] == '3':
        catcols.loc[i, 'Lesion_Type'] = 'Inflammation'
    elif catcols.loc[:, 'FourNumCodes'][i][1] == '4':
        catcols.loc[i, 'Lesion_Type'] = 'Other'
for i in range(len(catcols)): #FiveNumCodes
    if catcols.loc[:, 'FiveNumCodes'][i] == '?':
        pass
    elif (catcols.loc[:, 'FiveNumCodes'][i][0] != '1') \
    & (catcols.loc[:, 'FiveNumCodes'][i][1] == '1'):
        catcols.loc[i, 'Lesion_Type'] = 'Simple'
    elif catcols.loc[:, 'FiveNumCodes'][i][2] == '1':
        catcols.loc[i, 'Lesion_Type'] = 'Simple'
    elif catcols.loc[:, 'FiveNumCodes'][i][2] == '2':
        catcols.loc[i, 'Lesion_Type'] = 'Strangulation'
    elif catcols.loc[:, 'FiveNumCodes'][i][2] == '3':
        catcols.loc[i, 'Lesion_Type'] = 'Inflammation'
    elif catcols.loc[:, 'FiveNumCodes'][i][2] == '4':
        catcols.loc[i, 'Lesion_Type'] = 'Other'
# creating column 'Lesion_Subtype' from 3rd number of lesion code
for i in range(len(catcols)): # ThreeNumCodes and No Lesions
    if catcols.loc[:, 'Lesion_1'][i] == '0':
        catcols.loc[i, 'Lesion_Subtype'] = 'None'
    elif catcols.loc[:, 'ThreeNumCodes'][i] != '?':
        catcols.loc[i, 'Lesion_Subtype'] = 'None'
for i in range(len(catcols)): #FourNumCodes
    if catcols.loc[:, 'FourNumCodes'][i] == '?':
        pass
    elif catcols.loc[:, 'FourNumCodes'][i][2] == '0':
        catcols.loc[i, 'Lesion_Subtype'] = 'None'
    elif catcols.loc[:, 'FourNumCodes'][i][2] == '1':
        catcols.loc[i, 'Lesion_Subtype'] = 'Mechanical'
    elif (catcols.loc[:, 'FourNumCodes'][i][2] == '2') \
    | (catcols.loc[:, 'FourNumCodes'][i][2] == '3'):
        catcols.loc[i, 'Lesion_Subtype'] = 'Paralytic'
for i in range(len(catcols)): #FiveNumCodes
    if catcols.loc[:, 'FiveNumCodes'][i] == '?':
        pass
    elif (catcols.loc[:, 'FiveNumCodes'][i][0] != '1') \
    & (catcols.loc[:, 'FiveNumCodes'][i][2] == '1'):
        catcols.loc[i, 'Lesion_Subtype'] = 'Mechanical'
    elif catcols.loc[:, 'FiveNumCodes'][i][3] == '0':
        catcols.loc[i, 'Lesion_Subtype'] = 'None'
    elif catcols.loc[:, 'FiveNumCodes'][i][3] == '1':
        catcols.loc[i, 'Lesion_Subtype'] = 'Mechanical'
    elif catcols.loc[:, 'FiveNumCodes'][i][3] == '2':
        catcols.loc[i, 'Lesion_Subtype'] = 'Paralytic'
# creating column 'Lesion_Specific' from 4th number of lesion code
for i in range(len(catcols)): # ThreeNumCodes and No Lesions
    if catcols.loc[:, 'Lesion_1'][i] == '0':
        catcols.loc[i, 'Lesion_Specific'] = 'None'
    elif catcols.loc[:, 'ThreeNumCodes'][i] != '?':
        catcols.loc[i, 'Lesion_Specific'] = 'None'
for i in range(len(catcols)): #FourNumCodes
    if catcols.loc[:, 'FourNumCodes'][i] == '?':
        pass
    elif catcols.loc[:, 'FourNumCodes'][i][-1] == '0':
        catcols.loc[i, 'Lesion_Specific'] = 'None'
    elif catcols.loc[:, 'FourNumCodes'][i][-1] == '1':
        catcols.loc[i, 'Lesion_Specific'] = 'Obturation'
    elif catcols.loc[:, 'FourNumCodes'][i][-1] == '2':
        catcols.loc[i, 'Lesion_Specific'] = 'Intrinsic'
    elif catcols.loc[:, 'FourNumCodes'][i][-1] == '3':
        catcols.loc[i, 'Lesion_Specific'] = 'Extrinsic'
    elif catcols.loc[:, 'FourNumCodes'][i][-1] == '4':
        catcols.loc[i, 'Lesion_Specific'] = 'Adynamic'
    elif catcols.loc[:, 'FourNumCodes'][i][-1] == '5':
        catcols.loc[i, 'Lesion_Specific'] = 'Volvulus'
    elif catcols.loc[:, 'FourNumCodes'][i][-1] == '6':
        catcols.loc[i, 'Lesion_Specific'] = 'Intussusception'
    elif catcols.loc[:, 'FourNumCodes'][i][-1] == '7':
        catcols.loc[i, 'Lesion_Specific'] = 'Thromboembolic'
    elif catcols.loc[:, 'FourNumCodes'][i][-1] == '8':
        catcols.loc[i, 'Lesion_Specific'] = 'Hernia'
    elif catcols.loc[:, 'FourNumCodes'][i][-1] == '9':
        catcols.loc[i, 'Lesion_Specific'] = 'Lipoma'
for i in range(len(catcols)): #FiveNumCodes
    if catcols.loc[:, 'FiveNumCodes'][i] == '?':
        pass
    elif (catcols.loc[:, 'FiveNumCodes'][i][0] != '1') \
    & (catcols.loc[:, 'FiveNumCodes'][i][-2:] == '10'):
        catcols.loc[i, 'Lesion_Specific'] = 'Displacement'
    elif catcols.loc[:, 'FiveNumCodes'][i][-1] == '0':
        catcols.loc[i, 'Lesion_Specific'] = 'None'
    elif catcols.loc[:, 'FiveNumCodes'][i][-1] == '4':
        catcols.loc[i, 'Lesion_Specific'] = 'Adynamic'
    elif catcols.loc[:, 'FiveNumCodes'][i][-1] == '8':
        catcols.loc[i, 'Lesion_Specific'] = 'Hernia'
        
### Binning Categorical Values ###    
# catcols_nodrops: copy of catcols up to this point, no redundant column drops
catcols_nodrops = catcols.copy()
## Dropping obselete columns 'Lesion_1' and Three, Four, and Five 'NumCodes'
catcols = catcols.drop(['Lesion_1', 'FiveNumCodes',
                        'FourNumCodes', 'ThreeNumCodes'], axis=1)
    
## Dropping columns Lesion_Specific so as to group lesions solely by site, type,
# and subtype, reducing the number of unique categories
catcols = catcols.drop(['Lesion_Specific'], axis=1)

## Grouping 'Lesion_Type' values 'Other' and 'Inflammation' into one category: Other
# both categories have far fewer values in comparison to 'Simple' and 'Strangulation'
catcols.loc[catcols.loc[:, 'Lesion_Type'] == 'Inflammation', 'Lesion_Type'] = 'Other'
    
## Grouping 'Lesion_Site' values 'All_Intestines', 'Transverse_Colon', 'Cecum',
#  'Rectum', and 'Large_Colon_and_Cecum' into 'Other_Colon_Location'
Replace = (catcols.loc[:, 'Lesion_Site'] == 'All_Intestines') \
        | (catcols.loc[:, 'Lesion_Site'] == 'Transverse_Colon') \
        | (catcols.loc[:, 'Lesion_Site'] == 'Cecum') \
        | (catcols.loc[:, 'Lesion_Site'] == 'Rectum') \
        | (catcols.loc[:, 'Lesion_Site'] == 'Large_Colon_and_Cecum')
catcols.loc[Replace, 'Lesion_Site'] = 'Other_Colon_Location'

## Grouping 'Lesion_Site' values 'Uterus', 'Bladder', and 'Gastric' into 
#  'Non_Colon_Location'
Replace = (catcols.loc[:, 'Lesion_Site'] == 'Uterus') \
        | (catcols.loc[:, 'Lesion_Site'] == 'Bladder') \
        | (catcols.loc[:, 'Lesion_Site'] == 'Gastric')
catcols.loc[Replace, 'Lesion_Site'] = 'Non_Colon_Location'

## Grouping 'Abdomen' values 'Firm_Feces' and 'Normal' into 'Normal'
catcols.loc[catcols.loc[:, 'Abdomen'] == 'Firm_Feces', 'Abdomen'] = 'Normal'

## Grouping 'Abdomen' values 'Abnormal', 'Distended_Small_Intestine', and
#  'Distended_Large_Intestine' into 'Abnormal'
Replace = (catcols.loc[:, 'Abdomen'] == 'Distended_Small_Intestine') \
        | (catcols.loc[:, 'Abdomen'] == 'Distended_Large_Intestine')
catcols.loc[Replace, 'Abdomen'] = 'Abnormal'

### One Hot Encoding Categorical Columns ###
## Creating a copy of catcols for one-hot encoding: onehotcols
onehotcols = catcols.copy()
# 'Surgery' column
onehotcols.loc[:,'Surgery'] = (onehotcols.loc[:,'Surgery']=='Yes').astype(int)
# 'Age' column
onehotcols.loc[:,'Age_Adult'] = (onehotcols.loc[:,'Age']=='Adult').astype(int)
onehotcols = onehotcols.drop('Age', axis=1) # dropping obsolete column
# 'Temperature_of_Extremities' column
onehotcols.loc[:, 'ToE_Normal'] = (onehotcols.loc[:,
              'Temperature_of_Extremities']=='Normal').astype(int)
onehotcols.loc[:, 'ToE_Warm'] = (onehotcols.loc[:,
              'Temperature_of_Extremities']=='Warm').astype(int)
onehotcols.loc[:, 'ToE_Cool'] = (onehotcols.loc[:,
              'Temperature_of_Extremities']=='Cool').astype(int)
onehotcols.loc[:, 'ToE_Cold'] = (onehotcols.loc[:,
              'Temperature_of_Extremities']=='Cold').astype(int)
onehotcols = onehotcols.drop('Temperature_of_Extremities', axis=1) # obsolete
onehotcols = onehotcols.drop('ToE_Normal', axis=1) # redundant
# 'Peripheral_Pulse' column
onehotcols.loc[:, 'PerPul_Normal'] = (onehotcols.loc[:,
              'Peripheral_Pulse'] == 'Normal').astype(int)
onehotcols.loc[:, 'PerPul_Increased'] = (onehotcols.loc[:,
              'Peripheral_Pulse'] == 'Increased').astype(int)
onehotcols.loc[:, 'PerPul_Reduced'] = (onehotcols.loc[:,
              'Peripheral_Pulse'] == 'Reduced').astype(int)
onehotcols.loc[:, 'PerPul_Absent'] = (onehotcols.loc[:,
              'Peripheral_Pulse'] == 'Absent').astype(int)
onehotcols = onehotcols.drop('Peripheral_Pulse', axis=1)  # obsolete
onehotcols = onehotcols.drop('PerPul_Normal', axis=1) # redundant
# 'Mucous_Membranes' column
onehotcols.loc[:, 'MucMem_Normal'] = (onehotcols.loc[:,
              'Mucous_Membranes'] == 'Normal').astype(int)
onehotcols.loc[:, 'MucMem_Bright'] = (onehotcols.loc[:,
              'Mucous_Membranes'] == 'Bright').astype(int)
onehotcols.loc[:, 'MucMem_Pale'] = (onehotcols.loc[:,
              'Mucous_Membranes'] == 'Pale').astype(int)
onehotcols.loc[:, 'MucMem_Pale_Cyan'] = (onehotcols.loc[:,
              'Mucous_Membranes'] == 'Pale_Cyan').astype(int)
onehotcols.loc[:, 'MucMem_Red'] = (onehotcols.loc[:,
              'Mucous_Membranes'] == 'Red').astype(int)
onehotcols.loc[:, 'MucMem_Dark_Cyan'] = (onehotcols.loc[:,
              'Mucous_Membranes'] == 'Dark_Cyan').astype(int)   
onehotcols = onehotcols.drop('Mucous_Membranes', axis=1)  # obsolete
onehotcols = onehotcols.drop('MucMem_Normal', axis=1) # redundant
# 'Capillary_Refill_Time' column
onehotcols.loc[:, 'CapRefTime_<3sec'] = (onehotcols.loc[:,
              'Capillary_Refill_Time'] == '<3sec').astype(int)
onehotcols.loc[:, 'CapRefTime_>=3sec'] = (onehotcols.loc[:,
              'Capillary_Refill_Time'] == '>=3sec').astype(int)
onehotcols = onehotcols.drop('Capillary_Refill_Time', axis=1)  # obsolete
onehotcols = onehotcols.drop('CapRefTime_<3sec', axis=1) # redundant
# 'Pain' column
onehotcols.loc[:, 'Pain_None'] = (onehotcols.loc[:,
              'Pain'] == 'None').astype(int)
onehotcols.loc[:, 'Pain_Depressed'] = (onehotcols.loc[:,
              'Pain'] == 'Depressed').astype(int)
onehotcols.loc[:, 'Pain_Mild'] = (onehotcols.loc[:,
              'Pain'] == 'Mild').astype(int)
onehotcols.loc[:, 'Pain_Intermittent_Severe'] = (onehotcols.loc[:,
              'Pain'] == 'Intermittent_Severe').astype(int)
onehotcols.loc[:, 'Pain_Continuous_Severe'] = (onehotcols.loc[:,
              'Pain'] == 'Continuous_Severe').astype(int)
onehotcols = onehotcols.drop('Pain', axis=1)  # obsolete
onehotcols = onehotcols.drop('Pain_None', axis=1) # redundant
# 'Peristalis' column
onehotcols.loc[:, 'Peris_Hypermotile'] = (onehotcols.loc[:,
              'Peristalsis'] == 'Hypermotile').astype(int)
onehotcols.loc[:, 'Peris_Normal'] = (onehotcols.loc[:,
              'Peristalsis'] == 'Normal').astype(int)
onehotcols.loc[:, 'Peris_Hypomotile'] = (onehotcols.loc[:,
              'Peristalsis'] == 'Hypomotile').astype(int)
onehotcols.loc[:, 'Peris_Absent'] = (onehotcols.loc[:,
              'Peristalsis'] == 'Absent').astype(int)
onehotcols = onehotcols.drop('Peristalsis', axis=1)  # obsolete
onehotcols = onehotcols.drop('Peris_Normal', axis=1) # redundant
# 'Abdominal_Distension' column
onehotcols.loc[:, 'AbdDis_None'] = (onehotcols.loc[:,
              'Abdominal_Distension'] == 'None').astype(int)
onehotcols.loc[:, 'AbdDis_Slight'] = (onehotcols.loc[:,
              'Abdominal_Distension'] == 'Slight').astype(int)
onehotcols.loc[:, 'AbdDis_Moderate'] = (onehotcols.loc[:,
              'Abdominal_Distension'] == 'Moderate').astype(int)
onehotcols.loc[:, 'AbdDis_Severe'] = (onehotcols.loc[:,
              'Abdominal_Distension'] == 'Severe').astype(int)
onehotcols = onehotcols.drop('Abdominal_Distension', axis=1)  # obsolete
onehotcols = onehotcols.drop('AbdDis_None', axis=1) # redundant
# 'Nasogastric_Tube' column
onehotcols.loc[:, 'NasoTube_No_Gas'] = (onehotcols.loc[:,
              'Nasogastric_Tube'] == 'No_Gas').astype(int)
onehotcols.loc[:, 'NasoTube_Slight_Gas'] = (onehotcols.loc[:,
              'Nasogastric_Tube'] == 'Slight_Gas').astype(int)
onehotcols.loc[:, 'NasoTube_Significant_Gas'] = (onehotcols.loc[:,
              'Nasogastric_Tube'] == 'Significant_Gas').astype(int)
onehotcols = onehotcols.drop('Nasogastric_Tube', axis=1)  # obsolete
onehotcols = onehotcols.drop('NasoTube_No_Gas', axis=1) # redundant
# 'Nasogastric_Reflux' column
onehotcols.loc[:, 'NasoReflux_None'] = (onehotcols.loc[:,
              'Nasogastric_Reflux'] == 'None').astype(int)
onehotcols.loc[:, 'NasoReflux_>=1liter'] = (onehotcols.loc[:,
              'Nasogastric_Reflux'] == '>=1liter').astype(int)
onehotcols.loc[:, 'NasoReflux_<1liter'] = (onehotcols.loc[:,
              'Nasogastric_Reflux'] == '<1liter').astype(int)
onehotcols = onehotcols.drop('Nasogastric_Reflux', axis=1)  # obsolete
onehotcols = onehotcols.drop('NasoReflux_None', axis=1) # redundant
# 'Feces' column
onehotcols.loc[:, 'Feces_Normal'] = (onehotcols.loc[:,
              'Feces'] == 'Normal').astype(int)
onehotcols.loc[:, 'Feces_Increased'] = (onehotcols.loc[:,
              'Feces'] == 'Increased').astype(int)
onehotcols.loc[:, 'Feces_Decreased'] = (onehotcols.loc[:,
              'Feces'] == 'Decreased').astype(int)
onehotcols.loc[:, 'Feces_Absent'] = (onehotcols.loc[:,
              'Feces'] == 'Absent').astype(int)
onehotcols = onehotcols.drop('Feces', axis=1)  # obsolete
onehotcols = onehotcols.drop('Feces_Normal', axis=1) # redundant
# 'Abdomen' column
onehotcols.loc[:, 'Abdomen_Normal'] = (onehotcols.loc[:,
              'Abdomen'] == 'Normal').astype(int)
onehotcols.loc[:, 'Abdomen_Abnormal'] = (onehotcols.loc[:,
              'Abdomen'] == 'Abnormal').astype(int)
onehotcols = onehotcols.drop('Abdomen', axis=1)  # obsolete
onehotcols = onehotcols.drop('Abdomen_Normal', axis=1) # redundant
# 'Abdominocentesis_Appearance' column
onehotcols.loc[:, 'AbdominoAppear_Clear'] = (onehotcols.loc[:,
              'Abdominocentesis_Appearance'] == 'Clear').astype(int)
onehotcols.loc[:, 'AbdominoAppear_Cloudy'] = (onehotcols.loc[:,
              'Abdominocentesis_Appearance'] == 'Cloudy').astype(int)
onehotcols.loc[:, 'AbdominoAppear_Serosang'] = (onehotcols.loc[:,
              'Abdominocentesis_Appearance'] == 'Serosanguinous').astype(int)
onehotcols = onehotcols.drop('Abdominocentesis_Appearance', axis=1)  # obsolete
onehotcols = onehotcols.drop('AbdominoAppear_Clear', axis=1) # redundant
# 'Outcome' column
onehotcols.loc[:, 'Outcome_Lived'] = (onehotcols.loc[:,
              'Outcome'] == 'Lived').astype(int)
onehotcols.loc[:, 'Outcome_Died'] = (onehotcols.loc[:,
              'Outcome'] == 'Died').astype(int)
onehotcols.loc[:, 'Outcome_Euthanized'] = (onehotcols.loc[:,
              'Outcome'] == 'Euthanized').astype(int)
onehotcols = onehotcols.drop('Outcome', axis=1)  # obsolete
onehotcols = onehotcols.drop('Outcome_Lived', axis=1) # redundant
# 'Surgical_Lesion' column
onehotcols.loc[:, 'Surgical_Lesion'] = (onehotcols.loc[:,
              'Surgical_Lesion'] == 'Yes').astype(int)
# 'Lesion_Site' column
onehotcols.loc[:, 'LesionSite_None'] = (onehotcols.loc[:,
              'Lesion_Site'] == 'None').astype(int)
onehotcols.loc[:, 'LesionSite_NonColonLocation'] = (onehotcols.loc[:,
              'Lesion_Site'] == 'Non_Colon_Location').astype(int)
onehotcols.loc[:, 'LesionSite_OtherColonLocation'] = (onehotcols.loc[:,
              'Lesion_Site'] == 'Other_Colon_Location').astype(int)
onehotcols.loc[:, 'LesionSite_SmallIntestine'] = (onehotcols.loc[:,
              'Lesion_Site'] == 'Small_Intestine').astype(int)
onehotcols.loc[:, 'LesionSite_LargeIntestine'] = (onehotcols.loc[:,
              'Lesion_Site'] == 'Large_Intestine').astype(int)
onehotcols = onehotcols.drop('Lesion_Site', axis=1) # obsolete
onehotcols = onehotcols.drop('LesionSite_None', axis=1) # redundant
# 'Lesion_Type' column
onehotcols.loc[:, 'LesionType_None'] = (onehotcols.loc[:,
              'Lesion_Type'] == 'None').astype(int)
onehotcols.loc[:, 'LesionType_Other'] = (onehotcols.loc[:,
              'Lesion_Type'] == 'Other').astype(int)
onehotcols.loc[:, 'LesionType_Simple'] = (onehotcols.loc[:,
              'Lesion_Type'] == 'Simple').astype(int)
onehotcols.loc[:, 'LesionType_Strangulation'] = (onehotcols.loc[:,
              'Lesion_Type'] == 'Strangulation').astype(int)
onehotcols = onehotcols.drop('Lesion_Type', axis=1) # obsolete
onehotcols = onehotcols.drop('LesionType_None', axis=1) # redundant
# 'Lesion_Subtype' column
onehotcols.loc[:, 'LesionSubtype_None'] = (onehotcols.loc[:,
              'Lesion_Subtype'] == 'None').astype(int)
onehotcols.loc[:, 'LesionSubtype_Mechanical'] = (onehotcols.loc[:,
              'Lesion_Subtype'] == 'Mechanical').astype(int)
onehotcols.loc[:, 'LesionSubtype_Paralytic'] = (onehotcols.loc[:,
              'Lesion_Subtype'] == 'Paralytic').astype(int)
onehotcols = onehotcols.drop('Lesion_Subtype', axis=1) # obsolete
onehotcols = onehotcols.drop('LesionSubtype_None', axis=1) # redundant
    
### Finalization ###
## creating horsedf original: horsedf_original
horsedf_original = horsedf.copy()

## recombining numcol and catcol data into clean and numerically imputed dataframe
#horsedf: overwriting uncleaned, unimputed horse dataframe
horsedf = pd.concat([numcols, onehotcols], axis=1)

## Exporting as .csv file
horsedf.to_csv('ConnorHiggins-M02-Dataset.csv')







