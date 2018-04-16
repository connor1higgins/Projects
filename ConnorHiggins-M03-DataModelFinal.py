# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 13:00:38 2018

@author: Connor Higgins
"""

#ConnorHiggins-M03-DataModelFinal.py

### Objective ############################################################################################
#    Is it possible to accurately predict whether the horses in created test instances will live or die/be 
# euthanized, given their attributes and the attributes and known outcomes of horses within a training set?

### Import statements for libraries and data set #########################################################
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

## loading data set
# url: milestone2.csv file for horse-colic.data uploaded to github repository
url = 'https://raw.githubusercontent.com/connor1higgins/Projects/master/ConnorHiggins-M02-Dataset.csv'
# horsedf: using pd.read_csv to load .csv
horsedf = pd.read_csv(url, sep=',', index_col=0)

### Splitting the data set into training and testing sets ################################################
# separating feature columns and target column into X and y respectively
# X: all columns not describing outcome
X = horsedf.drop(['Outcome_Died', 'Outcome_Euthanized'], axis=1)
# y: combining 'Outcome_Died' and 'Outcome_Euthanized' into one column ('Outcome_Dead')
horsedf['Outcome_Dead'] = horsedf['Outcome_Died'] + horsedf['Outcome_Euthanized']
y = horsedf['Outcome_Dead']
# using sklearn's 'train_test_split' to split data (80-20 train to test ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Classifiers: Training and Performance Metrics ########################################################
## Decision Tree Classifier
# DTclf: random state of 42 (for reproducable results) and minimum samples split of 2 (default)
DTclf = DecisionTreeClassifier(min_samples_split=2, random_state=42)
# applying DTclf to training data
DTclf.fit(X_train, y_train)
# DT_y_pred: obtaining predicted values
DT_y_pred = DTclf.predict(X_test)
# DT_y_prob: obtaining underlying probabilities for predicted values. 
DT_y_prob = DTclf.predict_proba(X_test)
# creating Decision Tree confusion matrix
DT_CM = confusion_matrix(y_test, DT_y_pred)
# obtaining true negatives, false positives, false negatives, and true positives 
DT_tn, DT_fp, DT_fn, DT_tp = DT_CM.flatten()
# printing decorated confusion matrix
print('\n|-|-|-|-|-|-|-|-|-|-|-Decision Tree Classifier-|-|-|-|-|-|-|-|-|-|-|')
print("\nDT_Confusion Matrix\n \
                    | Predicted| Predicted|\n \
                    | Negative | Positive |\n \
  ---------------------------------------------\n \
    Actual Negative |    {:0>2d} TN |    {:0>2d} FP |\n \
    Actual Positive |    {:0>2d} FN |    {:0>2d} TP |\n".format(DT_tn, DT_fp, DT_fn, DT_tp))
# Decision Tree ROC Analysis: False Positive Rate (DT_fpr), True Positive Rate, 
# (DT_tpr) and Probability Thresholds (DT_th)
DT_fpr, DT_tpr, DT_th = roc_curve(y_test, DT_y_prob[:, 1])
DT_ROCAnalysisDF = pd.DataFrame(data={'False Positive Rate' : DT_fpr,
                                      'True Positive Rate' : DT_tpr,
                                      'Probability Threshold' : DT_th})
# obtaining DT_AUC score
DT_AUC = auc(DT_fpr, DT_tpr)
# printing fpr, tpr, th values in ROC Anaylsis dataframe, with AUC score listed as well
print('\nDT_ROC Anaylsis | DT_AUC Score: {}\n\
-------------------------------------------------------------------'.format(DT_AUC))
print(DT_ROCAnalysisDF)
print('-------------------------------------------------------------------\n\n')

## KNeighbors Classifier
# KNclf: nearest neighbors = 5 (default) and 'euclidean' distance metric (default)
KNclf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
# applying KNclf to training data
KNclf.fit(X_train, y_train)
# KN_y_pred: obtaining predicted values
KN_y_pred = KNclf.predict(X_test)
# KN_y_prob: obtaining underlying probabilities for predicted values. 
KN_y_prob = KNclf.predict_proba(X_test)
# creating KNeighbors confusion matrix
KN_CM = confusion_matrix(y_test, KN_y_pred)
# obtaining true negatives, false positives, false negatives, and true positives 
KN_tn, KN_fp, KN_fn, KN_tp = KN_CM.flatten()
# printing decorated confusion matrix
print('\n|-|-|-|-|-|-|-|-|-|-|-|-KNeighbors Classifier-|-|-|-|-|-|-|-|-|-|-|-|')
print("\nKN_Confusion Matrix\n \
                    | Predicted| Predicted|\n \
                    | Negative | Positive |\n \
  ---------------------------------------------\n \
    Actual Negative |    {:0>2d} TN |    {:0>2d} FP |\n \
    Actual Positive |    {:0>2d} FN |    {:0>2d} TP |\n".format(KN_tn, KN_fp, KN_fn, KN_tp))
# KNeighbors ROC Analysis: False Positive Rate (KN_fpr), True Positive Rate, 
# (KN_tpr) and Probability Thresholds (KN_th)
KN_fpr, KN_tpr, KN_th = roc_curve(y_test, KN_y_prob[:, 1])
KN_ROCAnalysisDF = pd.DataFrame(data={'False Positive Rate' : KN_fpr,
                                      'True Positive Rate' : KN_tpr,
                                      'Probability Threshold' : KN_th})
# obtaining KN_AUC score
KN_AUC = auc(KN_fpr, KN_tpr)
# printing fpr, tpr, th values in ROC Anaylsis dataframe, with AUC score listed as well
print('\nKN_ROC Anaylsis | KN_AUC Score: {}\n\
-------------------------------------------------------------------'.format(KN_AUC))
print(KN_ROCAnalysisDF)
print('-------------------------------------------------------------------\n\n')

## Random Forest Classifier
# RFclf: 100 trees (used as a balance between speed and accuracy), 
# random state of 42 (for reproducable results), and a minimum samples split of 2 (default)
RFclf = RandomForestClassifier(n_estimators=100, min_samples_split=2, random_state=42)
# applying RFclf to training data
RFclf.fit(X_train, y_train)
# RF_y_pred: obtaining predicted values
RF_y_pred = RFclf.predict(X_test)
# RF_y_prob: obtaining underlying probabilities for predicted values. 
RF_y_prob = RFclf.predict_proba(X_test)
# creating Random Forest confusion matrix
RF_CM = confusion_matrix(y_test, RF_y_pred)
# obtaining true negatives, false positives, false negatives, and true positives 
RF_tn, RF_fp, RF_fn, RF_tp = RF_CM.flatten()
# printing decorated confusion matrix
print('\n|-|-|-|-|-|-|-|-|-|-|-Random Forest Classifier-|-|-|-|-|-|-|-|-|-|-|')
print("\nRF_Confusion Matrix\n \
                    | Predicted| Predicted|\n \
                    | Negative | Positive |\n \
  ---------------------------------------------\n \
    Actual Negative |    {:0>2d} TN |    {:0>2d} FP |\n \
    Actual Positive |    {:0>2d} FN |    {:0>2d} TP |\n".format(RF_tn, RF_fp, RF_fn, RF_tp))
# Random Forest ROC Analysis: False Positive Rate (RF_fpr), True Positive Rate, 
# (RF_tpr) and Probability Thresholds (RF_th)
RF_fpr, RF_tpr, RF_th = roc_curve(y_test, RF_y_prob[:, 1])
RF_ROCAnalysisDF = pd.DataFrame(data={'False Positive Rate' : RF_fpr,
                                      'True Positive Rate' : RF_tpr,
                                      'Probability Threshold' : RF_th})
# obtaining RF_AUC score
RF_AUC = auc(RF_fpr, RF_tpr)
# printing fpr, tpr, th values in ROC Anaylsis dataframe, with AUC score listed as well
print('\nRF_ROC Anaylsis | RF_AUC Score: {}\n\
-------------------------------------------------------------------'.format(RF_AUC))
print(RF_ROCAnalysisDF)
print('-------------------------------------------------------------------\n\n')

## Gradient Boosting Classifier
# GBclf: 0.2 learning rate (used after testing values ranging from 0.001 to 0.9),
# 1000 trees (used primiarly due to accuracy, with speed still within a reasonable range),
# random_state of 42 (for reproducable results) and a minimum samples split of 2 (default)
GBclf = GradientBoostingClassifier(learning_rate=0.2, n_estimators=1000,
                                   min_samples_split=2, random_state=42)
# applying GBclf to training data
GBclf.fit(X_train, y_train)
# GB_y_pred: obtaining predicted values
GB_y_pred = GBclf.predict(X_test)
# GB_y_prob: obtaining underlying probabilities for predicted values. 
GB_y_prob = GBclf.predict_proba(X_test)
# creating Gradient Boosting confusion matrix
GB_CM = confusion_matrix(y_test, GB_y_pred)
# obtaining true negatives, false positives, false negatives, and true positives 
GB_tn, GB_fp, GB_fn, GB_tp = GB_CM.flatten()
# printing decorated confusion matrix
print('\n|-|-|-|-|-|-|-|-|-|-Gradient Boosting Classifier-|-|-|-|-|-|-|-|-|-|')
print("\nGB_Confusion Matrix\n \
                    | Predicted| Predicted|\n \
                    | Negative | Positive |\n \
  ---------------------------------------------\n \
    Actual Negative |    {:0>2d} TN |    {:0>2d} FP |\n \
    Actual Positive |    {:0>2d} FN |    {:0>2d} TP |\n".format(GB_tn, GB_fp, GB_fn, GB_tp))
# Gradient Boosting ROC Analysis: False Positive Rate (GB_fpr), True Positive Rate, 
# (GB_tpr) and Probability Thresholds (GB_th)
GB_fpr, GB_tpr, GB_th = roc_curve(y_test, GB_y_prob[:, 1])
GB_ROCAnalysisDF = pd.DataFrame(data={'False Positive Rate' : GB_fpr,
                                      'True Positive Rate' : GB_tpr,
                                      'Probability Threshold' : GB_th})
# obtaining RF_AUC score
GB_AUC = auc(GB_fpr, GB_tpr)
# printing fpr, tpr, th values in ROC Anaylsis dataframe, with AUC score listed as well
print('\nGB_ROC Anaylsis | GB_AUC Score: {}\n\
-------------------------------------------------------------------'.format(GB_AUC))
print(GB_ROCAnalysisDF)
print('-------------------------------------------------------------------\n\n')

### Plotting ROC curves using seaborn and matplotlib #####################################################
sns.set(context='paper', style='darkgrid')
plt.figure()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.plot(DT_fpr, DT_tpr, color='blue', label='ROC DTclf curve (area = %0.3f)' % DT_AUC)
plt.plot(KN_fpr, KN_tpr, color='yellow', label='ROC KNclf curve (area = %0.3f)' % KN_AUC)
plt.plot(RF_fpr, RF_tpr, color='red', label='ROC RFclf curve (area = %0.3f)' % RF_AUC)
plt.plot(GB_fpr, GB_tpr, color='green', label='ROC GBclf curve (area = %0.3f)' % GB_AUC)
plt.plot([0, 1], [0, 1], color='black', linestyle=':', label='Reference Line (random clf)')
plt.title('ROC Curves for Horse Colic dataset\
          \n Decision Tree (DT), KNeighbors (KN), Random Forest (RF), and Gradient Boost (GB) Classifiers')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.show()

### Creating horsedf_binned: binning numeric columns within horsedf ######################################
# numcols: pulling all numeric columns from horsedf data set
numcols = horsedf[['Rectal_Temperature', 'Pulse', 'Respiratory_Rate', 
                   'Packed_Cell_Volume', 'Total_Protein']]
# binning into 5 groups according to standard deviation
VeryHigh = numcols.loc[:] >= 3
AboveAVG = (numcols.loc[:] < 3) & (numcols.loc[:] >= 1)
Normal = (numcols.loc[:] < 1) & (numcols.loc[:] > -1)
BelowAVG = (numcols.loc[:] <= -1) & (numcols.loc[:] > -3) 
VeryLow = numcols.loc[:] <= -3
# catcols: creating categorical version of numcols
catcols = numcols.astype(str)
catcols[VeryHigh] = 'VeryHigh'
catcols[AboveAVG] = 'AboveAVG'
catcols[Normal] = 'Normal'
catcols[BelowAVG] = 'BelowAVG'
catcols[VeryLow] = 'VeryLow'
# 'Rectal_Temperature' onehot encoded columns
catcols.loc[:, 'Rect_Temp_VeryLow'] = (catcols.loc[:, 'Rectal_Temperature'] == 'VeryLow').astype(int)
catcols.loc[:, 'Rect_Temp_BelowAVG'] = (catcols.loc[:, 'Rectal_Temperature'] == 'BelowAVG').astype(int)
catcols.loc[:, 'Rect_Temp_Normal'] = (catcols.loc[:, 'Rectal_Temperature'] == 'Normal').astype(int)
catcols.loc[:, 'Rect_Temp_AboveAVG'] = (catcols.loc[:, 'Rectal_Temperature'] == 'AboveAVG').astype(int)
catcols.loc[:, 'Rect_Temp_VeryHigh'] = (catcols.loc[:, 'Rectal_Temperature'] == 'VeryHigh').astype(int)
catcols = catcols.drop('Rectal_Temperature', axis=1) # obsolete
catcols = catcols.drop('Rect_Temp_Normal', axis=1) # redundant
catcols = catcols.drop('Rect_Temp_VeryLow', axis=1) # no values
catcols = catcols.drop('Rect_Temp_VeryHigh', axis=1) # no values
# 'Pulse' onehot encoded columns
catcols.loc[:, 'Pulse_VeryLow'] = (catcols.loc[:, 'Pulse'] == 'VeryLow').astype(int)
catcols.loc[:, 'Pulse_BelowAVG'] = (catcols.loc[:, 'Pulse'] == 'BelowAVG').astype(int)
catcols.loc[:, 'Pulse_Normal'] = (catcols.loc[:, 'Pulse'] == 'Normal').astype(int)
catcols.loc[:, 'Pulse_AboveAVG'] = (catcols.loc[:, 'Pulse'] == 'AboveAVG').astype(int)
catcols.loc[:, 'Pulse_VeryHigh'] = (catcols.loc[:, 'Pulse'] == 'VeryHigh').astype(int)
catcols = catcols.drop('Pulse', axis=1) # obsolete
catcols = catcols.drop('Pulse_Normal', axis=1) # redundant
catcols = catcols.drop('Pulse_VeryLow', axis=1) # no values
catcols = catcols.drop('Pulse_VeryHigh', axis=1) # no values
# 'Respiratory_Rate' onehot encoded columns
catcols.loc[:, 'RespRate_VeryLow'] = (catcols.loc[:, 'Respiratory_Rate'] == 'VeryLow').astype(int)
catcols.loc[:, 'RespRate_BelowAVG'] = (catcols.loc[:, 'Respiratory_Rate'] == 'BelowAVG').astype(int)
catcols.loc[:, 'RespRate_Normal'] = (catcols.loc[:, 'Respiratory_Rate'] == 'Normal').astype(int)
catcols.loc[:, 'RespRate_AboveAVG'] = (catcols.loc[:, 'Respiratory_Rate'] == 'AboveAVG').astype(int)
catcols.loc[:, 'RespRate_VeryHigh'] = (catcols.loc[:, 'Respiratory_Rate'] == 'VeryHigh').astype(int)
catcols = catcols.drop('Respiratory_Rate', axis=1) # obsolete
catcols = catcols.drop('RespRate_Normal', axis=1) # redundant
catcols = catcols.drop('RespRate_VeryLow', axis=1) # no values
# 'Packed_Cell_Volume' onehot encoded columns
catcols.loc[:, 'PkdCellVol_VeryLow'] = (catcols.loc[:, 'Packed_Cell_Volume'] == 'VeryLow').astype(int)
catcols.loc[:, 'PkdCellVol_BelowAVG'] = (catcols.loc[:, 'Packed_Cell_Volume'] == 'BelowAVG').astype(int)
catcols.loc[:, 'PkdCellVol_Normal'] = (catcols.loc[:, 'Packed_Cell_Volume'] == 'Normal').astype(int)
catcols.loc[:, 'PkdCellVol_AboveAVG'] = (catcols.loc[:, 'Packed_Cell_Volume'] == 'AboveAVG').astype(int)
catcols.loc[:, 'PkdCellVol_VeryHigh'] = (catcols.loc[:, 'Packed_Cell_Volume'] == 'VeryHigh').astype(int)
catcols = catcols.drop('Packed_Cell_Volume', axis=1) # obsolete
catcols = catcols.drop('PkdCellVol_Normal', axis=1) # redundant
catcols = catcols.drop('PkdCellVol_VeryHigh', axis=1) # no values
# 'Total_Protein' column onehot encoded columns
catcols.loc[:, 'TotalProtein_VeryLow'] = (catcols.loc[:, 'Total_Protein'] == 'VeryLow').astype(int)
catcols.loc[:, 'TotalProtein_BelowAVG'] = (catcols.loc[:, 'Total_Protein'] == 'BelowAVG').astype(int)
catcols.loc[:, 'TotalProtein_Normal'] = (catcols.loc[:, 'Total_Protein'] == 'Normal').astype(int)
catcols.loc[:, 'TotalProtein_AboveAVG'] = (catcols.loc[:, 'Total_Protein'] == 'AboveAVG').astype(int)
catcols.loc[:, 'TotalProtein_VeryHigh'] = (catcols.loc[:, 'Total_Protein'] == 'VeryHigh').astype(int)
catcols = catcols.drop('Total_Protein', axis=1) # obsolete
catcols = catcols.drop('TotalProtein_Normal', axis=1) # redundant
catcols = catcols.drop('TotalProtein_VeryLow', axis=1) # no values
catcols = catcols.drop('TotalProtein_VeryHigh', axis=1) # no values
## horsedf_binned: new version of horsedf where all numeric columns are now categorical
horsedf_binned = pd.concat([catcols, horsedf.drop(numcols.columns, axis=1)], axis=1)

### Comparing winning classifier performance on binned /non-binned data sets #############################
## splitting the horsedf_binned data set into training and testing sets
# separating feature columns and target column into Xbin and ybin respectively
# Xbin: all columns not describing outcome
Xbin = horsedf_binned.drop(['Outcome_Died', 'Outcome_Euthanized', 'Outcome_Dead'], axis=1)
# ybin: combining 'Outcome_Died' and 'Outcome_Euthanized' into one column ('Outcome_Dead')
horsedf_binned['Outcome_Dead'] = horsedf_binned['Outcome_Died'] + horsedf_binned['Outcome_Euthanized']
ybin = horsedf_binned['Outcome_Dead']
# using sklearn's 'train_test_split' to split data (80-20 train to test ratio)
Xbin_train, Xbin_test, ybin_train, ybin_test = train_test_split(Xbin, ybin, test_size=0.2, random_state=30)
# RFbinclf: 100 trees (used as a balance between speed and accuracy), 
# random state of 30 (for reproducable results), and a minimum samples split of 2 (default)
RFbinclf = RandomForestClassifier(n_estimators=100, min_samples_split=2, random_state=30)
# applying RFbinclf to training data
RFbinclf.fit(Xbin_train, ybin_train)
# RFbin_y_pred: obtaining predicted values
RFbin_y_pred = RFbinclf.predict(Xbin_test)
# RFbin_y_prob: obtaining underlying probabilities for predicted values. 
RFbin_y_prob = RFbinclf.predict_proba(Xbin_test)
# creating RFbinclf confusion matrix
RFbin_CM = confusion_matrix(ybin_test, RFbin_y_pred)
# obtaining true negatives, false positives, false negatives, and true positives 
RFbin_tn, RFbin_fp, RFbin_fn, RFbin_tp = RFbin_CM.flatten()
# printing decorated confusion matrix
print('\n|-|-|-|-|-|-Random Forest Classifier for binned data set-|-|-|-|-|-|')
print("\nRFbin_Confusion Matrix\n \
                    | Predicted| Predicted|\n \
                    | Negative | Positive |\n \
  ---------------------------------------------\n \
    Actual Negative |    {:0>2d} TN |    {:0>2d} FP |\n \
    Actual Positive |    {:0>2d} FN |    {:0>2d} TP |\n".format(RFbin_tn, RFbin_fp, RFbin_fn, RFbin_tp))
# Random Forest on Binned data set ROC Analysis: False Positive Rate (RFbin_fpr), 
# True Positive Rate, (RFbin_tpr) and Probability Thresholds (RFbin_th)
RFbin_fpr, RFbin_tpr, RFbin_th = roc_curve(ybin_test, RFbin_y_prob[:, 1])
RFbin_ROCAnalysisDF = pd.DataFrame(data={'False Positive Rate' : RFbin_fpr,
                                         'True Positive Rate' : RFbin_tpr,
                                         'Probability Threshold' : RFbin_th})
# obtaining RFbin_AUC score
RFbin_AUC = auc(RFbin_fpr, RFbin_tpr)
# printing fpr, tpr, th values in ROC Anaylsis dataframe, with AUC score listed as well
print('\nRFbin_ROC Anaylsis | RFbin_AUC Score: {}\n\
-------------------------------------------------------------------'.format(RFbin_AUC))
print(RFbin_ROCAnalysisDF)
print('-------------------------------------------------------------------\n\n')

## Plotting ROC curves for Random Forest Classifer on binned and nonbinned data sets
plt.figure()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.plot(RF_fpr, RF_tpr, color='red', label='ROC RFclf curve (area = %0.3f)' % RF_AUC)
plt.plot(RFbin_fpr, RFbin_tpr, color='purple', label='ROC RFbinclf curve (area = %0.3f)' % RFbin_AUC)
plt.plot([0, 1], [0, 1], color='black', linestyle=':', label='Reference Line (random clf)')
plt.title('ROC Curves for Horse Colic dataset\
          \n Random Forest Classifer on horsedf (RFclf) and horsedf_binned (RFbinclf)') 
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.show()

### Summary & Conclusions ################################################################################ 
#    The code above imports a cleaned version of the horse-colic dataset from github as DataFrame horsedf.
# horsedf is then split into feature 'X' and target 'y' variables, then further split into training and 
# testing variables using sklearn's train_test_split function. 
#
#    As the objective is to predict a binary outcome for horses in the test set, 4 different sklearn 
# classifiers/ensembles were used: a DecisionTreeClassifier with a random state of 42 (for reproducable
# results); a KNeighborsClassifier with default parameters; a RandomForestClassifier with 100 trees 
# (for increased accuracy) and a random state of 42 (for reproducable results); and a 
# GradientBoostingClassifier with a 0.2 learning rate (used after testing values ranging from 0.001 to 
# 0.9), 1000 trees (for increased accuracy, with speed still within a reasonable range), and random_state 
# of 42 (for reproducable results).
#
#   For each of these four classifiers/ensembles, they are fitted to the X_train and y_train series; then,
# they are applied to the X_test series to produce an array of predicted values (y_pred) and an array of
# underlying  probabilities (y_prob). Using y_test and y_pred, a confusion matrix is created and printed. 
# Using y_test, y_prob[:, 1], and the roc_curve function, the false positive rate (fpr), true positive rate
# (tpr), and probability thresholds (th) are obtained and compiled into a printed DataFrame 
# (ROCAnalysisDF), along with the AUC score obtained using fpr, tpr, and the auc function.
#
#   Next, using matplotlib.pyplot and seaborn modules, the ROC Curves for these four classifiers are
# plotted against that of a purely random classifier (Figure 1). Overall, the RandomForestClassifier
# performed the best, as evident by its relatively high auc score (0.822, see legend), though it performed
# slightly worse in comparison to the GradientBoostClassifier when the probability threshold was either
# very high or very low. Conversely, the single DecisionTreeClassifier performed the worst of the four,
# (0.607, see legend), though still moderately better than a purely random classifer (0.50). 
#
#   As a follow-up, in order to determine whether binning and one-hot encoding numeric data would increase
# the auc score of the winning classifier, the numeric columns of horsedf were converted into categorical
# columns containing strings referencing the data's standard deviations away from the mean (with all
# numeric data originally being standardized). These categorical columns were then converted into one-hot 
# encoded columns, with redundant, obsolete, or empty columns being dropped. Finally, these transformed
# columns were concatenated with the originally one-hot encoded columns to form a newly binned data set:
# horsedf_binned. 
#
#   As with the original horsedf dataset, horsedf_binned is then split into feature 'X' and target 'y' 
# variables, then further split into training and testing variables using sklearn's train_test_split 
# function. Another Random Forest Classifier (RFbinclf) is applied to the train and test sets in the same 
# manner as the original Random Forest Classifier (RFclf), and the results are printed in the same format
# as the other classifiers/ensembles. Finally, the ROC Curves for the RandomForestClassifier on both
# data set variants were plotted (Figure 2) along with their auc_scores (see legend). Overall, the 
# RandomForestClassifier performed slightly better with the horsedf_binned dataset (0.846) than with the
# original horsedf dataset (0.822). Only when a very low proability threshold was set did the 
# RandomForestClassfieir perform better with horsedf (and with both numeric and categorical data, rather
# than purely categorical data).
#
#  In conclusion, using a RandomForestClassifier with 100 trees on a data set containing purely one-hot
# encoded data and columns, we were able to achieve a reasonably high auc score (~84.6%) when predicting 
# whether a horses, in a created test set, would live or die/be euthanized based on a collection of their
# health attributes, and the attributes and known outcomes of horses within a training set. This score 
# could likely improve with a more rigorous analysis of RandomForestClassifier parameters and various
# methods for accurately binning and one-hot encoding numeric data.


