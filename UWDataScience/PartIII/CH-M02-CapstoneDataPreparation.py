# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 15:37:26 2018

@author: Connor Higgins
"""


# Datasets Location
github_folder_url = (
        "https://raw.githubusercontent.com/connor1higgins/"
        "Projects/master/UWDataScience/PartIII/"
        )
sea_energy_2015_file = "2015-building-energy-benchmarking.csv"
# Source: https://www.kaggle.com/city-of-seattle/sea-building-energy-benchmarking
sea_energy_2016_file = "2016-building-energy-benchmarking.csv"
# Source: https://www.kaggle.com/city-of-seattle/sea-building-energy-benchmarking
sea_energy_2017_file = "2017-building_energy_benchmarking.csv"
# Source: https://data.seattle.gov/dataset/2017-Building-Energy-Benchmarking/
chicago_energy_file = "Chicago_Energy_Benchmarking.csv"
# Source: https://catalog.data.gov/dataset/chicago-energy-benchmarking

# Importing libraries
import pandas as pd
import numpy as np

# Reading .csv files as DataFrames
df_sea2015 = pd.read_csv(github_folder_url + sea_energy_2015_file)
df_sea2016 = pd.read_csv(github_folder_url + sea_energy_2016_file)
df_sea2017 = pd.read_csv(github_folder_url + sea_energy_2017_file)
df_chicago = pd.read_csv(github_folder_url + chicago_energy_file)

# Cleaning df_sea2015
# Removing known outliers from dataset
df_sea2015['Outlier'] = df_sea2015['Outlier'].astype(str)
df_sea2015 = df_sea2015[~df_sea2015['Outlier'].str.contains('Outlier')]
# Removing entries without SiteEnergyUseWN(kBtu) values
df_sea2015 = df_sea2015[~df_sea2015['SiteEnergyUseWN(kBtu)'].isnull()]
# Removing entries without recorded ENERGYSTARScore values
df_sea2015 = df_sea2015[~df_sea2015['ENERGYSTARScore'].isnull()]
df_sea2015.reset_index(inplace=True, drop=True)
# Converting 'DataYear' and 'CouncilDistrictCode' features from int dtype to str
df_sea2015['DataYear'] = df_sea2015['DataYear'].astype(str)
df_sea2015['CouncilDistrictCode'] = df_sea2015['CouncilDistrictCode'].astype(str)
# Removing newline characters from PrimaryPropertyType values
df_sea2015['PrimaryPropertyType'] = \
    df_sea2015['PrimaryPropertyType'].str.replace('\n', '')
# Renaming SPS-District K-12 to K-12 School for PrimaryPropertyType
df_sea2015['PrimaryPropertyType'].replace(
          {'SPS-District K-12' : 'K-12 School'}, inplace=True)
# Separating Location feature into Latitude and Longitude features
import json
df_sea2015['Location'] = df_sea2015['Location'].str.split('human').str[0]
df_sea2015['Location'] = df_sea2015['Location'].str.slice(stop=-3) + '}'
df_sea2015['Location'] = df_sea2015['Location'].str.replace("\'", "\"")
loclist = df_sea2015['Location'].tolist()
loclist = [json.loads(string) for string in loclist]
df_sea2015locs = pd.DataFrame(loclist)
df_sea2015locs['latitude'] = df_sea2015locs['latitude'].astype(float)
df_sea2015locs['longitude'] = df_sea2015locs['longitude'].astype(float)
df_sea2015 = pd.concat([df_sea2015, df_sea2015locs], axis=1)
df_sea2015.rename(index=str, inplace=True,
                  columns={'latitude':'Latitude', 'longitude': 'Longitude'})
# Converting ENERGYSTARScore feature to categorical variable: ENERGYSTARQuintile
bins = [0, 20, 40, 60, 80, np.inf]
names = ['quin0', 'quin1', 'quin2', 'quin3', 'quin4']
df_sea2015['ENERGYSTARQuintile'] = \
    pd.cut(df_sea2015['ENERGYSTARScore'], bins, labels=names)

# Cleaning df_sea2016
# Removing known outliers from dataset
df_sea2016['Outlier'] = df_sea2016['Outlier'].astype(str)
df_sea2016 = df_sea2016[~df_sea2016['Outlier'].str.contains('outlier')]
# Removing entries without recorded ENERGYSTARScore values
df_sea2016 = df_sea2016[~df_sea2016['ENERGYSTARScore'].isnull()]
df_sea2016.reset_index(inplace=True, drop=True)
# Converting 'DataYear' and 'CouncilDistrictCode' features from int dtype to str
df_sea2016['DataYear'] = df_sea2016['DataYear'].astype(str)
df_sea2016['CouncilDistrictCode'] = df_sea2016['CouncilDistrictCode'].astype(str)
# Renaming several features to match df_sea2015
df_sea2016.rename(index=str, inplace=True,
    columns={'GHGEmissionsIntensity':'GHGEmissionsIntensity(kgCO2e/ft2)',
             'TotalGHGEmissions': 'GHGEmissions(MetricTonsCO2e)'})
# Renaming several PrimaryPropertyType values to match df_sea2015
conv = {'Warehouse': 'Non-Refrigerated Warehouse',
        'Supermarket / Grocery Store': 'Supermarket/Grocery Store',
        'Office': 'Small- and Mid-Sized Office',
        'Residence Hall': 'Residence Hall/Dormitory'}
df_sea2016['PrimaryPropertyType'].replace(conv, inplace=True)
# Converting ENERGYSTARScore feature to categorical variable: ENERGYSTARQuintile
bins = [0, 20, 40, 60, 80, np.inf]
names = ['quin0', 'quin1', 'quin2', 'quin3', 'quin4']
df_sea2016['ENERGYSTARQuintile'] = \
    pd.cut(df_sea2016['ENERGYSTARScore'], bins, labels=names)

# Cleaning df_sea2017
# Removing known outliers from dataset
df_sea2017['Outlier'] = df_sea2017['Outlier'].astype(str)
df_sea2017 = df_sea2017[~df_sea2017['Outlier'].str.contains('outlier')]
# Removing entries without recorded ENERGYSTARScore values
df_sea2017 = df_sea2017[~df_sea2017['ENERGYSTARScore'].isnull()]
# Removing entires without recorded CouncilDistrictCode values
df_sea2017 = df_sea2017[~df_sea2017['CouncilDistrictCode'].isnull()]
df_sea2017.reset_index(inplace=True, drop=True)
# Converting 'DataYear' and 'CouncilDistrictCode' features from int dtype to str
df_sea2017['DataYear'] = df_sea2017['DataYear'].astype(str)
df_sea2017['CouncilDistrictCode'] = \
    df_sea2017['CouncilDistrictCode'].astype(int).astype(str)
# Renaming several features to match df_sea2015 and df_sea2016
df_sea2017.rename(index=str, inplace=True,
    columns={'GHGEmissionsIntensity':'GHGEmissionsIntensity(kgCO2e/ft2)',
             'TotalGHGEmissions': 'GHGEmissions(MetricTonsCO2e)'})
# Renaming Nonresidental WA to NonResidential for BuildingType feature
df_sea2017['BuildingType'].replace(
        {'Nonresidential WA': 'NonResidential'}, inplace=True)
# Renaming several PrimaryPropertyType values to match df_sea2015 and df_sea2016
conv = {'Warehouse': 'Non-Refrigerated Warehouse',
        'Supermarket / Grocery Store': 'Supermarket/Grocery Store',
        'Residence Hall': 'Residence Hall/Dormitory'}
df_sea2017['PrimaryPropertyType'].replace(conv, inplace=True)
# Converting ENERGYSTARScore feature to categorical variable: ENERGYSTARQuintile
bins = [0, 20, 40, 60, 80, np.inf]
names = ['quin0', 'quin1', 'quin2', 'quin3', 'quin4']
df_sea2017['ENERGYSTARQuintile'] = \
    pd.cut(df_sea2017['ENERGYSTARScore'], bins, labels=names)

# Cleaning df_chicago
# Removing entries without recorded ENERGY STAR Score values
df_chicago = df_chicago[~df_chicago['ENERGY STAR Score'].isnull()]
df_chicago.reset_index(inplace=True, drop=True)
# Renaming several features to match seattle datasets
df_chicago.rename(index=str, inplace=True,
    columns={'Data Year' : 'DataYear',
            'ENERGY STAR Score': 'ENERGYSTARScore',
            'Year Built': 'YearBuilt',
            '# of Buildings': 'NumberofBuildings',
            'Gross Floor Area - Buildings (sq ft)': 'PropertyGFATotal',
            'Site EUI (kBtu/sq ft)': 'SiteEUI(kBtu/sf)',
            'Source EUI (kBtu/sq ft)': 'SourceEUI(kBtu/sf)',
            'Weather Normalized Site EUI (kBtu/sq ft)': 'SiteEUIWN(kBtu/sf)',
            'Weather Normalized Source EUI (kBtu/sq ft)': 'SourceEUIWN(kBtu/sf)',
            'District Steam Use (kBtu)': 'SteamUse(kBtu)',
            'Electricity Use (kBtu)': 'Electricity(kBtu)',
            'Natural Gas Use (kBtu)': 'NaturalGas(kBtu)',
            'Total GHG Emissions (Metric Tons CO2e)': 'GHGEmissions(MetricTonsCO2e)',
            'GHG Intensity (kg CO2e/sq ft)': 'GHGEmissionsIntensity(kgCO2e/ft2)',
            'Primary Property Type' : 'PrimaryPropertyType'})
# Converting 'DataYear' feature from int dtype to str
df_chicago['DataYear'] = df_chicago['DataYear'].astype(str)
# Renaming PrimaryPropertyType value to match seattle datasets
mask = df_chicago['PrimaryPropertyType'].str.contains('Hospital')
df_chicago.loc[mask, 'PrimaryPropertyType'] = 'Hospital'
# Creating 'BuildingType' feature to more closely match groups in seattle datasets
df_chicago['BuildingType'] = ''
NonResidential = ['Office', 'Hotel', 'Retail Store',
                  'Supermarket/Grocery Store', 'Hospital', 'Senior Care Community',
                  'Worship Facility', 'Medical Office', 'Wholesale Club/Supercenter',
                  'Mixed Use Property', 'Financial Office', 'Courthouse',
                  'Bank Branch', 'Distribution Center']
mask = df_chicago['PrimaryPropertyType'].isin(NonResidential)
df_chicago.loc[mask, 'BuildingType'] = 'NonResidential'
Campus = ['K-12 School', 'Residence Hall/Dormitory']
mask = df_chicago['PrimaryPropertyType'].isin(Campus)
df_chicago.loc[mask, 'BuildingType'] = 'Campus'
mask = df_chicago['PrimaryPropertyType'] == 'Multifamily Housing'
df_chicago.loc[mask, 'BuildingType'] = 'Multifamily Housing'
# Creating 'SiteEnergyUse(kBtu)' feature as the sum of all energy features
df_chicago['SiteEnergyUse(kBtu)'] = \
    df_chicago['Electricity(kBtu)'].fillna(0) \
    + df_chicago['NaturalGas(kBtu)'].fillna(0) \
    + df_chicago['SteamUse(kBtu)'].fillna(0) \
    + df_chicago['District Chilled Water Use (kBtu)'].fillna(0) \
    + df_chicago['All Other Fuel Use (kBtu)'].fillna(0)
# Converting ENERGYSTARScore feature into categorical variable: ENERGYSTARQuintile
bins = [0, 20, 40, 60, 80, np.inf]
names = ['quin0', 'quin1', 'quin2', 'quin3', 'quin4']
df_chicago['ENERGYSTARQuintile'] = \
    pd.cut(df_chicago['ENERGYSTARScore'], bins, labels=names)

# Reducing datasets to useable columns
df_sea2015 = df_sea2015[[
        'DataYear', 'BuildingType', 'PrimaryPropertyType',
        'YearBuilt', 'NumberofBuildings', 'NumberofFloors',
        'PropertyGFATotal', 'PropertyGFAParking', 'PropertyGFABuilding(s)',
        'ENERGYSTARScore', 'SiteEUI(kBtu/sf)', 'SiteEUIWN(kBtu/sf)',
        'SourceEUI(kBtu/sf)', 'SourceEUIWN(kBtu/sf)', 'SiteEnergyUse(kBtu)',
        'SiteEnergyUseWN(kBtu)', 'SteamUse(kBtu)', 'Electricity(kWh)',
        'Electricity(kBtu)', 'NaturalGas(therms)', 'NaturalGas(kBtu)',
        'GHGEmissions(MetricTonsCO2e)', 'GHGEmissionsIntensity(kgCO2e/ft2)',
        'Latitude', 'Longitude', 'ENERGYSTARQuintile', 'CouncilDistrictCode'
        ]]
df_sea2016 = df_sea2016[[
        'DataYear', 'BuildingType', 'PrimaryPropertyType',
        'YearBuilt', 'NumberofBuildings', 'NumberofFloors',
        'PropertyGFATotal', 'PropertyGFAParking', 'PropertyGFABuilding(s)',
        'ENERGYSTARScore', 'SiteEUI(kBtu/sf)', 'SiteEUIWN(kBtu/sf)',
        'SourceEUI(kBtu/sf)', 'SourceEUIWN(kBtu/sf)', 'SiteEnergyUse(kBtu)',
        'SiteEnergyUseWN(kBtu)', 'SteamUse(kBtu)', 'Electricity(kWh)',
        'Electricity(kBtu)', 'NaturalGas(therms)', 'NaturalGas(kBtu)',
        'GHGEmissions(MetricTonsCO2e)', 'GHGEmissionsIntensity(kgCO2e/ft2)',
        'Latitude', 'Longitude', 'ENERGYSTARQuintile', 'CouncilDistrictCode'
        ]]
df_sea2017 = df_sea2017[[
        'DataYear', 'BuildingType', 'PrimaryPropertyType',
        'YearBuilt', 'NumberofBuildings', 'NumberofFloors',
        'PropertyGFATotal', 'PropertyGFAParking', 'PropertyGFABuilding(s)',
        'ENERGYSTARScore', 'SiteEUI(kBtu/sf)', 'SiteEUIWN(kBtu/sf)',
        'SourceEUI(kBtu/sf)', 'SourceEUIWN(kBtu/sf)', 'SiteEnergyUse(kBtu)',
        'SiteEnergyUseWN(kBtu)', 'SteamUse(kBtu)', 'Electricity(kWh)',
        'Electricity(kBtu)', 'NaturalGas(therms)', 'NaturalGas(kBtu)',
        'GHGEmissions(MetricTonsCO2e)', 'GHGEmissionsIntensity(kgCO2e/ft2)',
        'Latitude', 'Longitude', 'ENERGYSTARQuintile', 'CouncilDistrictCode'
        ]]
df_chicago = df_chicago[[
        'DataYear', 'BuildingType', 'PrimaryPropertyType',
        'YearBuilt', 'NumberofBuildings', 'PropertyGFATotal',
        'ENERGYSTARScore', 'SiteEUI(kBtu/sf)', 'SiteEUIWN(kBtu/sf)',
        'SourceEUI(kBtu/sf)', 'SourceEUIWN(kBtu/sf)', 'SiteEnergyUse(kBtu)',
        'Electricity(kBtu)', 'NaturalGas(kBtu)', 'GHGEmissions(MetricTonsCO2e)',
        'GHGEmissionsIntensity(kgCO2e/ft2)', 'Latitude', 'Longitude',
        'ENERGYSTARQuintile'
        ]]

# Separating numeric, categorical, and target features
df_tar_sea2015 = df_sea2015[['ENERGYSTARQuintile', 'ENERGYSTARScore']].copy()
df_cat_sea2015 = df_sea2015[['DataYear', 'BuildingType',
                             'PrimaryPropertyType', 'CouncilDistrictCode']].copy()
df_num_sea2015 = df_sea2015.drop(df_tar_sea2015, axis=1).copy()
df_num_sea2015 = df_num_sea2015.drop(df_cat_sea2015, axis=1)
df_tar_sea2016 = df_sea2016[['ENERGYSTARQuintile', 'ENERGYSTARScore']].copy()
df_cat_sea2016 = df_sea2016[['DataYear', 'BuildingType',
                             'PrimaryPropertyType', 'CouncilDistrictCode']].copy()
df_num_sea2016 = df_sea2016.drop(df_tar_sea2016, axis=1).copy()
df_num_sea2016 = df_num_sea2016.drop(df_cat_sea2016, axis=1)
df_tar_sea2017 = df_sea2017[['ENERGYSTARQuintile', 'ENERGYSTARScore']].copy()
df_cat_sea2017 = df_sea2017[['DataYear', 'BuildingType',
                             'PrimaryPropertyType', 'CouncilDistrictCode']].copy()
df_num_sea2017 = df_sea2017.drop(df_tar_sea2017, axis=1).copy()
df_num_sea2017 = df_num_sea2017.drop(df_cat_sea2017, axis=1)
df_tar_chicago = df_chicago[['ENERGYSTARQuintile', 'ENERGYSTARScore']].copy()
df_cat_chicago = df_chicago[['DataYear', 'BuildingType', 'PrimaryPropertyType']].copy()
df_num_chicago = df_chicago.drop(df_tar_chicago, axis=1).copy()
df_num_chicago = df_num_chicago.drop(df_cat_chicago, axis=1)


# Imputing missing numeric data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df_num_sea2015 = pd.DataFrame(
        data=imputer.fit_transform(df_num_sea2015.values),
        columns=df_num_sea2015.columns
        )
df_num_sea2016 = pd.DataFrame(
        data=imputer.fit_transform(df_num_sea2016.values),
        columns=df_num_sea2016.columns
        )
df_num_sea2017 = pd.DataFrame(
        data=imputer.fit_transform(df_num_sea2017.values),
        columns=df_num_sea2017.columns
        )
df_num_chicago = pd.DataFrame(
        data=imputer.fit_transform(df_num_chicago.values),
        columns=df_num_chicago.columns
        )

# Winsorizing certain numeric features
from scipy.stats import mstats
winsor_seacols = [
        'PropertyGFATotal', 'PropertyGFAParking', 'PropertyGFABuilding(s)',
        'SiteEUI(kBtu/sf)', 'SiteEUIWN(kBtu/sf)', 'SourceEUI(kBtu/sf)',
        'SourceEUIWN(kBtu/sf)', 'SiteEnergyUse(kBtu)', 'SiteEnergyUseWN(kBtu)',
        'SteamUse(kBtu)', 'Electricity(kWh)', 'Electricity(kBtu)',
        'NaturalGas(therms)', 'NaturalGas(kBtu)',
        'GHGEmissions(MetricTonsCO2e)', 'GHGEmissionsIntensity(kgCO2e/ft2)',
        ]
for column in winsor_seacols:
    df_num_sea2015[column] = mstats.winsorize(
            df_num_sea2015[column].values, limits=0.01
            )
    df_num_sea2016[column] = mstats.winsorize(
            df_num_sea2016[column].values, limits=0.01
            )
    df_num_sea2017[column] = mstats.winsorize(
            df_num_sea2017[column].values, limits=0.01
            )
winsor_chicols = [
        'PropertyGFATotal', 'SiteEUI(kBtu/sf)', 'SiteEUIWN(kBtu/sf)',
        'SourceEUI(kBtu/sf)', 'SourceEUIWN(kBtu/sf)', 'SiteEnergyUse(kBtu)',
        'Electricity(kBtu)', 'NaturalGas(kBtu)',
        'GHGEmissions(MetricTonsCO2e)', 'GHGEmissionsIntensity(kgCO2e/ft2)',
        ]
for column in winsor_chicols:
    df_num_chicago[column] = mstats.winsorize(
            df_num_chicago[column].values, limits=0.01
            )

# Recombining features for pre-onehotencoding/pre-normalization .csv exports
df_cat_sea2015.index = df_cat_sea2015.index.astype(int)
df_num_sea2015.index = df_num_sea2015.index.astype(int)
df_tar_sea2015.index = df_tar_sea2015.index.astype(int)
dfpre_sea2015 = pd.concat([df_cat_sea2015, df_num_sea2015, df_tar_sea2015], axis=1)
df_cat_sea2016.index = df_cat_sea2016.index.astype(int)
df_num_sea2016.index = df_num_sea2016.index.astype(int)
df_tar_sea2016.index = df_tar_sea2016.index.astype(int)
dfpre_sea2016 = pd.concat([df_cat_sea2016, df_num_sea2016, df_tar_sea2016], axis=1)
df_cat_sea2017.index = df_cat_sea2017.index.astype(int)
df_num_sea2017.index = df_num_sea2017.index.astype(int)
df_tar_sea2017.index = df_tar_sea2017.index.astype(int)
dfpre_sea2017 = pd.concat([df_cat_sea2017, df_num_sea2017, df_tar_sea2017], axis=1)
dfpre_seattle = pd.concat([dfpre_sea2015, dfpre_sea2016, dfpre_sea2017], axis=0)
dfpre_seattle.to_csv('Energy_Benchmarking_Seattle_pre_norm_and_enc.csv')
df_cat_chicago.index = df_cat_chicago.index.astype(int)
df_num_chicago.index = df_num_chicago.index.astype(int)
df_tar_chicago.index = df_tar_chicago.index.astype(int)
dfpre_chicago = pd.concat([df_cat_chicago, df_num_chicago, df_tar_chicago], axis=1)
dfpre_chicago.to_csv('Energy_Benchmarking_Chicago_pre_norm_and_enc.csv')

# Logarithmizing certain numeric features
for column in winsor_seacols:
    dfpre_seattle[column].replace({0: np.nan}, inplace=True)
    dfpre_seattle[column] = np.log(dfpre_seattle[column])
    dfpre_seattle[column].replace({np.nan: 0}, inplace=True)
for column in winsor_chicols:
    dfpre_chicago[column].replace({0: np.nan}, inplace=True)
    dfpre_chicago[column] = np.log(dfpre_chicago[column])
    dfpre_chicago[column].replace({np.nan: 0}, inplace=True)

# Normalizing numeric features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_cols_sea = list(dfpre_seattle.select_dtypes(include=['number']).columns)
num_cols_sea.remove('ENERGYSTARScore')
for column in num_cols_sea:
    dfpre_seattle[column] = \
    scaler.fit_transform(dfpre_seattle[column].values.reshape(-1, 1))
num_cols_chi = list(dfpre_chicago.select_dtypes(include=['number']).columns)
num_cols_chi.remove('ENERGYSTARScore')
for column in num_cols_chi:
    dfpre_chicago[column] = \
    scaler.fit_transform(dfpre_chicago[column].values.reshape(-1, 1))

# One hot encoding categorical features; creating final dataframes
cat_cols_sea = dfpre_seattle.select_dtypes(include=['object']).columns
df_seattle = pd.get_dummies(dfpre_seattle, columns=cat_cols_sea)
cat_cols_chi = dfpre_chicago.select_dtypes(include=['object']).columns
df_chicago = pd.get_dummies(dfpre_chicago, columns=cat_cols_chi)

# Reordering several columns and exporting datasets as .csv files
new_order_sea = [23, 24, 25] + list(range(21)) + list(range(26, 60)) + [21, 22]
df_seattle = df_seattle[df_seattle.columns[new_order_sea]]
df_seattle.to_csv('Energy_Benchmarking_Seattle_clean.csv')
new_order_chi = [16, 17, 18] + list(range(14)) + list(range(19, 39)) + [14, 15]
df_chicago = df_chicago[df_chicago.columns[new_order_chi]]
df_chicago.to_csv('Energy_Benchmarking_Chicago_clean.csv')
