# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:45:17 2018

@author: Connor Higgins
"""
import pandas as pd
import re


# single_parse_analysis.py

def Single_Parse_Analysis(df):

    # changing all values to strings
    df = df.astype(str)

    # matching any values that aren't a-z characters or spaces
    abc = re.compile('[^a-zA-Z ]')

    # using abc to remove any values that aren't a-z characters or spaces
    # also, lowercasing 
    for i in range(len(df.columns)):
        for j in range(len(df)):
            df.loc[j][i] = df.loc[j][i].lower()
            df.loc[j][i] = abc.sub('', df.loc[j][i])

    # counts for each column
    # also, splitting words using spaces
    sitetitle_count = df['Site Title'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    firstheader_count = df['1st Headers'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    secondheader_count = df['2nd Headers'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    thirdheader_count = df['3rd Headers'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    fourthheader_count =  df['4th Headers'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    metakeywords_count = df['Meta Keywords'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    italics_count = df['Italics'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    bolds_count = df['Bolds'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    emphasized_count = df['Emphasized'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    paragraphs_count = df['Paragraphs'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    centeredtext_count = df['Centered Text'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    sectiontext_count = df['Section Text'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    spantext_count = df['Span Text'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    tableheadertext_count = df['Table Header Text'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    tablecelltext_count = df['Table Cell Text'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    itemizedtext_count = df['Itemized Text'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    description_count = df['Description'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    hyperlinks_count = df['Hyperlinks'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    imagenames_count = df['Image Names'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    imagelinks_count = df['Image Links'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    videolinks_count = df['Video Links'].apply(
            lambda x: pd.value_counts(x.split(' '))
            ).sum(axis=0).sort_values(ascending=False)

    # combining all counts into a single list
    total_count = pd.concat([sitetitle_count, firstheader_count,
                             secondheader_count, thirdheader_count,
                             fourthheader_count, metakeywords_count,
                             italics_count, bolds_count,
                             emphasized_count, paragraphs_count,
                             centeredtext_count, sectiontext_count,
                             spantext_count, tableheadertext_count,
                             tablecelltext_count, itemizedtext_count,
                             description_count, hyperlinks_count,
                             imagenames_count, imagelinks_count,
                             videolinks_count], axis=0)
    
    # grouping words in total count and summing their total counts 
    total_count = total_count.groupby(total_count.index, axis=0
                                      ).sum().sort_values(ascending=False)
    
    return(total_count)