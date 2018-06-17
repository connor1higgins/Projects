# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:20:55 2018

@author: Connor Higgins
"""
from pytrends.request import TrendReq
import numpy as np

def GoogleTopQuery(kw):
    ## Setting up pytrends
    pytrends = TrendReq(hl='en-US', tz=420)
    pytrends.build_payload([kw], cat=0, timeframe='2017-02-28 2018-02-28'
                           , geo='', gprop='')
    related_queries = pytrends.related_queries()
    try:
        top_queries = list(related_queries[kw]['top']['query'])
    except TypeError:
        top_queries = ['str'] * 25 # Dummy list if top_query data is not found
    return top_queries


#############
    
def GoogleQueryIterator(kw):
    FirstList = GoogleTopQuery(kw) # performing an initial query with kw
    TopQueryArray = np.array(FirstList.copy()).transpose()
    for i in FirstList: # performing query on all top_queries for kw
        QueryList = GoogleTopQuery(i)
        TopQueryArray = np.hstack((TopQueryArray, QueryList))
    TopQueryList = TopQueryArray.tolist()
    result = []
    for i in TopQueryList:
        if i != 'str': # removing dummy list(s) from result
            result.append(i)
    return result

##############

def List2String(querylist): # for a wordcloud friendly output for GoogleQueryIterator
    longstring = ''
    for i in querylist:
        longstring = longstring + ' ' + i
    return longstring
        
