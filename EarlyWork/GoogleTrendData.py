# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:51:22 2018

@author: Connor Higgins
"""
from pytrends.request import TrendReq

def GoogleTrendData(kw):
    ## Setting up pytrends
    pytrends = TrendReq(hl='en-US', tz=420)
    pytrends.build_payload([kw], cat=0, timeframe='2017-02-28 2018-02-28'
                           , geo='US', gprop='')
    ## Creating inital DataFrames
    interest_over_time = pytrends.interest_over_time()
    regional_interest = pytrends.interest_by_region(resolution='CITY')
    related_topics = pytrends.related_topics()
    related_queries = pytrends.related_queries()
    ## Creating specific DataFrames
    top_queries = related_queries[kw]['top']
    rising_queries = related_queries[kw]['rising']
    top_topics = related_topics[kw][['title', 'type', 'value']]
    ## Creating Dictionary of specific DataFrames
    trendDict = {'Top Queries': top_queries,
                 'Rising Queries': rising_queries,
                 'Related Topics' : top_topics,
                 'Regional Interest' : regional_interest,
                 'Interest Over Time' : interest_over_time}
    return trendDict
