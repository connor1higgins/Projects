# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:10:18 2018

@author: Connor Higgins
"""

from pytrends.request import TrendReq
import matplotlib.pyplot as plt

def GoogleMultiGrapher(kw_list):
    ## Creating pytrends in american english and time zone PST (-7h)
    pytrends = TrendReq(hl='en-US', tz=420)
    ## Building Payload
    pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='US-WA', gprop='')
    ## Creating DataFrames
    interest_over_time = pytrends.interest_over_time()
    regional_interest = pytrends.interest_by_region(resolution='CITY')
    related_queries = pytrends.related_queries()
    ## Plotting Interest Over Time
    interest_over_time.plot(figsize=(15, 6))
    plt.xlabel('Date (year)')
    plt.ylabel('Popularity relative to most popular day (%)')
    plt.title('Keyword Trend Over 5 Years')
    plt.show()
    ## Plotting Regional Interest
    region_plot = regional_interest.plot(kind='bar', figsize=(15, 6))
    region_plot.set_title('Interest by Region')
    region_plot.set_ylabel('Interest Relative to top Region (%)')
    region_plot.set_xlabel('Regions')
    region_plot.set_xticklabels(('Portland', 'SEATAC', 'Spokane', 'Yakima'),
                                rotation = 45)
    plt.show()
    ## Plotting Rising Queries
    for i in kw_list:
        rising_queries_plot = related_queries[i]['rising'].plot(kind='barh', figsize=(15, 6))
        rising_queries_plot.set_title('Rising Queries for {}'.format(i))
        rising_queries_plot.set_xlabel('Number of Queries (#)')
        rising_queries_plot.set_ylabel('Topics')
        rising_queries_plot.set_yticklabels(list(related_queries[i]['rising'
                                                 ].loc[:, 'query']))
        plt.show()
    ## Plotting Top Queries
    for i in kw_list:
        top_queries_plot = related_queries[i]['top'].plot(kind='barh', figsize=(15, 6))
        top_queries_plot.set_title('Top Queries for {}'.format(i))
        top_queries_plot.set_xlabel('Relatedness Relative to Top Connection (%)')
        top_queries_plot.set_ylabel('Topics')
        top_queries_plot.set_yticklabels(list(related_queries[i]['top'
                                              ].loc[:, 'query']))
        plt.show()