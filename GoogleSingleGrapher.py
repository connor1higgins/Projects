# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:10:18 2018

@author: Connor Higgins
"""

from pytrends.request import TrendReq
import matplotlib.pyplot as plt


def GoogleSingleGrapher(kw):
    ## Creating pytrends in american english and time zone PST (-7h)
    pytrends = TrendReq(hl='en-US', tz=420)
    ## Building Payload
    pytrends.build_payload([kw], cat=0, timeframe='today 5-y', geo='US-WA', gprop='')
    ## Creating DataFrames
    interest_over_time = pytrends.interest_over_time()
    regional_interest = pytrends.interest_by_region(resolution='CITY')
    related_topics = pytrends.related_topics()
    related_queries = pytrends.related_queries()
    ## Plotting Interest Over Time
    interest_over_time.plot(figsize=(15, 6))
    plt.xlabel('Date (year)')
    plt.ylabel('Popularity relative to most popular day (%)')
    plt.title('Keyword Trend Over 5 Years')
    plt.show()
    ## Plotting Regional Interest
    region_plot = regional_interest.plot(kind='bar')
    region_plot.set_title('Interest by Region')
    region_plot.set_ylabel('Interest Relative to top Region (%)')
    region_plot.set_xlabel('Regions')
    region_plot.set_xticklabels(('Portland', 'SEATAC', 'Spokane', 'Yakima'),
                                rotation = 45)
    plt.show()
    ## Plotting Relatedness of Various Topics
    topics_plot = related_topics[kw][['title', 'value']].plot(kind='bar')
    topics_plot.set_title('Related Topics')
    topics_plot.set_ylabel('Relatedness Relative to Top Connection (%)')
    topics_plot.set_xlabel('Topics')
    topics_plot.set_xticklabels(list(related_topics[kw].loc[:, 'title']),
                                rotation = 45)
    plt.show()
    ## Plotting Rising Queries
    rising_queries_plot = related_queries[kw]['rising'].plot(kind='bar')
    rising_queries_plot.set_title('Rising Queries')
    rising_queries_plot.set_ylabel('Number of Queries (#)')
    rising_queries_plot.set_xlabel('Topics')
    rising_queries_plot.set_xticklabels(list(related_queries[kw]['rising'
                                             ].loc[:, 'query']), rotation = 90)
    plt.show()
    ## Plotting Top Queries
    top_queries_plot = related_queries[kw]['top'].plot(kind='bar')
    top_queries_plot.set_title('Top Queries')
    top_queries_plot.set_ylabel('Relatedness Relative to Top Connection (%)')
    top_queries_plot.set_xlabel('Topics')
    top_queries_plot.set_xticklabels(list(related_queries[kw]['top'
                                          ].loc[:, 'query']), rotation = 90)
    plt.show()