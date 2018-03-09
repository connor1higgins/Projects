# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 09:02:56 2018

@author: Connor Higgins
"""
import requests

""" Takes search term (str) and the number of results required (int). Returns
    tuple list containing url names and urls in search return order """

def BingQuery(search_term, numberof):
    # Key Setup
    #Key 1: c4f4db1eb965418e8e00f82a20a67694
    #Key 2: b1cf48375e024ae5a3049daefe404425
    subscription_key = 'c4f4db1eb965418e8e00f82a20a67694'
    assert subscription_key
    
    # Param Setup
    search_url = 'https://api.cognitive.microsoft.com/bing/v7.0/search'
    headers = {'Ocp-Apim-Subscription-Key' : subscription_key}
    params = {"q": search_term,
              "textDecorations":True,
              "textFormat":"HTML",
              "count": numberof}
    # Request to Bing
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()

    # Unpacking response for urls and url names
    sitenames = []
    siteurls = []
    for i in range(len(search_results['webPages']['value'])):
        sitename = search_results['webPages']['value'][i]['name']
        sitenames.append(sitename)
        siteurl = search_results['webPages']['value'][i]['url']
        siteurls.append(siteurl)
    nameurls = list(zip(sitenames, siteurls))
    
    # Returning tuple of urls and url names in index order
    return nameurls
    
    
    