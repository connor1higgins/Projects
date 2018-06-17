# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 10:48:20 2018

@author: Connor Higgins
"""

from googlesearch import search
import re

""" Takes a query keyword (search term), a website (name in url), and an int 
value (numberof); pulls numberof urls from google search for query keyword; and
counts the total number of urls that contained the website name, returning the
result as a formatted string """

def GoogleQuery2SiteMatcher(kw, site, numberof):
    urls = []
    for url in search(str(kw), stop=int(numberof)):
        urls.append(url)
    counts = 0
    for i in urls:
        m = re.findall(site, i)
        try: 
            if len(m) > 0:
                counts += 1
        except AttributeError:
            pass
    return 'For the query keyword {}, the site {} shows up {} times in ' \
            '{} searches'.format(kw, site, counts, numberof)
