# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 08:44:47 2018

@author: Connor Higgins
"""

from googlesearch import search
import time
def GoogleQuery(kw, numberof):
    time.sleep(2) # to prevent annoying lord google
    urls = []
    for url in search(str(kw), stop=int(numberof)):
        urls.append(url)
    return(urls)
