# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 09:47:43 2018

@author: Connor Higgins
"""

import gensim
from nltk.tokenize import word_tokenize
from pytrends.request import TrendReq
import numpy as np

""" Takes a list of strings and a keyword, determines the top google queries 
    related to the keyword, determines the queries of those queries, packages all
    queries and queries of queries into a single string, compares this querystring
    against the list of strings using LSA, converts array of similarity values into
    a single aggregate value, and converts this value into a percentage. """

    
def TextGoogleQuerySimAnalyzer(corpuslistofstrings, kw):
# Setting up corpus
    # gen_docs: a list of tokens
    gen_docs = [[w.lower() for w in word_tokenize(text)] 
                for text in corpuslistofstrings]
    # dictionary: maps every word to a number
    dictionary = gensim.corpora.Dictionary(gen_docs)
    # creating a corpus
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    # creating a tf-idf model from the corpus
    tf_idf = gensim.models.TfidfModel(corpus)
    sims = gensim.similarities.Similarity('C:', tf_idf[corpus],
                                      num_features=len(dictionary))
# Changing kw into Query of Queries String
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
    
    def List2String(querylist):
        longstring = ''
        for i in querylist:
            longstring = longstring + ' ' + i
        return longstring
    
    querystring = List2String(GoogleQueryIterator(kw))
# Comparing querystring to corpus
    query_doc = [w.lower() for w in word_tokenize(querystring)]
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    # simil: comparing similarity between query and each corpus text
    simil = sims[query_doc_tf_idf]
    # expvals: single metric to compare query to the entire corpus
    expval = (sum(simil) * (1/ len(simil))) * 100.0 # in % similarity to corpus
    return '{} is {}% similar to corpus'.format(kw, expval)

