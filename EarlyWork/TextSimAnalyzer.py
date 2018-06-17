# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 09:18:15 2018

@author: Connor Higgins
"""

import gensim
from nltk.tokenize import word_tokenize

""" Takes a string and a list of strings and, using Latent Semantic Analysis,
    compares them, returning an array of decimal values. """

def TextSimAnalyzer(corpuslistofstrings, querystring):
    # gen_docs: a list of tokens
    gen_docs = [[w.lower() for w in word_tokenize(text)] 
                for text in corpuslistofstrings]
    # dictionary: maps every word to a number
    dictionary = gensim.corpora.Dictionary(gen_docs)
    # creating a corpus: a list of bags of words. A bag-of-words representation
    # for a document just list the number of times each word occurs in the document
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    # creating a tf-idf model from the corpus
    # tf-idf: Term Frequency-Inverse Document Frequency. Term frequency is how often
    # the word shows up in the document and inverse document frequency scales the value
    # by how rare the word is in the corpus.
    tf_idf = gensim.models.TfidfModel(corpus)
    sims = gensim.similarities.Similarity('C:', tf_idf[corpus],
                                      num_features=len(dictionary))
    # creating a query document and converting it to tf-idf
    query_doc = [w.lower() for w in word_tokenize(querystring)]
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    # simil: comparing similarity between query and each corpus text
    simil = sims[query_doc_tf_idf]
    return simil
