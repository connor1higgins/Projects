# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:16:59 2018

@author: Connor Higgins
"""
# List of tags
#  'p' : #defines a paragraph
#  'param', name="movie", value=True : #defines a parameter for an object (movie),
#  'h1, h2, h3, h4': HTML headings (biggest to smallest)
#  'th' : header cell in a table
#  'td' : general cell in a table
#  'center' : defines center text (Not used in HTML5, older)
#  'section' : defines a section in a document
#  'a, href=True : defines hyperlinks
#  'i' : italicized text
#  'b' : bold text
#  'strong' : generally emphasized
#  'li' : defines a list item
#  'span' : defines a section in a document (similar to section)
#  'meta', name="keywords" : metadata keywords
#  'img' : image

## imports ##
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import urllib.request
import re

def WebsiteParser(url):
    
## initialization ##    
    # soup: url parsed
    #soup = BeautifulSoup(requests.get(url).text, 'html5lib') # switch if 403 forbidden
    #html_doc = urllib.request.urlopen(url).read() ## 2nd option switch if 403 forbidden
    #soup = BeautifulSoup(html_doc, 'html.parser') ## 2nd option switch if 403 forbidden
    
    
    # If still 403 forbidden (3rd option)
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)' \
                 'Chrome/64.0.3282.167 Safari/537.36'
    
    req = urllib.request.Request(url, headers={'User-Agent': user_agent})
    resp = urllib.request.urlopen(req)
    html_doc = resp.read()
    soup = BeautifulSoup(html_doc, 'html.parser')
    
    
## tag parsers ##    
    # title of webpage
    webtitle = soup.title.get_text()
    
    # list of all 1st headers
    firstheaders = []
    for a in soup.find_all('h1'): # iterates over a list of strings with h1 tags
        #resub: removes all extra spaces and tabs in the string
        resub = re.sub('[ \t]+', ' ', a.get_text()).strip()
        firstheaders.append(resub) # appends string to list of 1st headers
        
    # list of all 2st headers    
    secondheaders = []
    for a in soup.find_all('h2'):
        resub = re.sub('[ \t]+', ' ', a.get_text()).strip()
        secondheaders.append(resub)
    
    # list of all 3rd headers    
    thirdheaders = []
    for a in soup.find_all('h3'):
        resub = re.sub('[ \t]+', ' ', a.get_text()).strip()
        thirdheaders.append(resub)
    
    # list of all 4th headers    
    fourthheaders = []
    for a in soup.find_all('h4'):
        resub = re.sub('[ \t]+', ' ', a.get_text()).strip()
        fourthheaders.append(resub)
    
    # list of metadata keywords
    metakeywords = []
    for a in soup.head.find_all('meta', content=True):
        if 'keywords' in str(a):
            resub = re.sub('[ \t]+', ' ', a['content']).strip()
            metakeywords.append(resub)

    # list of all italicized text
    italics = []
    for a in soup.find_all('i'):
        resub = re.sub('[ \t]+', ' ', a.get_text()).strip()
        italics.append(resub)
    
    # list of all bolded text
    bolds = []
    for a in soup.find_all('b'):
        resub = re.sub('[ \t]+', ' ', a.get_text()).strip()
        bolds.append(resub)

    # list of all emphasized text
    emphasized = []
    for a in soup.find_all('strong'):
        resub = re.sub('[ \t]+', ' ', a.get_text()).strip()
        emphasized.append(resub)
    
    # list of all paragraphs
    p = []
    for a in soup.find_all('p'):
        resub = re.sub('[ \t]+', ' ', a.get_text()).strip()
        p.append(resub)
    
    # list of all center defined text    
    center = []
    for a in soup.find_all('center'):
        resub = re.sub('[ \t]+', ' ', a.get_text()).strip()
        center.append(resub)

    # list of all document sections
    sections = []
    for a in soup.find_all('section'):
        resub = re.sub('[ \t]+', ' ', a.get_text()).strip()
        sections.append(resub)
    
    # list of spans (sections)
    spans = []
    for a in soup.find_all('span'):
        resub = re.sub('[ \t]+', ' ', a.get_text()).strip()
        spans.append(resub)
    
    # list of all header cells
    headercells = []
    for a in soup.find_all('th'):
        resub = re.sub('[ \t]+', ' ', a.get_text()).strip()
        headercells.append(resub)
    
    # list of all cells
    allcells = []
    for a in soup.find_all('td'):
        resub = re.sub('[ \t]+', ' ', a.get_text()).strip()
        allcells.append(resub)
    
    # list of all listed items
    listitems = []
    for a in soup.find_all('li'):
        resub = re.sub('[ \t]+', ' ', a.get_text()).strip()
        listitems.append(resub)
    
    # list of team_desc
    team_desc = []
    for a in soup.find_all('div', class_="team_desc"):
        resub = re.sub('[ \t]+', ' ', a.get_text()).strip()
        team_desc.append(resub)

    # list of all hyperlinks
    hyperlinks = []
    for a in soup.find_all('a', href=True):
        if a['href'][0] != '#':
            resub = re.sub('[ \t]+', ' ', a['href']).strip()
            hyperlinks.append(resub)
        
    # tuple list of all image alts and sources
    imagealts = []
    imagesrcs = []
    for a in soup.find_all('img', alt=True, src=True):
        resub = re.sub('[ \t]+', ' ', a['alt']).strip()
        imagealts.append(resub)
        resub = re.sub('[ \t]+', ' ', a['src']).strip()
        imagesrcs.append(resub)
        
    # list of all video sources    
    videos = []
    for a in soup.find_all('param'):
        if a['name'] == 'movie':
            resub = re.sub('[ \t]+', ' ', a['value']).strip()
            videos.append(resub)
            
## dataframe creation ##
    # dictionary of lists for dataframe data  
    dictoflists = {'Site Title': [webtitle],
                   '1st Headers': firstheaders,
                   '2nd Headers': secondheaders,
                   '3rd Headers': thirdheaders,
                   '4th Headers': fourthheaders,
                   'Meta Keywords': metakeywords,
                   'Italics' : italics,
                   'Bolds' : bolds,
                   'Emphasized' : emphasized,
                   'Paragraphs' : p,
                   'Centered Text' : center,
                   'Section Text' : sections,
                   'Span Text' : spans,
                   'Table Header Text' : headercells,
                   'Table Cell Text' : allcells,
                   'Itemized Text' : listitems,
                   'Description' : team_desc,
                   'Hyperlinks' : hyperlinks,
                   'Image Names' : imagealts,
                   'Image Links' : imagesrcs,
                   'Video Links' : videos}
    # df: dataframe containg all info from dictoflists, first oriented toward 
    # index to concatenate arrays of different lengths, then transposed. 
    df = (pd.DataFrame.from_dict(dictoflists, orient='index')).transpose()
    
    # replaces empty cells with nan
    df.replace('', np.nan, inplace=True)
    
    # replaces none entities with nan
    df.fillna(value=np.nan, inplace=True)
    
    # returns dataframe
    return(df)