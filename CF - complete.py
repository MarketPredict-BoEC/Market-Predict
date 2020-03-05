# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 06:03:49 2020

@author: Novin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 06:23:50 2019

@author: Novin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 08:33:28 2018

@author: Novin
"""
import pandas as pd
import re
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import unicodedata



def cleanText(corpus):
    imgPattern = r'img.*'
    srcPattern = r'src.*'
    urlPattern1 = r'http[s]\s?:.*?\s'
    urlPattern2 = r'www.*?\s'
    numPattern1 = r'\d*\.\d+'
    numPattern2 = r'\d+'
    lenPattern1 = r'[a-z]{12,20}'
    pattern1 = r'(%)'
    pattern2 = r'S&pampP|S&ampampP'
    #pattern3 = r'br&gt|b&gt|blockquote&gt|.*&amp#.*|.*&amp'
    corpus = re.sub(imgPattern,"", corpus , flags=re.MULTILINE)
    corpus = re.sub(srcPattern,"", corpus , flags=re.MULTILINE)
    corpus = re.sub(urlPattern1,"", corpus , flags=re.MULTILINE)
    corpus = re.sub(urlPattern2,"", corpus , flags=re.MULTILINE)
    corpus = re.sub(numPattern1,"", corpus , flags=re.MULTILINE)
    corpus = re.sub(numPattern2,"", corpus , flags=re.MULTILINE)
    corpus = re.sub(pattern1,"", corpus , flags=re.MULTILINE)
    corpus = re.sub(pattern2,"S&P", corpus , flags=re.MULTILINE)
   # corpus = re.sub(pattern3,"&", corpus , flags=re.MULTILINE)
    corpus = re.sub('/', '', corpus)
    corpus = re.sub(r'[0-9]+', '', corpus)
    corpus = re.sub(lenPattern1,"", corpus , flags=re.MULTILINE)
    
    return corpus


def denoise_text(text):
    text = cleanText(text)
    return text

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(text):
    text = cleanText(text)
    words = text.split(' ')    
    words = remove_non_ascii(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    words = lemmatize_verbs(words)
    return words

def doc2vec(doc,length,clusters):
    if doc.loc['articleBody'] :
         text = doc['title'] + doc.loc['articleBody']
    else :
         text = doc['title']
    
    vec = pd.Series(np.zeros(length))
    words = normalize(text)
    for word  in words:
            vec [clusters[clusters['word'] == word]['cluster']] = vec [clusters[clusters['word'] == word]['cluster']] +1
      
    return vec
    
    

#++++++++++++++++++++++++++++++++++++ variable declaration ++++++++++++++++#


clusternumbers = [i  for i in range(10,310,10)]

clusterInfoFileNames = ['topicInfo'+str(i)+'.xlsx' for i in clusternumbers]
newsInfoFilename = 'EURUSDlabeledNews 60 minutes1.xlsx'


#+++++++++++++++ function +++++++++++++++++++++#
i=0
for item in clusterInfoFileNames:
    
    dfClusters = pd.read_excel(item)
    dfDocuments = pd.read_excel(newsInfoFilename)
    
    
    df = pd.Series(np.zeros(dfDocuments['title'].count()))
    
    for rowIndex,row in dfDocuments.iterrows():
          vec = doc2vec(row,clusternumbers[i],dfClusters )
          df[rowIndex] =  str([w for w in vec])
    dfDocuments['vector'] = df
          
    outputFileName = 'output'+item
          
    dfDocuments.to_excel(outputFileName)      
    i = i+1
        
    
    
        
    
