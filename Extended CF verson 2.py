# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 19:33:26 2020

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
import numpy as np
import gensim
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
    words = []
    expanded_words = normalize(doc['title'])
    model = gensim.models.Word2Vec.load('ForexNews.embedding')
    vec = pd.Series(np.zeros(length))
    
    for word in expanded_words:
        if word in model.wv.vocab :
           synset = model.most_similar(word, topn=3)
          
           for s in synset:
               
               vec [clusters[clusters['word'] == s[0]]['cluster']] = vec [clusters[clusters['word'] == s[0]]['cluster']] +1
      
    return vec

def modify(s):
    vec = s.replace('[','')
    vec = vec.replace(']','')
    vec = vec.split(',')
    vec = [float(x) for x in vec]
    vec = map(float , vec)
    vec = list(vec)
    return vec

#++++++++++++++++++++++++++++++++++++ variable declaration ++++++++++++++++#
clusternumbers = 210

clusterInfoFileNames = 'topicInfo210.xlsx'
newsInfoFilename = 'CF output.xlsx'



#+++++++++++++++ function +++++++++++++++++++++#

dfClusters = pd.read_excel(clusterInfoFileNames)
dfDocuments = pd.read_excel(newsInfoFilename)

df = pd.Series(np.zeros(dfDocuments['title'].count()))
i = 0
for rowIndex,row in dfDocuments.iterrows():
          expand_vec = doc2vec(row,clusternumbers,dfClusters )
          cf_vec = modify(row['vector'])
          extende_vec  =np.add(expand_vec ,cf_vec)
          print( i)
          df[rowIndex] =  str([w for w in extende_vec ])
          i = i+1
dfDocuments['expand vector'] = df
outputFileName = 'output extended 3 similar.xlsx'
          
dfDocuments.to_excel(outputFileName)      
   
    
    



    
    
