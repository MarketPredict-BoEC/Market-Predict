# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 00:48:56 2019

@author: Novin
"""

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import re
import string
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
    pattern3 = r'br&gt|b&gt|blockquote&gt|.*&amp#.*|.*&amp'
    corpus = re.sub(imgPattern,"", corpus , flags=re.MULTILINE)
    corpus = re.sub(srcPattern,"", corpus , flags=re.MULTILINE)
    corpus = re.sub(urlPattern1,"", corpus , flags=re.MULTILINE)
    corpus = re.sub(urlPattern2,"", corpus , flags=re.MULTILINE)
    corpus = re.sub(numPattern1,"", corpus , flags=re.MULTILINE)
    corpus = re.sub(numPattern2,"", corpus , flags=re.MULTILINE)
    corpus = re.sub(pattern1,"", corpus , flags=re.MULTILINE)
    corpus = re.sub(pattern2,"S&P", corpus , flags=re.MULTILINE)
    corpus = re.sub(pattern3,"", corpus , flags=re.MULTILINE)
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
    text = denoise_text(text)
    words = text.split(' ')    
    words = remove_non_ascii(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    words = lemmatize_verbs(words)
    test =' '.join(words)
    return test


fileName = open('EURUSDnewsCourpus.txt','r')
corpus = fileName.readlines()
text = ''
words = []
for line in corpus:
    words.append(normalize(line))
    text = ' '.join(words)
filename2 = open('Preprocess EURUSDnewsCourpus.txt','w')
filename2.write(text)

fileName.close()
filename2.close()