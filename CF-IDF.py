# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 01:07:21 2019

@author: Novin
"""

import pandas as pd
import re
import string
import numpy as np
import nltk
import numpy as np
import math
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import unicodedata


    
def modify(s):
    vec = s.replace('[','')
    vec = vec.replace(']','')
    vec = vec.split(',')
    vec = map(float , vec)
    vec = list(vec)
    mylist = [0 if math.isnan(x) else x for x in vec]

    return mylist
def IDF(dfDocuments,clusterNumber):
    
    clusterFreq = np.zeros(clusterNumber)
    docNumber = dfDocuments['title'].count()
    for rowIndex,row in dfDocuments.iterrows():
        vec  = modify(row['expand vector'])
        for index  in range(clusterNumber):
            count =  vec[index]
            clusterFreq[index] = clusterFreq[index] + count
    clusterFreq1 = [math.log(docNumber / w ,10) for w in clusterFreq ]
    clusterFreq2 = [(docNumber / w ) for w in clusterFreq ]
    return [clusterFreq1,clusterFreq2,clusterFreq]

#++++++++++++++++++++++++++++++++++++ variable declaration ++++++++++++++++#

newsInfoFilename = 'Ecf idf.xlsx'
clusterNumber = 210


#+++++++++++++++ function +++++++++++++++++++++#


dfDocuments = pd.read_excel(newsInfoFilename)


df = pd.Series(np.zeros(dfDocuments['title'].count()))

clusterFreq1,clusterFreq2,clusterFreq = IDF(dfDocuments , clusterNumber)
print (clusterFreq)
i = 0 
for rowIndex,row in dfDocuments.iterrows():
    vec = modify(row['expand vector'])
    idf_vec = np.multiply(vec , clusterFreq1 )
    df[rowIndex] =  str([w for w in idf_vec ])    
    i = i+1
    print(i)


dfDocuments['idf expand vector'] = df
outputFileName = 'Ecf idf.xlsx'
          
dfDocuments.to_excel(outputFileName)      
#[7.0, 7.0, 2.0, 16.0, 5.0, 1.0, 7.0, 1.0, 9.0, 10.0]
#[6.0, 17.0, 4.0, 6.0, 1.0, 1.0, 5.0, 0.0, 2.0, 0.0]  

    
    
