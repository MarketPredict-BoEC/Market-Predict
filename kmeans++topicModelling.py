# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 01:55:21 2019

@author: Novin
"""

import gensim
import pandas as pd
from sklearn.cluster import KMeans
model = gensim.models.Word2Vec.load('ForexNews.embedding')
vectors = model.wv
X = model[model.wv.vocab]

for i in range(210,310,10):
    
    NUM_CLUSTERS=i
    kmeans_model = KMeans(NUM_CLUSTERS, init='k-means++', max_iter=100)  
    Z = kmeans_model.fit(X)
    labels=kmeans_model.labels_.tolist()
    l = kmeans_model.fit_predict(X)
    
    words = list(model.wv.vocab)
    clusterWords = {'word':words,'cluster':labels}
    clusterWordsData = pd.DataFrame(clusterWords)
    filename = 'topicInfo'+ str(i)+'.xlsx'
    clusterWordsData.to_excel(filename)
