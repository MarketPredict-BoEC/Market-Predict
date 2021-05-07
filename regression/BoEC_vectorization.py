# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 08:48:22 2020

@author: Novin
"""
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from BERT_vectorization import getEmbedding

def clustering(output_dict , NUM_CLUSTERS ):
    X = []
    words = []
    articles = []
    
    for item in output_dict:
        if item != None:
            X .append(item['vector'] )
            words.append(item['key'])
            articles.append(item['articleID'])
           
    
    #fit the model
    
    kmeans_model = KMeans(NUM_CLUSTERS, init='k-means++', max_iter=100)  
    Z = kmeans_model.fit(X)
    labels = kmeans_model.labels_.tolist()
    clusterWords = {'word':words,'cluster':labels , 'vector': X,'article_ID':articles}
    
    # write to file
    clusterWordsData = pd.DataFrame(clusterWords)
    filename = 'topicInfo.xlsx'
    clusterWordsData.to_excel(filename)
    return (clusterWordsData)
 

def readVector(text):
    vec = []
    text = text.replace('[','')
    text = text.split(' ')
    for v in text:
        if v!= '':
            vec.append(float(v))
    return vec

def readEmbedding(text, articleID):
    output = {'key':'','vector' : [],'articleID':''}
    
    if len(text) >1:
        text = text.split('\t')
        output['key'] = text[0].replace('\n','')
        output['articleID'] = articleID
        output['vector'] = readVector( text[1])
    return(output)

    
def get_embeddings( dataframe):
    dataframe = getEmbedding(dataframe)
    embeddings = dataframe['embedding']
    
    for text in embeddings:
            text = text.replace('[CLS]','#CLS#')
            text = text.replace('[SEP]','#SEP#')
            lines = text.split(']')
            for line in lines:
                output = readEmbedding(line , articleID)
                if  output['key']!='':
                     embeddings.append( output) 
    return (embeddings)
            
     
def vectorization(name,data , NUM_CLUSTERS):
    output = {'articleID':0 , 'words':[],'vector':[]}
    output['articleID'] = name
    vector = np.zeros(NUM_CLUSTERS)
    output['words'] =[w for w in  data['word']]
    for item in data['cluster']:
           vector[item]  =   vector[item] + 1 
    output['vector'] = vector
    return output
    

def getAppendTopsimilar(group,topK):
    data = group[1]
    index = np.argsort(data['simScore'])
    clusters = [data['simScore'].iloc[v] for v in index]
    return clusters


def vectorize(data,topK,clusterNum):
    
    groups = data.groupby('targetKey')
    vec = np.zeros(clusterNum)
    for group in groups.iterrows():
               exAppendVec = getAppendTopsimilar(group,topK)
               for i in exAppendVec:
                   vec[i] = vec[i]+1
    return vec
            
def titleExpansion(topK ,path):
    
    expandedDataframe = pd.DataFrame(columns = ['articleID','expand Vec'])
    for r,d, f in os.walk(path):
          for file in f:
            if file.endswith('.xlsx'):
                print(file)
                df = pd.read_excel(file)
                expVec = vectorize(df , topK)
                articleId = file.replace('extended','')
                articleId = articleId.replace('.xlsx','')
                row = {'articleID': articleId , 'expand Vec':expVec}
                expandedDataframe.append(row)
    
    return   expandedDataframe
                
def BoEC_bert(dataframe, NUM_CLUSTERS = 210 , topN = 7):
    data_output =[]
    vectors = []
    outputData =  get_embeddings(dataframe)
    
    #latent concept modeling
    clusterd_data = clustering( outputData , NUM_CLUSTERS )
    
    #document vectorization process
    groups = clusterd_data.groupby(['article_ID'])
    for name , group in groups:
        data_output.append(vectorization(name,group,NUM_CLUSTERS))
    df = pd.DataFrame(data = data_output, columns=['articleID','words','vector'])
    
    #title expansion process
    expandedDF = titleExpansion(topN ,path)
    # add vector for both title, content and title expansion
    for rowindex, row in df.iterrows():
            vec1 = expandedDF.get_value(index = rowindex , col ='expand Vec')
            vec2 = row['vector']
            vectors.append(np.add(vec1,vec2))
    df['extended vector'] = vectors        
    df.to_excel('BERT content Vectorization.xlsx')    
    return df

            
            
    

        
    

        