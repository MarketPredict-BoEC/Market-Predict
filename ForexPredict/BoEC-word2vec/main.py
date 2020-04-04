# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 00:06:31 2020

@author: Novin
"""
import BoEC_WORD2VEC
import newsLabeling
import crossValidation_Shuffel

corpusPath ='EURUSD news.xlsx'
outputFileName ='labeled news.xlxs'
marketInfoPath = 'EURUSD5.xlsx'
conceptNumbers = 210
topK =7
embeddingDim = 100 
windowSize = 3 
deltaT = 60

def main():
    # latent concept modeling and document vectorization
    dfDocuments = BoEC_WORD2VEC.BoEC_word2vec (corpusPath , 
                                 outputFileName,
                                 conceptNumbers ,
                                 topK , 
                                 embeddingDim  ,
                                 windowSize )
    #news labeling based on market data
    dfDouments = newsLabeling.news_labeling (marketInfoPath , dfDocuments , deltaT)
    #train the model with SVM and 5Fold stratified K fold
    crossValidation_Shuffel.classifier(dfDocument['vector'] ,dfDocument['label'])
    return

if __name__ == "__main__":
    #calling main function
    main()

