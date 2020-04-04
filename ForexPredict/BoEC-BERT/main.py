# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 00:47:07 2020

@author: Novin
"""

import BoEC_BERT
import newsLabeling
import crossValidation_Shuffel

# please set this variable to the folder of contextualized embedding Rep. for news Documents 
Path = '//embeddings'
marketInfoPath = 'EURUSD5.xlsx'
conceptNumbers = 210
topN =7
deltaT = 60

def main():
    # latent concept modeling and document vectorization
    dfDocuments = BoEC_BERT.BoEC_bert(Path , conceptNumbers ,topN )
    #news labeling based on market data
    dfDocuments = newsLabeling.news_labeling (marketInfoPath , dfDocuments , deltaT)
    #train the model with SVM and 5Fold stratified K fold
    crossValidation_Shuffel.classifier(dfDocuments['extended vector'] ,dfDocuments['label'])
    return

if __name__ == "__main__":
    #calling main function
    main()
