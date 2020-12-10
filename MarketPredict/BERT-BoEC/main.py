# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 00:47:07 2020

@author: Novin
"""

import BERT_BoEC


# please set this variable to the folder of contextualized embedding Rep. for news Documents 
Path = '//embeddings'
MarketPath = 'EURUSD.xlsx'
NewsPath = 'EURUSDNews.xlsx'
conceptNumbers = 210
topN =7
deltaT = 60



def main():
    # latent concept modelling and document vectorization
	train_x,train_news_x,
	train_y,validation_x, 
	validation_news_x,
	validation_y,test_x,
	test_news_x , test_y = BERT_BoEC.prepair_data(conceptNumbers = 210,topN = 7, NewsPath,MarketPath,Path)
    
	BERT_BoEC. Build_BERsT_BoEc( 
	train_x,train_news_x,
	train_y,validation_x, 
	validation_news_x,
	validation_y)
	
	BERT_BoEC. predict(test_x, test_news_x , test_y )
    
    return

if __name__ == "__main__":
    #calling main function
    main()
