# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 00:47:07 2020

@author: Novin
"""

from BERT_BoEC import BERT_BOEC_train



def trainModels():
	pairs = ['EURUSD','USDJPY','GBPUSD','BTCUSDT']
	for pair in pairs:
		BERT_BOEC_train(pair, topN =7,max_L = 15,SEQ_LEN_news = 7,SEQ_LEN = 7,embedding_dim = 210)

def main():
	trainModels()


if __name__ == "__main__":
    #calling main function
    main()
