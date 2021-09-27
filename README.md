# MarketPredict tool
**MarketPredict** is an implementation of the **BERT based Bag-of-Economic-Concepts 
(BERT-BoEC)** for the Financial Market _(FOREX and cryptocurrencies)_ trend and price prediction.

The current implementation of our model answers two problems of trend (up/down binary classification) and price prediction as well as our news scraping tool. In the source code, there are two folders for 1) classification and 2) regression, both of which use the same file for news vectorization called BoEC_vectorization. The first step for news document vectorization is computing concept clusters through the chain of function calls (Main => BERT_BOEC_train => BoEC_bert(newsDF) => clustering(outputData, NUM_CLUSTERS)). In the BoEC_vectorization file, at the first step, we use k_means clustering of BERT word embedding for constructing latent economic concepts (Please refer to the line 124 and also 13-36 at the BoEC_vectorization file in both folders. The research community can benefit our tool to train BERT-BoEC predictive model and also to scrape financial news releases especially for FOREX and Cryptocurrency markets

For the implementation that uses BERT-BOEC for trend prediction,
please refer to folder _BERT_BoEC/classification_  in MarketPredict. 

For the implementation that uses BERT-BOEC for price prediction,
please refer to folder _BERT_BoEC/regression_  in MarketPredict. 

For the implementation of MarketPredict news scraping tool, please refer to folder _BERT_BoEC/crawler_.

DataSets DOI in figshare: 10.6084/m9.figshare.11977908

[DataSets link in figshare](https://figshare.com/s/7257c70ba9e726093026)
