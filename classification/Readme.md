# MarketPredict tool
**MarketPredict** is an implementation of the **BERT based Bag-of-Economic-Concepts (BERT-BoEC)** for the Financial Market _(FOREX and cryptocurrencies)_ trend prediction.

Download dataset from following link and put files in Data folder.
This implementation train model to predict _up/down_ trend. 

DataSets DOI in figshare: 10.6084/m9.figshare.11977908
[DataSets link in figshare](https://figshare.com/s/7257c70ba9e726093026)

For training the model, use pip to install dependencies. In our implementation, we extract BERT contextualized word embedding through functions in _BERT_vectorization.py_ file. Then we extract concepts clusters space through functions available in _BoEC_vectorization.py_ and also news vectorization is done in this file.  For news and market data alignment, we use functions available in _BERT_BoEC.py_ file. We also implement the model architecture in this file. 

After installing the dependencies, put the dataset in the Data folder and run the _main.py_ file. You can find four files with _"*.h5"_ extension corresponding to each currency pairs predictive model for trend prediction. 
