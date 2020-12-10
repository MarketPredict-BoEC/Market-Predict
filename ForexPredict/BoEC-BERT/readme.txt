For using BERT-BoEC you must at first run the code available in the file 'wordembeddingBERT.ipynb' in your TPU device. 
This implementation done for hourly EUR/USD trend prediction.
Then set some variables in main.py file:

newsInfoPath : The path of news Info dataSet.
Path :  The directory that contains the contextualized embedding vectors for new documents. 
marketInfoPath : The path of Market Info dataSet
conceptNumbers : Number of concept clusters
topN : Number of top N most similar words for title expansion


Then you must put the News and Market dataset in following path and run the main.py.
