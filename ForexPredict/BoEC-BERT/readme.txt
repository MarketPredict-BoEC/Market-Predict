For using BoEC-BERT you must at first run the code available in the file 'wordembeddingBERT.ipynb' in your TPU device. 

Then set some variable in main.py file:

newsInfoPath : The path of news Info dataSet.
Path :  The directory that contains the contextualized embedding vectors for new documents. 
marketInfoPath : The path of Market Info dataSet
conceptNumbers : Number of concept clusters
topN : Number of top N most similar words for title expansion
deltaT : time interval for news labeling based on it

Then you must put the News and Market dataset in following path and run the main.py.The model got the evaluation metrics.
