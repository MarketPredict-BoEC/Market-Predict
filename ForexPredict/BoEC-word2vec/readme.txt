For using BoEC-Word2vec you must set some variable in main.py file:


corpusPath : The path of news dataset
outputFileName : after creation of embedded document space, the corresponding vectors saved in this file
marketInfoPath : The path of Market Info dataSet
conceptNumbers : Number of concept clusters
topN : Number of top N most similar words for title expansion
embeddingDim : Embedding dimension in word2vec training process 
windowSize : Contextual window size
deltaT : time interval for news labeling based on it

Then you must put the News and Market dataset in following path and run the main.py.The model got the evaluation metrics.
