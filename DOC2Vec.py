# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 06:58:39 2020

@author: Novin
"""
import pandas as pd
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import utils
from sklearn import svm
import csv
from tqdm import tqdm
import multiprocessing
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
import time

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

tqdm.pandas(desc="progress-bar")

# Initializing the variables

df = pd.read_csv('CF output - Copy.csv')
tags_index = {'down': 1 , 'up': 2}

y = df['label60']
data = []

for rowIndex,row in df.iterrows():
    x = tokenize_text(str(row['title']) + str(row['articleBody']))
    
    print(tags_index[row['label60']])
    data.append(TaggedDocument(words = x , tags = [tags_index[row['label60']]]))
    

train_documents, test_documents, y_train, y_test = train_test_split(data, y, test_size=0.2)



# Initializing the variables




# doc2vec prepare
cores = multiprocessing.cpu_count()

model_dbow = Doc2Vec(dm=1, vector_size=210, negative=5, hs=0, min_count=3, sample = 0, workers=cores, alpha=0.025, min_alpha=0.001)

model_dbow.build_vocab([x for x in tqdm(train_documents)])
                        
#train_documents  = utils.shuffle(train_documents)
model_dbow.train(train_documents,total_examples=len(train_documents), epochs=30)    

def vector_for_learning(model, input_docs):
    sents = input_docs
    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, feature_vectors
model_dbow.save('./forexModel.d2v')


y_train, X_train = vector_for_learning(model_dbow, train_documents)
y_test, X_test = vector_for_learning(model_dbow, test_documents)


clf = svm.SVC(random_state=42 , gamma = 'auto')
print("Default Parameters are: \n",clf.get_params)

start_time = time.time()
clf.fit(X_train, y_train)
fittime = time.time() - start_time
print("Time consumed to fit model: ",time.strftime("%H:%M:%S", time.gmtime(fittime)))
start_time = time.time()
score=clf.score(X_test, y_test)
print("Accuracy: ",score)

y_pred = clf.predict(X_test)

scoretime = time.time() - start_time
print("Time consumed to score: ",time.strftime("%H:%M:%S", time.gmtime(scoretime)))
case1=[score,fittime,scoretime]
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


