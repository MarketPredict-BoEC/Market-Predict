# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 05:39:07 2020

@author: Novin
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import   cross_val_score
from sklearn.svm import SVC
import seaborn as sns
import time 
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report


df = pd.read_csv('myForexNews.csv')


#y = df['label60']
data = []
for rowIndex,row in df.iterrows():
    data.append(str(row['title']) + str(row['articleBody']))
    
    



#X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)




# ##############################################Count vectorization#######

count_vect = CountVectorizer()
data_counts = count_vect.fit_transform(data)


# ##############################################TFIDF#####################
tfidf_transformer = TfidfTransformer()
data_tfidf = tfidf_transformer.fit_transform(data_counts)
print(data_tfidf.shape)
# #################### model training######################################
'''
X_train, X_test, y_train, y_test = train_test_split(data_tfidf, y, test_size=0.2)

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
'''