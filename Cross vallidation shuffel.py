# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 20:58:55 2020

@author: Novin
"""

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import   cross_val_score , StratifiedKFold
import time
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import math 
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

random_state = np.random.RandomState(0)
def modify(s):
    vec = s.replace('[','')
    vec = vec.replace(']','')
    vec = vec.split(',')
    vec = map(float , vec)
    vec = list(vec)
    mylist = [0 if math.isnan(x) else x for x in vec]

    return mylist
f = open ('cecf 10.txt','a')
f.write('accuracy\tprecision\trecall\tf1_macro\tf1_micro\n')
filename = 'output extended 3 similar.xlsx'


values = [0,0,0,0,0]
dfDocument = pd.read_excel(filename)
index = 'label60'
y =  dfDocument[index]
data = [modify(w) for w in dfDocument['expand vector']]
data , y = shuffle(data,y)
cv = StratifiedKFold(n_splits=5,shuffle=True)
j=0
random_state = np.random.RandomState(0)
#clf = RandomForestClassifier(random_state=random_state)
clf = SVC(random_state=random_state)
#clf =  xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.5 ,random_state=random_state , probability = True)

#clf = SVC(random_state=random_state)
for score in ["accuracy", "precision", "recall" ,'f1_macro' , 'f1_micro']:
     
    values[j] = cross_val_score(clf, data, y, scoring=score, cv=cv ).mean()
    j = j+1
print(values)

        
        
