"""
Created on Wed Feb 19 20:58:55 2020

@author: Novin
"""

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import   cross_val_score , StratifiedKFold
import time
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def modify(s):
    vec = s.replace('[','')
    vec = vec.replace(']','')
    vec = vec.split(',')
    vec = map(float , vec)
    vec = list(vec)
    return vec
def classifier(data ,label):
    
    random_state = np.random.RandomState(0)
    cv = StratifiedKFold(n_splits=5,shuffle=True)
    data , y = shuffle(data,label)
    j=0
    values = [0,0,0,0,0]
    for score in ["accuracy", "precision", "recall" ,'f1_macro' , 'f1_micro']:
        values[j] = cross_val_score(SVC(random_state=random_state), data, y, scoring=score, cv=cv ).mean()
        j = j+1
    print(values)
            
    