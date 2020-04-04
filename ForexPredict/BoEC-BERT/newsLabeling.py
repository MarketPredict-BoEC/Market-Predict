"""
Created on Sat Aug 17 02:38:09 2019



@author: Novin
"""

import pandas as pd
import datetime
import math

#+++++++++++++++++++++++++ variable declaration++++++++++++++++++#

mode = 'close'
format_str = '%Y-%m-%dT%H:%M:%SZ'
marketInfoDelayInMinutes= 5




#+++++++++++++++++++++++++ Close price ++++++++++++++++++#
def getClose(date,marketInfo):
    i = 0
    while(marketInfo['dateTime'][i] <= date):
        i = i+1
    return(marketInfo['close'][i])
    
#+++++++++++++++++++++++++ getLable++++++++++++++++++#
def getLable(timestamp , deltaT1, deltaT2 , marketInfo):
    startTime = timestamp - datetime.timedelta( minutes = deltaT1)
    endTime = timestamp + datetime.timedelta( minutes = deltaT2)
    closePrev = getClose(startTime,marketInfo)
    closeNext = getClose(endTime,marketInfo)
    temp = (closeNext - closePrev)
    if temp>0:
        return 1
    
    return -1



#+++++++++++++++++++++++++ read Data from files++++++++++++++++++#
def news_labeling (marketInfoPath , newsInfoPath , deltaT):
        

    dfNews = pd.read_excel(newsInfoPath)
    dfMarket = pd.read_excel(marketInfoPath)
    format_str1 = '%Y.%m.%d %H:%M:%S'

   
    #++++++++++++++++++++++++ assign Labels +++++++++++++++++++++++++#
    d=[]
    for item in dfMarket.iterrows():
        s =pd.to_datetime(item[1].date)
        s = s + datetime.timedelta(hours = item[1].time.hour ,minutes = item[1].time.minute , seconds = item[1].time.second )
        d.append(s)
           
    dfMarket['dateTime'] =  d  
    labels = []      
    deltaT1 = deltaT2 = deltaT
    for i in range(len(dfNews)):
        timestamp = dfNews['date'][i]
        timestamp = datetime.datetime.strptime(timestamp , format_str)
        labels.append( getLable(timestamp, deltaT1 , deltaT2, dfMarket))
            
        
    #++++++++++++++++++++++++ write In files +++++++++++++++++++++++++#
    dfNews['label'] = labels
    dfNews.to_excel('labeled news.xlsx')
    return dfNews
