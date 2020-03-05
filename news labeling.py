# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 02:38:09 2019



@author: Novin
"""

import pandas as pd
import datetime
import math

#+++++++++++++++++++++++++ variable declaration++++++++++++++++++#
timeFrameInMinutes = 60 
mode = 'close'
format_str = '%Y-%m-%dT%H:%M:%SZ'
marketInfoDelayInMinutes= 5


#+++++++++++++++++++++++++ getClose ++++++++++++++++++#
def getClose(startDate,endDate,marketInfo,dateInfo):
    i=0
    
    while( dateInfo[i] < startDate) :
        i= i+1
    closePrev =float( marketInfo['close'][i])
    closeNext =float( marketInfo['close'][i+ (timeFrameInMinutes/marketInfoDelayInMinutes)])
    
    return(closeNext-closePrev)
#+++++++++++++++++++++++++ getLable++++++++++++++++++#
def getLable(timestamp , timeFrame , marketInfo,dateInfo ):
    startTime = timestamp - datetime.timedelta( minutes = timeFrame)
    endTime = timestamp + datetime.timedelta( minutes = timeFrame)
    temp = getClose(startTime,endTime,marketInfo,dateInfo)
    state = 0;
    if temp > 0:
        state = 1
    if temp < 0:
        state = -1
    return state



#+++++++++++++++++++++++++ getClose ++++++++++++++++++#
def getClose2(startDate,endDate,marketInfo,dateInfo):
    i=0
    
    while( dateInfo[i] < startDate) :
        i= i+1
    closePrev =float( marketInfo['close'][i])
    closeNext =float( marketInfo['close'][i+ (timeFrameInMinutes/marketInfoDelayInMinutes)])
    
    return(math.log(closeNext/closePrev))
#+++++++++++++++++++++++++ getLable++++++++++++++++++#
def getLable2(timestamp , timeFrame , marketInfo,dateInfo ):
    startTime = timestamp - datetime.timedelta( minutes = timeFrame)
    endTime = timestamp + datetime.timedelta( minutes = timeFrame)
    temp = getClose2(startTime,endTime,marketInfo,dateInfo)
    state = 0;
    if temp > 0:
        state = 1
    if temp < 0:
        state = -1
    return state



#+++++++++++++++++++++++++ read Data from files++++++++++++++++++#

df = pd.read_csv('EURUSDnews.csv')
marketInfo = pd.read_csv('EURUSD5-2.csv')
format_str1 = '%Y.%m.%d %H:%M:%S'


marketInfo['datetime'] = marketInfo['date']+' '+marketInfo['time']
d = [pd.to_datetime(w) for w in marketInfo['date']]
marketInfo['datetime'] = d

    
#++++++++++++++++++++++++ assign Labels +++++++++++++++++++++++++#

for i in range(len(df)):
    day = df['date'][i]
    datetime_obj = datetime.datetime.strptime(day , format_str)
    df['label'][i] = getLable(datetime_obj, timeFrameInMinutes, marketInfo,d)
    df['label2'][i] = getLable2(datetime_obj, timeFrameInMinutes, marketInfo,d)


#++++++++++++++++++++++++ write In files +++++++++++++++++++++++++#
df.to_excel('EURUSDlabeledNews 60 minutes1.xlsx')
