# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:09:25 2019

@author: Novin
"""

from newsapi.newsapi_client import NewsApiClient
import pymongo
import pandas as pd
from datetime import datetime,timedelta


def getForexNews(startDate,endDate):
    try:
        newsapi = NewsApiClient(api_key='ef9f89cce9b24cfe9ed9b61f900cc1b1')
        news =[]
        eurusd_articles = newsapi.get_everything(q='eurusd',
                                              sources='crypto-coins-news,bloomberg,reuters,google-news',
                                              domains='cnn,bloomberg,reuters,google',
                                              from_param=startDate,
                                              to=endDate,
                                              language='en')
        usdjpy_articles = newsapi.get_everything(q='usdjpy',
                                      sources='crypto-coins-news,bloomberg,reuters,google-news',
                                      domains='cnn,bloomberg,reuters,google',
                                      from_param=startDate,
                                      to=endDate,
                                      language='en')
      
        forex_articles = newsapi.get_everything(q='forex',
                                              sources='crypto-coins-news,bloomberg,reuters,google-news',
                                              domains='cnn,bloomberg,reuters,google',
                                              from_param = startDate,
                                              to = endDate,
                                              language='en')
         
        oil_articles = newsapi.get_everything(q='oil',
                                      sources='crypto-coins-news,bloomberg,reuters,google-news',
                                      domains='cnn,bloomberg,reuters,google',
                                      from_param=startDate,
                                      to=endDate,
                                      language='en')
        gold_articles = newsapi.get_everything(q='gold',
                                      sources='crypto-coins-news,bloomberg,reuters,google-news',
                                      domains='cnn,bloomberg,reuters,google',
                                      from_param=startDate,
                                      to=endDate,
                                      language='en')





        for item in  forex_articles['articles']:
            item['label'] = 'Forex'
            news.append(item)
            
        for item in  eurusd_articles['articles']:
            item['label'] = 'EURUSD'
            news.append(item)
      
        for item in  usdjpy_articles['articles']:
            item['label'] = 'USDJPY'
            news.append(item)
        for item in  oil_articles['articles']:
            item['label'] = 'oil'
            news.append(item)
        for item in  gold_articles['articles']:
            item['label'] = 'gold'
            news.append(item)
    
    except :
        print('Error in Reading Forex2 News!')
    
    return (news)

def createCollection():
    try:
        
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient["EconomicNewsDataBase"]
        mycol = mydb["forexNews2"] 
        item = {'key1':'value1'}
        mycol.insert_one(item)
        print('forexNews2 created succsesfully!')
     
    except:
        print('error in collection creation!!')

def saveInMongo(newsItem):
    try:
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient["EconomicNewsDataBase"]
        mycol = mydb["forexNews2"] 
        
       
        for item in newsItem:
            querry = {'title':str(item['title'])}
            mydoc = mycol.find(querry)
            exist = len(list(mydoc))
            if not exist :
                 x = mycol.insert_one(item)
                 print (x)
        print('+---------------------------------------------+')

    except pymongo.errors.ConnectionFailure as err :
        print('DataBase ConnectionFailuure Error:',err)        
    except pymongo.errors.DocumentTooLarge as err :
        print('DataBase DocumentTooLarge Error:',err)        
    except pymongo.errors.BSONError as err :
        print('DataBase BSONError Error:',err)        
    except:
        print('DataBase Error:') 
    return

def exportForexCSV():
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["EconomicNewsDataBase"]

    cursor = mydb["forexNews2"].find()

    # Expand the cursor and construct the DataFrame
    df =  pd.DataFrame(list(cursor))
    df.to_csv('forexnews2.csv')

def ForexNewsApi():
    
    endDate = datetime.today()
    startDate = datetime.today() - timedelta(days=29)
    #load RSS File From Url
    print('Crawling of Forex News from API Started!!')
    print('+---------------------------------------------+')
    newsitems = getForexNews(startDate,endDate) 
    # store news items in a csv file 
    saveInMongo(newsitems)


    

    
    
