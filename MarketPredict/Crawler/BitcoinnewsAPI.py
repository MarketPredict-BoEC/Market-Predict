# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:09:25 2019

@author: Novin
"""

from newsapi.newsapi_client import NewsApiClient
import pymongo
import pandas as pd
from datetime import datetime,timedelta


def getCryptoNews(startDate,endDate):
    try:
        newsapi = NewsApiClient(api_key='ef9f89cce9b24cfe9ed9b61f900cc1b1')
        news =[]
        bitcoin_articles = newsapi.get_everything(q='bitcoin',
                                              sources='crypto-coins-news,bloomberg,reuters,google-news',
                                              domains='cnn,bloomberg,reuters,google',
                                              from_param = startDate,
                                              to = endDate,
                                              language='en')
        BTC_articles = newsapi.get_everything(q='btc',
                                              sources='crypto-coins-news,bloomberg,reuters,google-news',
                                              domains='cnn,bloomberg,reuters,google',
                                              from_param=startDate,
                                              to=endDate,
                                              language='en')
        crypto_articles = newsapi.get_everything(q='cryptocurrency',
                                      sources='crypto-coins-news,bloomberg,reuters,google-news',
                                      domains='cnn,bloomberg,reuters,google',
                                      from_param=startDate,
                                      to=endDate,
                                      language='en')
        blockchain_articles = newsapi.get_everything(q='blockchain',
                                      sources='crypto-coins-news,bloomberg,reuters,google-news',
                                      domains='cnn,bloomberg,reuters,google',
                                      from_param=startDate,
                                      to=endDate,
                                      language='en')





        for item in  bitcoin_articles['articles']:
            news.append(item)
            
        for item in  BTC_articles['articles']:
            news.append(item)
        
        for item in  crypto_articles['articles']:
            news.append(item)
        for item in  blockchain_articles['articles']:
            news.append(item)
    
    except :
        print('Error in Reading BTC News!')
    
    return (news)

def createCollection():
    try:
            
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient["EconomicNewsDataBase"]
        mycol = mydb["BitcoinNews"] 
        item = {'key1':'value1'}
        mycol.insert_one(item)
        print('BitcoinNews created succsesfully!')
     
    except:
        print('error in collection creation!!')
        
def saveInMongo(newsItem):
    try:
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient["EconomicNewsDataBase"]
        mycol = mydb["BitcoinNews"] 
        item = {'key1':'value1'}
        mycol.insert_one(item)
        mycol.delete_one(item)
       
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

def exportBitcoinCSV():
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["EconomicNewsDataBase"]

    cursor = mydb["BitcoinNews"].find()

    # Expand the cursor and construct the DataFrame
    df =  pd.DataFrame(list(cursor))
    df.to_csv('BitcoinNews.csv')

def BitcoinNewsApi():
    
    endDate = datetime.today()
    startDate = datetime.today() - timedelta(days=29)
    #load RSS File From Url
    print('Crawling of Bitcoin News from API Started!!')
    print('+---------------------------------------------+')
    newsitems = getCryptoNews(startDate,endDate) 
    # store news items in a csv file 
    saveInMongo(newsitems)



    

    
    
