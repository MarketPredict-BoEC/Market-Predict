# -*- coding: utf-8 -\*-
"""
Created on Sat Jan 18 11:25:59 2020

@author: Novin
"""
import xml.etree.ElementTree as ET 
from bs4 import BeautifulSoup 
import pymongo
import pandas as pd
import requests


def loadPage( url ,fileName = None): 
    # url of rss feed 
  try:  
    # creating HTTP response object from given url 
        resp = requests.get(url,timeout = 3) 
        fail ='fail'
        if resp.status_code == 200:
             # saving the xml file 
             with open(fileName, 'wb') as f: 
                 f.write(resp.content) 
                 return 1
        else:
             with open(fileName, 'wb') as f: 
                 f.write(fail) 
                 return -1
             
        resp.close()
        resp.raise_for_status()
   
  except requests.exceptions.HTTPError as htError:
        print('Http Error: ',htError)
        
  except requests.exceptions.ConnectionError as coError:
        print('Connection Error: ',coError)
  except requests.exceptions.Timeout as timeOutError:
        print('TimeOut Error: ',timeOutError)
  except requests.exceptions.RequestException as ReError:
        print('Something was wrong: ',ReError)
  return(-1)



def parseXML(xmlfile): 
    tree = ET.parse(xmlfile) 
        # get root element 
    root = tree.getroot() 
        # create empty list for news items 
    newsitems = [] 
        # iterate news items 
    for item in root:
        for item in item.findall('item'):
            news = {}
            category = {}
            for child in item:
                   
                   if child.tag == '{http://purl.org/rss/1.0/modules/content/}encoded':
                       news['encoded'] =child.text
                   elif child.tag == '{http://wellformedweb.org/CommentAPI/}commentRss':
                       news['commentRss'] =child.text
                   elif child.tag == '{http://purl.org/rss/1.0/modules/slash/}comments':
                       news['comments'] =child.text
                   elif child.tag == '{com-wordpress:feed-additions:1}post-id':
                       news['post-id'] =child.text
                   elif child.tag == '{http://purl.org/dc/elements/1.1/}creator':
                       news['creator'] = child.text
                   elif child.tag == 'category':
                       category ['item{}'.format(len(category)+1)] = child.text
                       
                   else:
                       news[child.tag] = child.text
                   news['category'] = category    
            newsitems.append(news)
    return newsitems 

def createCollection():
    try:
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient["EconomicNewsDataBase"]
        mycol = mydb["BitcoinNews2"] 
        item = {'key1':'value1'}
        mycol.insert_one(item)
        print('BitcoinNews2 created succsesfully!')
    except:
        print('error in collection creation!!')

def chechforexist(querry):
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient["EconomicNewsDataBase"]
        mycol = mydb["BitcoinNews2"]   
        mydoc = mycol.find(querry)
        exist = len(list(mydoc))
        return exist
    
def saveInMongo(newsItem):
    try:
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient["EconomicNewsDataBase"]
        mycol = mydb["BitcoinNews2"]   
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

def exportbtcCSV():
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["EconomicNewsDataBase"]

    cursor = mydb["BitcoinNews2"].find()

    # Expand the cursor and construct the DataFrame
    df =  pd.DataFrame(list(cursor))
    df.to_csv('bitcoinNews2.csv')  

        
def BitcoinNewsScrapper():
    url = 'https://www.newsbtc.com/feed/'
    filename = 'topBTCnewsfeed.xml'
    #load RSS File From Url
    print('Crawling of bitcoin news Started!!')
    print('+---------------------------------------------+')
    code = loadPage(url,filename) 
    if code == 1:
        #parse xmlzz file 
        newsitems = parseXML(filename) 
        # store news items in a csv file 
        saveInMongo(newsitems)

