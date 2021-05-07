
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:06:39 2020

@author: Novin
"""

#Python code to illustrate parsing of XML files 
# importing the required modules 
 
import requests 
import json
import xml.etree.ElementTree as ET 
from bs4 import BeautifulSoup 
import pymongo
import pandas as pd
from datetime import datetime

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
             with open(fileName, 'a') as f: 
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

def getArticleBody(url , filename):
    loadPage(url,filename)
    f1 = open('nonScrapedLink.txt','a')
    try:
        
        description={}
        f = open(filename)
        content = f.read()
        description = {}
        if content != 'fail':
            
            soup = BeautifulSoup(content,'html.parser')
            json_output= BeautifulSoup(str(soup.find_all("script",id = {"SeoApplicationJsonId"})), 'lxml')
            jsonText = json_output.get_text()
            jsonData = json.loads(jsonText , strict = False)
            for child in jsonData:
                 description['articleBody'] = child['articleBody']
                 description['keywords'] = child['keywords']
                 description['author'] = child['author']['name']
        else:
            f1.write(url)
            f1.write('\n')
            f1.close()
                        
    except  json.JSONDecodeError as err:
        print('read article body error: ',err)
    return(description)
    
    
def parseXML(xmlfile): 
    # create element tree object 
    tree = ET.parse(xmlfile) 
    # get root element 
    root = tree.getroot() 
    # create empty list for news items 
    newsitems = [] 
    # iterate news items 
    for item in root:
        
        for item in item.findall('item'):
            news={}
            for child in item:
               if child.tag == '{http://www.fxstreet.com/syndicate/rss/namespaces/}pair':
                   news['pair'] =child.text
               elif child.tag == '{http://www.fxstreet.com/syndicate/rss/namespaces/}market':
                   news['market'] =child.text
               elif child.tag == '{http://www.fxstreet.com/syndicate/rss/namespaces/}TechAnalysis':
                   for c in child:
                       if c.tag == '{http://www.fxstreet.com/syndicate/rss/namespaces/}TrendIndex':
                           news['TrendIndex'] =c.attrib
                       elif c.tag == '{http://www.fxstreet.com/syndicate/rss/namespaces/}OBOSIndex':
                           news['OBOSIndex'] =c.attrib
                       elif c.tag == '{http://www.fxstreet.com/syndicate/rss/namespaces/}PivotPoints':
                           news['PivotPoints'] =c.attrib
                       
                   news['TechAnalysis'] =child.text
               elif child.tag == '{http://www.fxstreet.com/syndicate/rss/namespaces/}headline':
                   news['headline'] =child.text
               elif child.tag == '{http://www.fxstreet.com/syndicate/rss/namespaces/}summary':
                   news['summary'] =child.text
                   
               elif child.tag == '{http://www.fxstreet.com/syndicate/rss/namespaces/}provider':
                   news['provider'] =child.text
               else:
                   news[child.tag] = child.text
            querry = {'link' : news['link']}
            exist = checkForExist(querry)
            if not exist: 
                         
                desc =  getArticleBody(news['link'],'articlebody.html')
                for c in desc :
                       news[c] = desc[c]
                saveInMongo(news)
            newsitems.append(news) 
      
    # return news items list 
    return newsitems 



def createCollection():
    try:
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient["EconomicNewsDataBase"]
        mycol = mydb["myForexNews"] 
        item = {'key1':'value1'}
        mycol.insert_one(item)
        print('ForexNews created sucsessfullys!')
     
    except:
        print('error in collection creation!!')


def checkForExist(querry):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["EconomicNewsDataBase"]
    mycol = mydb["myForexNews"] 
    mydoc = mycol.find(querry)
    exist = len(list(mydoc))
    return exist
    
      
def saveInMongo(item):
    try:
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient["EconomicNewsDataBase"]
        mycol = mydb["myForexNews"]   
        querry = {'link':str(item['link'])}
        exist = checkForExist(querry)
        if not exist :
                 x = mycol.insert_one(item)
                 print (x)
        

    except pymongo.errors.ConnectionFailure as err :
        print('DataBase ConnectionFailuure Error:',err)        
    except pymongo.errors.DocumentTooLarge as err :
        print('DataBase DocumentTooLarge Error:',err)        
    except pymongo.errors.BSONError as err :
        print('DataBase BSONError Error:',err)        
    except:
        print('DataBase Error:')        
    


def exportForexCSV():
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["EconomicNewsDataBase"]

    cursor = mydb["myForexNews"].find()

    # Expand the cursor and construct the DataFrame
    df =  pd.DataFrame(list(cursor))
    df.to_csv('ForexNews.csv')

def FxstreetScraper():
    f = open ('Forexlog.txt','a')
    url = 'http://xml.fxstreet.com/news/forex-news/index.xml'
    filename = 'topnewsfeed.xml'
    #load RSS File From Url
    print('crawling of fxstreet Started!!')
    print('+---------------------------------------------+')
    code = loadPage(url,filename) 
    if code == 1:
         parseXML(filename) 
         print('+---------------------------------------------+')
        # store news items in a csv file 
    else:
        f.write('Connection Error at time : '+ datetime.now().strftime('%y %m %d %H %M %S') + '\n')
        f.close()
    
