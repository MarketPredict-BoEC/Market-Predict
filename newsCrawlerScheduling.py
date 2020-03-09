# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 08:17:14 2020

@author: Novin
"""
import schedule
import time
import FxstreetScraper
import BitcoinnewsAPI
import BitcoinnewsScraper
import ForexNewsapi

import sys
sys.setrecursionlimit(1000)



def stopScheduling():
   
   schedule.cancel_job(FxstreetScraper.FxstreetScraper) 
   schedule.cancel_job(BitcoinnewsScraper.BitcoinNewsScrapper)  
   schedule.cancel_job(BitcoinnewsAPI.BitcoinNewsApi)  
   schedule.cancel_job(ForexNewsapi.ForexNewsApi)  


def exportNews():
     FxstreetScraper.exportForexCSV()
     ForexNewsapi.exportForexCSV()
     BitcoinnewsAPI.exportBitcoinCSV()
     BitcoinnewsScraper.exportbtcCSV()

def createCollections()   :
    FxstreetScraper.createCollection()
    BitcoinnewsAPI.createCollection()
    BitcoinnewsScraper.createCollection()
    ForexNewsapi.createCollection()
    print('+---------------------------------------------+')

def start(): 
    # load rss from web to update existing xml file 
    print('started!!')
    print('+---------------------------------------------+')
    schedule.clear()
    schedule.every(60).minutes.do(FxstreetScraper.FxstreetScraper)
    schedule.every(60).minutes.do(BitcoinnewsScraper.BitcoinNewsScrapper)
    schedule.every().days.do(BitcoinnewsAPI.BitcoinNewsApi)
    schedule.every().days.do(ForexNewsapi.ForexNewsApi)
    while True:
        schedule.run_pending()
        time.sleep(1)
        
      
def main():


    user_input = input('Give me one number:\n1 for create collections\n2 for start scraping\n3 for export csv files\n4 for exit : ')
    
    if user_input == '1':                   
        createCollections()  
    if user_input == '2' :
        start()
    if user_input == '3' :
        exportNews()
    else:
        print('Exit')            
if __name__ == "__main__":
  
    # calling mpai2n function 
    main()