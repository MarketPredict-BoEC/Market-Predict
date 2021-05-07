**crawler implementation**
News providing services.
Currently available for three categories of *Forex* , *Cryptocurrency* and *Commodity* and all currency pairs in these categories.

For example: ERRUSD, USDJPY, GBPUSD, BTCUSDT


After scraping, we statndard all scraped news based on following attributes. All news items store in **one** MongoDB collection. For each item in our news collection, we have following attributes:

| Attribute | Description | Example | Null |
|-----------|-------------|--------|--------|
|   title   | News Title. |    Canada: Will impose tariffs on imports of certain US aluminum products by September 6|No|
|   articleBody   | News body | USD stronger ahead of the weekly close, although the movement seems corrective.   |Yes|
|   pubDate| News release time based UTC timestamp |  1615513000 |No|
|   author   | News author | Valeria Bednarik   | No |
|   keywords  | News keywords. | Sentiment,EURUSD,Employment,Recommended,Coronavirus,   |No| 
   category      | News category     | `Forex`, `Cryptocurrency`, `Commodities`|No|
|   provider      | Newsgroup    | FXStreet, newsBTC, Reuters, Cointelegraph, Investing, Bloomberg|
|   summary      | news summary     | A breif summary about news|Yes|
|   link      | News link     |https://www.fxstreet.com/news/eur-usd-turkey-risks-could-trigger-overdue-correction-lower-for-the-euro-mufg-202008071657 |No|
|   thImage      | URL of news thumbnail image     | URL of news thumbnail image | Yes
|    images | URLs of images in news body    |  Array of URLs of news images  |No
The code crawled news documents from different news sources and store them in MongoDB collections.
The newsgroups are :
	
	www.fxstreet.com ( Forex market related news)
	www.newsbtc.com ( cryptocurrency related news)
	Google news API for Routers and Bloomberg sources, but in limitted version.
	

For running this code, please first run 'main.py' and follow the steps in output Ipython console. you must first create the collections and then start crawling. 
The crawler will be connected to the news sources pages and scheduled in 60 minutes or day for different sources.
