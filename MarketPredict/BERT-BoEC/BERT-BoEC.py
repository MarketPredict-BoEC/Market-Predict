#######################################
#BERT-BoEC market prediction module
#implemented on Keras in Tensorflow 2 
#with Keras Functional API
#######################################


import pandas as pd
from matplotlib import pyplot
import tensorflow as tf
import pandas as pd
from collections import deque
import random
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM,  BatchNormalization,concatenate
from tensorflow.keras.layers import  Input, Activation, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from tensorflow import keras
from matplotlib import pyplot as plt
import  BoEC_vectorization

"""

NewsPath = path of news dataset
Path = BERT embeddings
concepts number for Latent Concepts modelling
top most similar words for news title expansion

"""
def prepair_data(conceptNumbers = 210,topN = 7, NewsPath,MarketPath,Path ):

		#BERT-BoEC vectorization
		df1 = BoEC_vectorization.BoEC_bert(NewsPath, Path,conceptNumbers ,topN )

		#market data read from excel file
		marketDF = pd.read_excel(MarketPath)



		# market data labelling based on close price change 

		FUTURE_PERIOD_PREDICT = 1 
		def classify(current, future):
			if float(future) > float(current):  # if the future price is higher than the current, that's a buy, or a 1
				return 1
			else:  # otherwise... it's a 0!
				return 0

		marketDF.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
		marketDF.dropna(inplace=True)

		marketDF['future'] = marketDF['close'].shift(-FUTURE_PERIOD_PREDICT)
		marketDF['target'] = list(map(classify, marketDF['close'], marketDF['future']))
		marketDF = marketDF.drop('future',1)





		marketDF['timestamp'] = marketDF['date']+'T'+marketDF['time']
		marketDF['timestamp'] = pd.to_datetime(marketDF['timestamp'])
		marketDF = marketDF.set_index('timestamp')


		# market data normalization


		def normalization(df):
			df = df.drop("date", 1)
			df = df.drop("time", 1)
			df = df.drop("high", 1)  # don't need this anymore.
			df = df.drop("low", 1)  # don't need this anymore.
			df = df.drop("open", 1)  # don't need this anymore.
			
			for col in df.columns:  # go through all of the columns
				if col != "target":  # normalize all ... except for the target itself!
					df[col] =[float(e) for e in df[col]]
					df[col] = df[col].pct_change()  # pct changefor  "normalizes"
					df = df.replace([np.inf, -np.inf], None)
					df.dropna(inplace=True )  # remove the nas created by pct_change
					df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.
			
			df.dropna(inplace=True)  # cleanup again... jic.
			return df

		# SEQ_LEN is set to delay window 



		SEQ_LEN = 7
		def preprocess_df(df):
			
			df = normalization(df)
			sequential_data = []  # this is a list that will CONTAIN the sequences
			
			# These will be our actual sequences. 
			#they are made with deque, which keeps
			#the maximum length by popping out older 
			#values as new ones come in
			prev_days = deque(maxlen=SEQ_LEN)  

			for i in df.values:  # iterate over the values
			
				prev_days.append([n for n in i[:-1]])  # store all but the target
				if len(prev_days) == SEQ_LEN:  # make sure we have 10 sequences!
					sequential_data.append([np.array(prev_days), i[-1]])  # i[-1] is the sequence target

			
			#random.shuffle(sequential_data)  # shuffle for good measure.
			

			X = []
			y = []

			for seq, target in sequential_data:  # going over our new sequential data
				X.append(seq)  # X is the sequences
				y.append(target)  # y is the targets/labels (buys vs sell/notbuy)


			
			return np.array(X), y  # return X and y...and make X a numpy array!


		# plot market data 

		import matplotlib.pyplot as plt   # plotting

		get_ipython().run_line_magic('matplotlib', 'inline')
		marketDF.plot(subplots=True,
				layout=(6, 3),
				figsize=(22,22),
				fontsize=10, 
				linewidth=2,
				sharex=False,
				title='Visualization of the original Time Series')
		plt.show()


		# train and test split for market data

		dates = sorted(marketDF.index.values)  # get the dates
		last_5pct = sorted(marketDF.index.values)[-int(0.3*len(dates))]  # get the last 20% of the times

		test_main_df = marketDF[(marketDF.index >= last_5pct)]  # make the validation data where the index is in the last 5%
		main_df = marketDF[(marketDF.index < last_5pct)]  # now the main_df is all the data up to the last 5%

		dates = sorted(main_df.index.values)  # get the dates
		last_5pct = sorted(main_df.index.values)[-int(0.20*len(dates))]  # get the last 20% of the times
		validation_main_df =  main_df[(main_df.index > last_5pct)]  
		main_df = main_df[(main_df.index < last_5pct)] 

		print(main_df.shape)
		print(validation_main_df.shape)
		print(test_main_df.shape)

		print(main_df.head())
		print(main_df.describe())




		from datetime import datetime
		df= pd.read_excel('outputlabeled news.xlsx')
		df['timestamp'] = pd.to_datetime(df['date'])
		df['date'] = [w.date() for w in df['timestamp']]
		df['time'] = [w.time() for w in df['timestamp']]
		df['hour'] =   [datetime.strftime(w,'%H') for w in  df['timestamp']]
		df = df.reset_index()
		var  =[]
		for i in range(len(df)):
			var.append(df.loc[i,'date'].strftime('%Y-%m-%d') +'T'+str(df.loc[i,'hour'])+':00:00')

		df['timestamp'] = pd.to_datetime(var)

		print(df.head())


		# In[132]:



		traindata_start = main_df.index.values[0] 
		traindata_end = main_df.index.values[-1] 

		validationdata_start = validation_main_df.index.values[0] 
		validationdata_end =validation_main_df.index.values[-1] 

		testdata_start = test_main_df.index.values[0]
		testdata_end = test_main_df.index.values[-1]

		print(traindata_start)
		print(traindata_end)
		print(validationdata_start)
		print(validationdata_end)





		train_x, train_y = preprocess_df(main_df)
		train_y = keras.utils.to_categorical(train_y, 2)

		validation_x, validation_y = preprocess_df(validation_main_df)
		validation_y = keras.utils.to_categorical(validation_y, 2)


		print(train_x.shape)
		print(validation_x.shape)



		# set L with max number of news per hour
		max_L = 5
		# concept cluster number
		embedding_dim = 210
		#df = df.set_index('timestamp')
		def modify(s):
		   
			vec = s.replace('[','')
			vec = vec.replace(']','')
			vec = vec.split(',')
			vec = map(float , vec)
			vec = list(vec)
			return vec
		# this part of code, we align news with market date based on timestamp of news release!!!
		# in this part of code, we fixed total number of news per day in 5 because the max number of news per hour was 5

		aligned_news_data = []
		empty_vector = np.zeros((max_L,embedding_dim))
		for var in marketDF.index.values:
			if ( var not in news_date ):
				aligned_news_data.append(empty_vector)
			else:
				vector = np.zeros((max_L,embedding_dim))
				
				if(type( df.loc[var,'vector']))==type('test'):
					
					vector[0] =np.asarray( modify(df.loc[var,'vector']))
					i = i+1
				else:
					i = 0;
					for nItem in df.loc[var,'vector']:
						vector[i] =np.asarray( modify(nItem))
						i = i+ 1
				aligned_news_data .append( vector)
		aligned_news_data = np.asarray(aligned_news_data)




		print(len(aligned_news_data))
		print(len(marketDF))
		print(aligned_news_data.shape)


		# aligned train, validation and test set of news data with corresponding values in market data

		train_news_x = aligned_news_data[SEQ_LEN-1:train_x.shape[0]+SEQ_LEN-1]
		validation_news_x = aligned_news_data[train_x.shape[0]+SEQ_LEN-1:train_x.shape[0]+validation_x.shape[0]+SEQ_LEN-1]

		print(train_news_x.shape)
		print(validation_news_x.shape)
		test_x, test_y = preprocess_df(test_main_df)
		test_y = keras.utils.to_categorical(test_y, 2)
		test_news_x = aligned_news_data[train_x.shape[0]+validation_x.shape[0]+SEQ_LEN-1 :
		return(train_x,train_news_x,train_y,validation_x, validation_news_x, validation_y , test_x,test_news_x , test_y)

# build the model 

def build_model(dim):


	# trade data RNN

	# #7 delay window

	# define our RNN network for technical indicator feature extraction
	marketModel = Sequential()
	marketModel.add(LSTM(128, input_shape=(train_x.shape[1:])))
	marketModel.add(Dropout(0.2))
	marketModel.add(Activation("relu"))





	# news data recurrent convolution network
	inputShape = (train_news_x .shape[1],train_news_x .shape[2])
	inputs = Input(shape=inputShape)
	x = Conv1D(filters=64, kernel_size=3, activation='relu',padding='same', input_shape = inputShape)(inputs)
	x = MaxPooling1D(pool_size=2)(x)
	x = Dropout(0.2)(x)
	x = Activation("relu")(x)
	x = LSTM(128)(x)
	x = Flatten()(x)
	BoEC_RCNN = Model(inputs, x)



	# concatenate news and market output 

	combinedInput = concatenate([marketModel.output, BoEC_RCNN.output])
	x = Dense(2, activation="softmax")(combinedInput)

	model = Model(inputs=[marketModel.input, BoEC_RCNN.input], outputs=x)
	return model


def fit_model(model,epochs=60, batch_size=32):

	

	model.compile(loss="categorical_crossentropy", 
				  optimizer='Adam',
				   metrics=['accuracy'])
	model.summary()



	# train the model
	callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
	history = model.fit(
		[train_x, train_news_x], train_y,
		validation_data=([validation_x, validation_news_x], validation_y),
		epochs=60, batch_size=32, callbacks=[callback])


	model.save('BERT-BoEC.h5') 

def Build_BERsT_BoEc( 
	train_x,train_news_x,
	train_y,validation_x, 
	validation_news_x,
	validation_y):
	model = build_model(dim = train_x.shape[1] )
	fit_model(model,epochs=60, batch_size=32):
	return
	
def predict(test_x, test_news_x , test_y ):
	
    train_x.shape[0]+validation_x.shape[0]+test_x.shape[0]++SEQ_LEN-1]
	loss, acc = model.evaluate([test_x,test_news_x], test_y)
	print("Test accuracy = ", acc)
	return
	