from tensorflow.keras.models import load_model
import pandas as pd
from matplotlib import pyplot
import tensorflow as tf
import pandas as pd
from collections import deque
import random,pathlib
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, LSTM, BatchNormalization,concatenate
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation,Embedding, Conv1D, MaxPooling1D, Flatten,GlobalMaxPooling1D
from matplotlib import pyplot as plt
import ta
from tensorflow import keras
from matplotlib import pyplot as plt
from datetime import timedelta
from BoEC_vectorization import BoEC_bert



def classify(current, future):
    if float(future) > float(current):  # if the future price is higher than the current, that's a buy, or a 1
        return 1
    else:  # otherwise... it's a 0!
        return 0


def normalization(df):
    df = df.drop("Date", 1)
    # df = df.drop("time", 1)
    df = df.drop("High", 1)  # don't need this anymore.
    df = df.drop("Low", 1)  # don't need this anymore.
    df = df.drop("Open", 1)  # don't need this anymore.

    for col in df.columns:  # go through all of the columns
        if col != "target":  # normalize all ... except for the target itself!
            df[col] = [float(e) for e in df[col]]
            df[col] = df[col].pct_change()  # pct changefor  "normalizes"
            df = df.replace([np.inf, -np.inf], None)
            df.dropna(inplace=True)  # remove the nas created by pct_change
            df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

    df.dropna(inplace=True)  # cleanup again... jic.
    return df


def modify(s):
    vec = s.replace('[', '')
    vec = vec.replace(']', '')
    vec = vec.split(',')
    vec = map(float, vec)
    vec = list(vec)
    return vec


def getNews_embedding(currentDate, df):
    news_date = df.index.values
    prevDate = currentDate - timedelta(hours=SEQ_LEN_news)
    subDF = df.loc[prevDate:currentDate]
    if len(subDF) == 0:
        return (np.zeros((max_L, embedding_dim)))

    else:
        vector = np.zeros((max_L, embedding_dim))
        i = 0
        for d, row in subDF[:max_L].iterrows():
            vector[i] = np.asarray(modify(row['vector']))
            i = i + 1
        return vector


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Loss [Close]')
    plt.legend()
    plt.grid(True)


def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()


def preprocess_df(df, newsDF):
    df = normalization(df)
    aligned_news_data = []
    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)
    for d, row in df.iterrows():

        prev_days.append([n for n in row[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 10 sequences!
            sequential_data.append(
                [np.array(prev_days), row[-1], getNews_embedding(d, newsDF), d])  # i[-1] is the sequence target

    X = []
    y = []
    newsX = []
    dates = []

    for seq, target, newsEmbedding, d in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)
        newsX.append(newsEmbedding)
        dates.append(d)

    return np.array(X), np.array(y), np.array(newsX), dates  # return X and y...and make X a numpy array!


def prepairDataForTraining(marketDF, newsDF, max_L = 15,SEQ_LEN_news = 7,SEQ_LEN = 7,embedding_dim = 210):
    FUTURE_PERIOD_PREDICT = 1


    marketDF.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
    marketDF.dropna(inplace=True)

    marketDF['target'] = marketDF['Close'].shift(-FUTURE_PERIOD_PREDICT)
    marketDF['timestamp'] = marketDF['Date']  # +'T'+marketDF['time']
    marketDF['timestamp'] = pd.to_datetime(marketDF['timestamp'])
    marketDF = marketDF.set_index('timestamp')


    var = []
    for i in range(len(df)):
        d = pd.to_datetime(df.loc[i, 'date'])
        h = str(df.loc[i, 'time'])
        var.append(str(d.strftime('%Y-%m-%d') + 'T' + h))

    df['timestamp'] = pd.to_datetime(var)
    df.sort_values(by=['timestamp'], inplace=True)
    print(df.head())

    newsDF = newsDF.set_index('timestamp')

    train_x, train_y, train_news_x, train_dates = preprocess_df(main_df, newsDF)
    validation_x, validation_y, validation_news_x, validation_dates = preprocess_df(validation_main_df, newsDF)


    print(train_news_x.shape)
    print(validation_news_x.shape)
    return train_x, train_news_x, train_y, validation_x, validation_news_x, validation_y


def trainModel(train_x,train_news_x,train_y,
               validation_x, validation_news_x, validation_y,pair):
    # trade data RNN

    dim = train_x.shape[1]  # 7 delay window

    # define our RNN network for technical indicator feature extraction
    marketModel = Sequential()
    marketModel.add(LSTM(128, input_shape=(train_x.shape[1:])))
    marketModel.add(Dropout(0.2))

    from tensorflow.keras.regularizers import l2

    # news data recurrent convolution network
    inputShape = (train_news_x.shape[1], train_news_x.shape[2])
    inputs = Input(shape=inputShape)
    x = Conv1D(filters=64, kernel_size=3,
               activation='relu', padding='same',
               input_shape=inputShape, kernel_regularizer=l2(15e-3))(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = LSTM(128)(x)
    BoEC_RCNN = Model(inputs, x)

    # concatenate news and market output

    combinedInput = concatenate([marketModel.output, BoEC_RCNN.output])
    x = Dense(1)(combinedInput)

    model = Model(inputs=[marketModel.input, BoEC_RCNN.input], outputs=x)

    opt = tf.keras.optimizers.Adam(lr=0.1, decay=1e-3)
    model.compile(
        loss=tf.keras.losses.MeanAbsolutePercentageError(),
        optimizer=opt

    )

    model.summary()

    # In[19]:

    # train the model
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    history = model.fit(
        [train_x, train_news_x], train_y,
        validation_data=([validation_x, validation_news_x], validation_y),
        epochs=180, batch_size=32,callback=callback)

    model.save(pair+"WithNewsHourly.h5")
    plot_loss(history)
    plt.show()

    plot_accuracy(history)

    print(model.evaluate([test_x, test_news_x], test_y))
    plt.show()

def BERT_BOEC_train(pair, NUM_CLUSTERS = 210 , topN = 7,max_L = 15,SEQ_LEN_news = 7,SEQ_LEN = 7,embedding_dim = 210):
    newsData= pair+"news.Xlsx"
    marketData = pair+"Market.Xlsx"
    current_Path = pathlib.Path().absolute()
    newsPath = str(current_Path) + '/Data/' + newsData
    marketPath = str(current_Path) + '/Data/' + marketData
    newsDF = pd.read_excel(newsPath)
    newsDF = BoEC_bert(newsDF,)
    marketDF = pd.read_excel(marketPath)
    train_x, train_news_x, train_y,validation_x, validation_news_x, validation_y =  prepairDataForTraining(marketDF,
                                                                                                           newsDF,
                                                                                                           max_L = 15,
                                                                                                           SEQ_LEN_news = 7,
                                                                                                           SEQ_LEN = 7,
                                                                                                           embedding_dim = 210)
    trainModel(train_x, train_news_x, train_y,
               validation_x, validation_news_x, validation_y, pair)










