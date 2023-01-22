#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 20:40:59 2023

@author: deniz
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


import time




# Main Variables Used

mainDir = '/home/deniz/Downloads/DataChallenge'
dataSets = ['BigTechCompanies',
            'CryptoCurrencyTweets',
            'FinancialTweets',
            'FinancialTweets-Cleaned',
            'StockMarketTweets 04.2020 - 07.2020',
            'StockMarketTweets with StockMarket Data 09.2021 - 09.2022']

loadingFilePath = []
readFiles = {}

dirpath = os.path.join(mainDir, dataSets[-1])

graph_path = os.path.join(dirpath, 'graphs')

# For file loading from directory
for dirname, _, filenames in os.walk(os.path.join(mainDir, dataSets[-1])):
    for filename in filenames:
        if filename.endswith('.csv'):
            if 'sentiments' in filename:
                loadingFilePath.append(os.path.join(dirname, filename))
            if 'yfinance' in filename:
                loadingFilePath.append(os.path.join(dirname, filename))


start = time.time()

for i in range(len(loadingFilePath)):
    readFiles[i] = pd.read_csv(loadingFilePath[i], sep=',')

print(readFiles.keys())

df_tweets = readFiles.get(1)
df_stocks = readFiles.get(0)

stock_name_list = df_stocks['Stock Name'].unique()

for XXXX in stock_name_list:
    # STOCK RELATED VARS
    date_stock = df_stocks.loc[df_stocks['Stock Name'] == XXXX, ['Date']]
    date_stock['Date'] = pd.to_datetime(date_stock['Date'])
    date_close = df_stocks.loc[df_stocks['Stock Name'] == XXXX, ['Close']]
    
    # TWEET RELATED VARS
    stock_tweets = df_tweets.loc[
        df_tweets['Stock Name'] == XXXX, ['Date', 
                                            'negative',
                                            'neutral',
                                            'positive']]
    stock_tweets['Date'] = pd.to_datetime(stock_tweets['Date'])
    mean_sentiment = stock_tweets.groupby(pd.Grouper(key='Date', freq='D')).mean()
    
    fig, ax = plt.subplots(2, figsize=(15,10))
    fig.suptitle('Stock and Sentiment Graph for: ' + str(XXXX))
    ax[1].plot(mean_sentiment.index, mean_sentiment['positive'], label='positive', zorder=5)
    ax[1].plot(mean_sentiment.index, mean_sentiment['negative'], label='negative', zorder=10)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
    ax[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax[1].tick_params(axis='x', rotation=30)
    ax[1].set_title( str(XXXX) +' Sentiment Scores in Softmax')
    ax[1].set(ylabel='Sentiment Scores')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)
    ax[1].plot()
    
    
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
    ax[0].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax[0].set_xticklabels([])
    ax[0].set_title( str(XXXX) + ' Closing Prices')
    ax[0].set(ylabel='Price')
    ax[0].plot(date_stock, date_close)
    ax[0].grid(True)
    plt.show()
    fig.savefig(os.path.join(graph_path,(str(XXXX)+'.png')))



end = time.time()
print('Started at:' + str(start) + '// completed at: ' + str(end))