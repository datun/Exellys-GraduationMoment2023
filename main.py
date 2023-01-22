#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 15:34:42 2023

@author: deniz
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# Sentiment Analysis Libraries (Roberta)
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax
from transformers import pipeline

import time


# Variables for ROBERTA

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)



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
dirpath = ""

# Variables for appending sentiments

df_sentiment = pd.DataFrame(columns=['negative','neutral','positive'])

# Sentiment analysis for datasets that do not have it
# sentiment_pipeline = pipeline(model='cardiffnlp/twitter-roberta-base-sentiment-latest')

# For file loading from directory
for dirname, _, filenames in os.walk(os.path.join(mainDir, dataSets[-1])):
    for filename in filenames:
        if filename.endswith('.csv'):
            loadingFilePath.append(os.path.join(dirname, filename))
        dirpath = dirname
            
for i in range(len(loadingFilePath)):
    readFiles[i] = pd.read_csv(loadingFilePath[i], sep=',')

print(readFiles.keys())

df_tweets = readFiles.get(0)
df_stocks = readFiles.get(1)

start = time.time()

for i in range(len(df_tweets)):
    encoded_input = tokenizer(df_tweets.loc[i,'Tweet'], return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    temp_list = [[scores[0], scores[1], scores[2]]]
    df_sentiment = df_sentiment.append(pd.DataFrame(temp_list, columns=list(df_sentiment)), ignore_index=True)
    
    end = time.time()
    duration = end - start
    
    if duration >= 600:
        df_temp = df_tweets.iloc[:i]
        df_temp = df_temp.join(df_sentiment)
        temp_name = 'tmp_Tweets_with_sentiments.csv'
        df_temp.to_csv(os.path.join(dirname, temp_name), encoding='utf-8')
        print("Temp file is created at " + str(time.ctime()))
        start = time.time()

    
concatenated = df_tweets.join(df_sentiment)
concatenated.to_csv(os.path.join(dirname, "Tweets_with_sentiments.csv"), encoding='utf-8')
