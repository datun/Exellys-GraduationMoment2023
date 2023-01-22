# Exellys Data Challenge 2023

This is the project repository for the python codes that I've used throughout
the Exellys Data Manipulation Challenge and Whitepaper in 2022 - 2023.

The repository is named as graduation moment, since this is part of a "Graduation
Ceremony" and a project to be presented as part of that ceremony.

## Aim

Aim of this project is to prove/disprove the existence of correlation between
tweets/social media and stock prices.

This is done by extracting sentiments from tweets by using NLP model, comparing
stock closing prices with the sentiment scores of negative or positive.

The dataset used is limited and small at the moment of writing (2023-01-22),
but the project is to be expanded until the day of presentation.


## How to Run / Files

- `main.py` loads the tweet dataset and uses NLP model to score provided rows.  
The resulting columns of scores / table of score is appended to the tweet table.  
Then it is saved in the same directory.

- `plotting.py` loads the sentiment score including dataset and stock prices
dataset. Then filters tweets with respect to stock and groups them by day.  
After that, mean of positive and negative tweets are taken to have a general
indicator of said day's sentiment. Then seniment score and stock prices are
graphed together.


## Resources Used

To be filled



 
