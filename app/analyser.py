from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd 
import numpy as np
import re
import plotly 
import plotly.express as px 
import json 

analyser = SentimentIntensityAnalyzer()

def get_results(df):
    pos_tweets = df[df.sentiment == 'positive']
    pos_retweet = 0
    for tweet in pos_tweets: 
        pos_retweet = pos_retweet + tweet['retweet_count']
    neut_tweets = df[df.sentiment == 'neutral']
    neut_retweet = 0
    for tweet in neut_tweets: 
        neut_retweet = neut_retweet + tweet['retweet_count']
    neg_tweets = df[df.sentiment == 'negative']
    neg_retweet = 0
    for tweet in neg_tweets: 
        neg_retweet = neg_retweet + tweet['retweet_count']
    
    pos_count = pos_tweets.shape[0] + pos_retweet 
    neut_count = neut_tweets.shape[0] + neut_retweet
    neg_count = neg_tweets.shape[0] + neg_retweet

    results = [{'Positive Tweets' : pos_count,
                'Neutral Tweets': neut_count,
                'Negative Tweets' : neg_count}]
    return(results)  


def word_cloud(df):

    """ Get significant words/phrases for each sentiment """ 
    
    pos_tweets = df[df.sentiment == 'positive']
    neut_tweets = df[df.sentiment == 'neutral']
    neg_tweets = df[df.sentiment == 'negative']



def plot_chart(results):
    fig = px.pie(results)
    pieJSON = json.dumps(fig, clas = plotly.utils.PlotlyJSONEncoder)
    return pieJSON 

def clean_tweets(tweet):
    tweet = re.sub('@[A-Za-z0-9]+', '', tweet) #Removing @mentions
    tweet = re.sub('#', '', tweet) # Removing '#' hash tag
    tweet = re.sub('RT[\s]+', '', tweet) # Removing RT
    tweet = re.sub('https?:\/\/\S+', '', tweet) # Removing hyperlink
    
    return tweet

def get_polarity(tweet):
    sentiment = analyser.polarity_scores(tweet)
    return sentiment

def get_Sentiment(polarity):
    if polarity['compound'] >= 0.05:
        return 'positive'
    elif polarity['compound'] < 0.05 and polarity['compound'] > -0.05:
        return 'neutral'
    elif polarity['compound'] <= -0.05:
        return 'negative' 

def get_predictions(df):
    df['text'] = df['text'].apply(clean_tweets)
    df['polarity'] = df['text'].apply(get_polarity)
    df['sentiment'] = df['polarity'].apply(get_Sentiment)

    return(df)


    

