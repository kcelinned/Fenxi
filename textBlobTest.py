from textblob import TextBlob
import csv
import pandas as pd 
import numpy as np
import re


# Create a function to clean the tweets
def cleanTweets(tweet):
 tweet = re.sub('@[A-Za-z0–9]+', '', tweet) #Removing @mentions
 tweet = re.sub('#', '', tweet) # Removing '#' hash tag
 tweet = re.sub('RT[\s]+', '', tweet) # Removing RT
 tweet = re.sub('https?:\/\/\S+', '', tweet) # Removing hyperlink
 
 return tweet

# find polarity of each tweet 
def get_polarity(tweet):
    return  TextBlob(tweet).sentiment.polarity

# find subjectivity of each tweet 
def get_subjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity



#classify tweets
def get_multiSentiment(polarity):
    if polarity > 0:
        return 'positive'
    elif polarity == 0:
        return 'neutral'
    elif polarity < 0: 
        return 'negative'


def get_percentage(df):
    # get total number of tweets
    total = df.shape[0]

    pos_tweets = df[df.airline_sentiment == 'positive'].shape[0]
    pos_predicted = df[df.predict_sentiment == 'positive'].shape[0]

    print("Actual Number of Positive Tweets: ", pos_tweets)
    print("Sentiment Analyzer Number of Positive Tweets: ", pos_predicted)

    neut_tweets = df[df.airline_sentiment == 'neutral'].shape[0]
    neut_predicted = df[df.predict_sentiment == 'neutral'].shape[0]

    print("Actual Number of Neutral Tweets: ", neut_tweets)
    print("Sentiment Analyzer Number of Neutral Tweets: ", neut_predicted)

    neg_tweets = df[df.airline_sentiment == 'negative'].shape[0]
    neg_predicted = df[df.predict_sentiment == 'negative'].shape[0]

    print("Actual Number of Negative Tweets: ", neg_tweets)
    print("Sentiment Analyzer Number of Negative Tweets: ", neg_predicted)
    # if actual sentiment == predicted sentiment 
    # put 1 in match 
    # otherwise put 0 
    df['match'] = np.where(df['airline_sentiment']==df['predict_sentiment'],
    1, 0)

    # sum 
    true = df['match'].sum()
    print(true)

    accuracy = (true/total)*100
    print("Accuracy: ",accuracy,"%")

#print out percentages 

def main_airline():
    # import datasett
    dataset = pd.read_csv("Dataset/Tweets.csv")
    # extract the tweets from dataset 
    df = pd.DataFrame(dataset, columns= ['text','airline_sentiment'])
    # clean data
    #   remove @s / urls / hashtags 

    # Clean the tweets
    df['text'] = df['text'].apply(cleanTweets)

    df['polarity'] = df['text'].apply(get_polarity)
    df['subjectivity'] = df['text'].apply(get_subjectivity) 

    df['predict_sentiment'] = df['polarity'].apply(get_multiSentiment)

    get_percentage(df) 

def get_sentiment(polarity):
    if polarity >= 0:
        return 1
    elif polarity < 0:
        return 0

def get_percentage2(df, actual, predicted):
    # get total number of tweets
    total = df.shape[0]

    pos_tweets = df[df.actual == 'positive'].shape[0]
    pos_predicted = df[df.predicted == 'positive'].shape[0]

    print("Actual Number of Positive Tweets: ", pos_tweets)
    print("Sentiment Analyzer Number of Positive Tweets: ", pos_predicted)

    neut_tweets = df[df.actual == 'neutral'].shape[0]
    neut_predicted = df[df.predicted == 'neutral'].shape[0]

    print("Actual Number of Neutral Tweets: ", neut_tweets)
    print("Sentiment Analyzer Number of Neutral Tweets: ", neut_predicted)

    neg_tweets = df[df.actual == 'negative'].shape[0]
    neg_predicted = df[df.predicted == 'negative'].shape[0]

    print("Actual Number of Negative Tweets: ", neg_tweets)
    print("Sentiment Analyzer Number of Negative Tweets: ", neg_predicted)
    # if actual sentiment == predicted sentiment 
    # put 1 in match 
    # otherwise put 0 
    df['match'] = np.where(df[actual]==df[predicted],
    1, 0)

    # sum 
    true = df['match'].sum()
    print(true)

    accuracy = (true/total)*100
    print("Accuracy: ",accuracy,"%")



def main_example():
    dataset = pd.read_csv("Dataset/example.csv")
    
    
    dataset['SentimentText'] = dataset['SentimentText'].apply(cleanTweets)
    dataset['polarity'] = dataset['SentimentText'].apply(get_polarity)
    dataset['subjectivity'] = dataset['SentimentText'].apply(get_subjectivity) 

    dataset['predict_sentiment'] = dataset['polarity'].apply(get_sentiment)

    get_percentage2(dataset, 'Sentiment', 'predict_sentiment')

#main_example()
main_airline()



