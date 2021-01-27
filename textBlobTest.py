from textblob import TextBlob
import csv
import pandas as pd 
import numpy as np
import re

# import datasett
dataset = pd.read_csv("Dataset/Tweets.csv")
# extract the tweets from dataset 
df = pd.DataFrame(dataset, columns= ['text','airline_sentiment'])
# clean data
#   remove @s / urls / hashtags 

# Create a function to clean the tweets
def cleanTweets(tweet):
 tweet = re.sub('@[A-Za-z0–9]+', '', tweet) #Removing @mentions
 tweet = re.sub('#', '', tweet) # Removing '#' hash tag
 tweet = re.sub('RT[\s]+', '', tweet) # Removing RT
 tweet = re.sub('https?:\/\/\S+', '', tweet) # Removing hyperlink
 
 return tweet


# Clean the tweets
df['text'] = df['text'].apply(cleanTweets)

# remove stopwords?
# tokenization? 

# find polarity of each tweet 
def get_polarity(tweet):
    return  TextBlob(tweet).sentiment.polarity

# find subjectivity of each tweet 
def get_subjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity

df['polarity'] = df['text'].apply(get_polarity)
df['subjectivity'] = df['text'].apply(get_subjectivity) 

print(df['polarity'][1])

#classify tweets
def get_sentiment(polarity):
    if polarity > 0:
        return 'positive'
    elif polarity == 0:
        return 'neutral'
    elif polarity < 0: 
        return 'negative'

df['predict_sentiment'] = df['polarity'].apply(get_sentiment)

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

get_percentage(df)   



