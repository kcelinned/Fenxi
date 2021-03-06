from textblob import TextBlob
import csv
import pandas as pd 
import numpy as np
import re

#variable names of all documents

# Create a function to clean the tweets
def clean_tweets(tweet):
    tweet = re.sub('@[A-Za-z0-9]+', '', tweet) #Removing @mentions
    tweet = re.sub('#', '', tweet)
    tweet = re.sub('RT[\s]+', '', tweet) # Removing RT
    tweet = re.sub('https?:\/\/\S+', '', tweet) # Removing hyperlink
    
    return tweet

# find polarity of each tweet 
def get_polarity(tweet):
    return  TextBlob(tweet).sentiment.polarity

# Dataset cleaning  

def STS_cleaning(polarity):
    if polarity == 0:
        return 'negative'
    elif polarity == 2:
        return 'neutral'
    elif polarity == 4: 
        return 'positive'

def covid_cleaning(polarity):
    if polarity == 1:
        return 'positive'
    elif polarity == 2: 
        return 'negative'
    elif polarity == 3:
        return 'neutral'

# Different Classifiers

def get_Sentiment_1(polarity):
    if polarity > 0:
        return 'positive'
    elif polarity == 0:
        return 'neutral'
    elif polarity < 0: 
        return 'negative'

def get_Sentiment_2(polarity):
    if polarity >= 0.1:
        return 'positive'
    elif polarity < 0.1 and polarity > -0.1:
        return 'neutral'
    elif polarity <= -0.1:
        return 'negative'

def get_Sentiment_3(polarity):
    if polarity >= 0.05:
        return 'positive'
    elif polarity < 0.05 and polarity > -0.05:
        return 'neutral'
    elif polarity <= -0.05:
        return 'negative' 

# Evaluations 
def F1_Score(df, label):

    tp = df[(df['match'] == 1) & (df['actual'] == label)].shape[0]
    fp = df.loc[(df['match'] == 0) & (df['predicted'] == label)].shape[0]
    fn = df.loc[(df['match'] == 0) & (df['actual'] == label)].shape[0]

    if ((tp + fp) == 0) or ((tp+fn) == 0):
        f1 = 0
    else:
        precision = tp / (tp + fp) 
        recall = tp / (tp + fn)
        f1 = 2 * (precision*recall / (precision + recall))
    return f1

def get_evaluation(df):
    total = df.shape[0]

    pos_tweets = df[df.actual == 'positive'].shape[0]
    pos_predicted = df[df.predicted == 'positive'].shape[0]

    print("Actual Number of Positive Tweets: ", pos_tweets)
    print("Sentiment Analyzer Number of Positive Tweets: ", pos_predicted)

    neut_tweets = df[df.actual == 'neutral'].shape[0]
    neut_predicted = df[df.predicted== 'neutral'].shape[0]

    print("Actual Number of Neutral Tweets: ", neut_tweets)
    print("Sentiment Analyzer Number of Neutral Tweets: ", neut_predicted)

    neg_tweets = df[df.actual == 'negative'].shape[0]
    neg_predicted = df[df.predicted == 'negative'].shape[0]

    print("Actual Number of Negative Tweets: ", neg_tweets)
    print("Sentiment Analyzer Number of Negative Tweets: ", neg_predicted)
    # if actual sentiment == predicted sentiment 
    # put 1 in match 
    # otherwise put 0 
    df['match'] = np.where(df['actual']==df['predicted'],
    1, 0)

    print("Positive F1 Score:", F1_Score(df, 'positive'))
    print("Neutral F1 Score:", F1_Score(df, 'neutral'))
    print("Negative F1 Score:", F1_Score(df, 'negative'))

    wrongDf = df.loc[df['match'] == 0]
    wrongDf.to_csv("New/wrongExample.csv", index = False)

    # sum 
    true = df['match'].sum()
    print(true)

    accuracy = (true/total)*100
    print("Accuracy: ", accuracy,"%")

# STS Dataset
def main_STS():
    dataset = pd.read_csv("Dataset/STS.csv")
    
    dataset.columns = ['actual', 'id', 'date', 'query', 'user', 'text']
    df = pd.DataFrame(dataset, columns= ['actual','text'])

    df['text'] = df['text'].apply(clean_tweets)
    df['actual'] = df['actual'].apply(STS_cleaning)
    df['polarity'] = df['text'].apply(get_polarity)
    for x in range(0,3):
        if x == 0:
            df['predicted'] = df['polarity'].apply(get_Sentiment_1)
            get_evaluation(df)
        elif x == 1:
            df['predicted'] = df['polarity'].apply(get_Sentiment_2)
            get_evaluation(df)
        else:
            df['predicted'] = df['polarity'].apply(get_Sentiment_3)
            get_evaluation(df)

print("---- STS ----")
main_STS()

# Covid Datset
def main_covid():
    dataset = pd.read_csv("Dataset/covid.csv")
    dataset = dataset.rename(columns = {'label':'actual' ,'tweet':'text'})

    dataset['text'] = dataset['text'].apply(clean_tweets)
    dataset['actual'] = dataset['actual'].apply(covid_cleaning)
    dataset['polarity'] = dataset['text'].apply(get_polarity)

    for x in range(0,3):
        if x == 0:
            dataset['predicted'] = dataset['polarity'].apply(get_Sentiment_1)
            get_evaluation(dataset)
        elif x == 1:
            dataset['predicted'] = dataset['polarity'].apply(get_Sentiment_2)
            get_evaluation(dataset)
        else:
            dataset['predicted'] = dataset['polarity'].apply(get_Sentiment_3)
            get_evaluation(dataset)

print("---- COVID ---")
main_covid()

# Airlines Dataset

def main_airlines():
    dataset = pd.read_csv("Dataset/airline.csv")
    #dataset = dataset.rename(columns = {'sentiment':'actual'})
    dataset = dataset.rename(columns = {'airline_sentiment':'actual'})
   
    dataset['text'] = dataset['text'].apply(clean_tweets)
    dataset['polarity'] = dataset['text'].apply(get_polarity)
    for x in range(0,3):
        if x == 0:
            dataset['predicted'] = dataset['polarity'].apply(get_Sentiment_1)
            get_evaluation(dataset)
        elif x == 1:
            dataset['predicted'] = dataset['polarity'].apply(get_Sentiment_2)
            get_evaluation(dataset)
        else:
            dataset['predicted'] = dataset['polarity'].apply(get_Sentiment_3)
            get_evaluation(dataset)

print("---- Airlines ----")
main_airlines()


# Debate Dataset

def main_debate():
    dataset = pd.read_csv("Dataset/Debate.csv")
    df = pd.DataFrame(dataset, columns=['sentiment', 'text'])
    df = df.rename(columns = {'sentiment':'actual'})

    df['text'] = df['text'].apply(clean_tweets)
    df['polarity'] = df['text'].apply(get_polarity)
    for x in range(0,3):
        if x == 0:
            df['predicted'] = df['polarity'].apply(get_Sentiment_1)
            get_evaluation(df)
        elif x == 1:
            df['predicted'] = df['polarity'].apply(get_Sentiment_2)
            get_evaluation(df)
        else:
            df['predicted'] = df['polarity'].apply(get_Sentiment_3)
            get_evaluation(df)

print("----Debate-----")
main_debate()


