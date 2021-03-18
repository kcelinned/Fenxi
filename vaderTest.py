from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv
import pandas as pd 
import numpy as np
import re
import nltk 
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

analyser = SentimentIntensityAnalyzer()
en_stops = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer() 
  
# create a function to clean the tweets

def clean_tweets(tweet):
    tweet = re.sub('@[A-Za-z0–9]+', '', str(tweet)) #Removing @mentions
    tweet = re.sub('#', '', str(tweet)) # Removing '#' hash tag
    tweet = re.sub('RT[\s]+', '', str(tweet)) # Removing RT
    tweet = re.sub('https?:\/\/\S+', '', str(tweet)) # Removing hyperlink
    
    return tweet

def remove_stopw(tweet):
    words = tweet.split()
    tweet = ""
    for word in words: 
        if word not in en_stops:
            tweet = tweet + " " + word
    return tweet

def get_pos(word):
    tag = nltk.pos_tag([word])[0][1]
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatization(tweet):
    words = nltk.word_tokenize(tweet)
    tweet = ""
    for word in words:
        new = lemmatizer.lemmatize(word,get_pos(word))
        tweet = tweet + " " + new
    return tweet
    
# find polarity of each tweet
def get_polarity(tweet):
    sentiment = analyser.polarity_scores(tweet)
    return sentiment

# dataset cleaning 
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

# different classifiers 
def get_Sentiment_1(polarity):
    if polarity['compound'] > 0:
        return 'positive'
    elif polarity['compound'] == 0:
        return 'neutral'
    elif polarity['compound'] < 0: 
        return 'negative'


# Evaluations 
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
    print("Removing Stopwords")
    df['text'] = df['text'].apply(remove_stopw)
    df['polarity'] = df['text'].apply(get_polarity)
    df['predicted'] = df['polarity'].apply(get_Sentiment_1)
    get_evaluation(df)
    print("Lemmatization")
    df['text'] = df['text'].apply(lemmatization)
    df['polarity'] = df['text'].apply(get_polarity)
    df['predicted'] = df['polarity'].apply(get_Sentiment_1)
    get_evaluation(df)

print("---- STS ----")
main_STS()

# Covid Datset
def main_covid():
    dataset = pd.read_csv("Dataset/covid.csv")
    dataset = dataset.rename(columns = {'label':'actual' ,'tweet':'text'})

    dataset['text'] = dataset['text'].apply(clean_tweets)
    dataset['actual'] = dataset['actual'].apply(covid_cleaning)
    print("Removing Stopwords")
    dataset['text'] = dataset['text'].apply(remove_stopw)
    dataset['polarity'] = dataset['text'].apply(get_polarity)
    dataset['predicted'] = dataset['polarity'].apply(get_Sentiment_1)
    get_evaluation(dataset)
    print("Lemmatization")
    dataset['text'] = dataset['text'].apply(lemmatization)
    dataset['polarity'] = dataset['text'].apply(get_polarity)
    dataset['predicted'] = dataset['polarity'].apply(get_Sentiment_1)
    get_evaluation(dataset)


print("---- COVID ---")
main_covid()

# Debate Dataset

def main_debate():
    dataset = pd.read_csv("Dataset/Debate.csv")
    df = pd.DataFrame(dataset, columns=['sentiment', 'text'])
    df = df.rename(columns = {'sentiment':'actual'})

    df['text'] = df['text'].apply(clean_tweets)
    print("Removing Stopwords")
    df['text'] = df['text'].apply(remove_stopw)
    df['polarity'] = df['text'].apply(get_polarity)
    df['predicted'] = df['polarity'].apply(get_Sentiment_1)
    get_evaluation(df)
    print("Lemmatization")
    df['text'] = df['text'].apply(lemmatization)
    df['polarity'] = df['text'].apply(get_polarity)
    df['predicted'] = df['polarity'].apply(get_Sentiment_1)
    get_evaluation(df)


print("----Debate-----")
main_debate()

# Airlines Dataset

def main_airlines():
    dataset = pd.read_csv("Dataset/airline.csv")
    dataset = dataset.rename(columns = {'airline_sentiment':'actual'})
   
    dataset['text'] = dataset['text'].apply(clean_tweets)
    print("Removing Stopwords")
    dataset['text'] = dataset['text'].apply(remove_stopw)
    dataset['polarity'] = dataset['text'].apply(get_polarity)
    dataset['predicted'] = dataset['polarity'].apply(get_Sentiment_1)
    get_evaluation(dataset)
    print("Lemmatization")
    dataset['text'] = dataset['text'].apply(lemmatization)
    dataset['polarity'] = dataset['text'].apply(get_polarity)
    dataset['predicted'] = dataset['polarity'].apply(get_Sentiment_1)
    get_evaluation(dataset)


print("---- Train ----")
main_airlines()
