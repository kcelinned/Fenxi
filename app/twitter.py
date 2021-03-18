import tweepy
import config
import pandas as pd 
import numpy as np

auth = tweepy.AppAuthHandler(config.CONSUMER_KEY, config.CONSUMER_SECRET)
api = tweepy.API(auth, wait_on_rate_limit = True) 

# write a function to extract tweets 
def get_tweets(query):
    count = 5
    tweets_dict = [] 
    fil_query = query + " -filter:retweets -has:media"
    print(fil_query)
    try:
        for tweet in api.search(q= fil_query, count = count, lang = 'en'):
            tweets_dict.append({ 'id' : tweet.id,
                                'text' : tweet.text,
                                'retweets': tweet.retweet_count})
    except BaseException as e:
        print('failed on_status', str(e)) 
    
    df = pd.DataFrame.from_dict(tweets_dict)
    return(df)
