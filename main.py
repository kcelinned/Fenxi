import tweepy
import config

auth = tweepy.AppAuthHandler(config.CONSUMER_KEY, config.CONSUMER_SECRET)
api = tweepy.API(auth, wait_on_rate_limit = True) 

collectTweets = api.search("Boris", 'en', count = 5, result_type = 'mixed')

for tweet in collectTweets:
    print(tweet.text )

