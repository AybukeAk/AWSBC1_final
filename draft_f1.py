import tweepy
import json

consumer_key=""
consumer_secret=""
access_token=""
access_token_secret=""

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit = True)

keyword = "miuul"

data = []

for tweets in tweepy.Cursor(api.search_tweets, lang="tr",geocode='38.9637,35.2433,1000km', q=keyword, count=100).pages():
    for tweet in tweets:
        data.append(tweet._json)
    
filename = keyword+"search_tweets_29.json"

with open (filename,"w") as outfile:
     json.dump(data,outfile)
     
