import tweepy
import csv
import pandas as pd
####input your credentials here
consumer_key = 'iBok2Zb1OoZgYmHGaG0kS9ypB'
consumer_secret = 'dJwEN2FGfvRUzBGWTzKKHTiN7rXC0NFUrLwYQVx5rzW6JjW80m'
access_token = '958545588806660097-2KW6NZBcdh769sBUGyoJR59iFjT1fvC'
access_token_secret = '4lOjod7zICMgxg690abGb7A3lQ7eMEJoqf755BE1iwdqJ'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

# Open/Create a file to append data
csvFile = open('dataset.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search,q="#spyder",count=100,
                           lang="en",
                           since="2017-01-03").items():
    print (tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
