import os
from tweepy import Client
from dotenv import load_dotenv
load_dotenv()

import pandas as pd

BEARER = os.getenv("TWITTER_BEARER")
client = Client(bearer_token=BEARER, wait_on_rate_limit=True)

query = '("ChatGPT" OR "AI" OR "Artificial Intelligence" OR "OpenAI" OR "large language model" OR "LLM" OR "Automation") -is:retweet lang:en'

tweets = client.search_recent_tweets(
    query=query,
    tweet_fields=["created_at","public_metrics","lang"],
    expansions=["author_id"],
    user_fields=["username","public_metrics"],
    max_results=100
)

data = []
if tweets.data:
    for tweet in tweets.data:
        data.append({
            "id": tweet.id,
            "text": tweet.text,
            "author_id": tweet.author_id,
            "created_at": tweet.created_at,
            "retweets": tweet.public_metrics.get("reteet_count"),
            "likes": tweet.public_metrics.get("like_count"),
        })

df = pd.DataFrame(data)
df.to_csv("twitter_ai_recent.csv", index=False)