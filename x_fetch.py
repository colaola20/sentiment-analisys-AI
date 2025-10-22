import os
from tweepy import Client
from dotenv import load_dotenv
load_dotenv()

import pandas as pd

BEARER = os.getenv("TWITTER_BEARER")
client = Client(bearer_token=BEARER, wait_on_rate_limit=True)

query = '("ChatGPT" OR "generative AI" OR "large language model" OR "LLM") -is:retweet lang:en'

resp = client.search_recent_tweets(
    query=query,
    tweet_fields=["author_id","created_at","public_metrics","lang"],
    max_results=100
)

rows = []
if resp.data:
    for t in resp.data:
        rows.append({
            "id": t.id,
            "text": t.text,
            "author_id": t.author_id,
            "created_at": t.created_at,
            "retweets": t.public_metrics.get("reteet_count"),
            "likes": t.public_metrics.get("like_count"),
        })

df = pd.DataFrame(rows)
df.to_csv("twitter_ai_recent.csv", index=False)