import os
import praw
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret = os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="ai-sentiment-project by u/Curious_Art_8680"
)

SUBREDDITS = ["MachineLearning", "technology", "cscareerquestions"]
SEARCH_LIMIT = 1000
TOP_LIMIT = 100
KEYWORDS = ["chatgpt", "generative ai", "artificial intelligence", "ai tools", "ai jobs", "llm"]


def matches_keywords(text: str) -> bool:
    if not text:
        return False
    
    lower = text.lower()
    return any(k in lower for k in KEYWORDS)

def fetch_reddit():
    rows = []
    for sub in SUBREDDITS:
        print(f"Processing subreddit: {sub}")
        subreddit = reddit.subreddit(sub)
        query = " OR ".join(KEYWORDS)
        for search_post in subreddit.search(query, limit=SEARCH_LIMIT, sort="new"):
            rows.append({
                "source": "post",
                "subreddit": sub,
                "id": search_post.id,
                "created_utc": search_post.created_utc,
                "text": search_post.title + "\n" + (search_post.selftext or ""),
                "score": getattr(search_post, "score", None),
                "num_comments": getattr(search_post, "num_comments", None)
            })

        
        for top_post in subreddit.top(limit=TOP_LIMIT):
            try:
                top_post.comments.replace_more(limit=0)
            except Exception:
                continue
            for c in top_post.comments.list():
                body = getattr(c, "body", "") or ""
                if matches_keywords(body):
                    rows.append({
                        "source": "comment",
                        "subreddit": sub,
                        "id": c.id,
                        "created_utc": getattr(c, "created_utc", None),
                        "text": body,
                        "score": getattr(c, "score", None)
                    })

    df = pd.DataFrame(rows)
    df.to_csv("reddit_ai_posts_comments.csv", index=True)
    print(f"Done. Total rows saved: {len(df)}")


if __name__ == "__main__":
    fetch_reddit()
