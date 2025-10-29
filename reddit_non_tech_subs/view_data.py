#!/usr/bin/env python3
"""Quick data viewer for Reddit sentiment data."""

import pandas as pd
import sys
from pathlib import Path

def view_latest_data():
    """View the most recent data file."""
    data_dir = Path("reddit_non_tech_subs/data")
    csv_files = list(data_dir.glob("reddit_ai_nontech_*.csv"))
    
    if not csv_files:
        print("No data files found in reddit_non_tech_subs/data/")
        return
    
    # Get the most recent file
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"Viewing: {latest_file.name}")
    
    df = pd.read_csv(latest_file)
    print(f"\nDataset Summary:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    
    # Sentiment breakdown
    if 'sent_label' in df.columns:
        print(f"\nSentiment Distribution:")
        sentiment_counts = df['sent_label'].value_counts()
        for label, count in sentiment_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {label}: {count:,} ({pct:.1f}%)")
    
    # Subreddit breakdown
    if 'subreddit' in df.columns:
        print(f"\nTop Subreddits:")
        subreddit_counts = df['subreddit'].value_counts().head(10)
        for sub, count in subreddit_counts.items():
            print(f"  r/{sub}: {count:,} posts/comments")
    
    # Sample posts
    print(f"\nSample Posts (first 3):")
    sample_posts = df[df['type'] == 'post'].head(3)
    for i, (_, row) in enumerate(sample_posts.iterrows(), 1):
        print(f"\n  {i}. r/{row['subreddit']} - {row['sent_label']}")
        print(f"     Score: {row['score']} | Comments: {row['num_comments']}")
        print(f"     Content: {row['content'][:100]}...")
        print(f"     URL: {row['url']}")
    
    # Sample comments
    print(f"\nSample Comments (first 3):")
    sample_comments = df[df['type'] == 'comment'].head(3)
    for i, (_, row) in enumerate(sample_comments.iterrows(), 1):
        print(f"\n  {i}. r/{row['subreddit']} - {row['sent_label']}")
        print(f"     Score: {row['score']}")
        print(f"     Content: {row['content'][:100]}...")

def view_sentiment_analysis():
    """Show sentiment analysis by subreddit."""
    data_dir = Path("reddit_non_tech_subs/data")
    csv_files = list(data_dir.glob("reddit_ai_nontech_*.csv"))
    
    if not csv_files:
        print("No data files found")
        return
    
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)
    
    if 'sent_label' not in df.columns:
        print("No sentiment data found")
        return
    
    print(f"Sentiment Analysis by Subreddit:")
    print("=" * 60)
    
    # Group by subreddit and sentiment
    sentiment_by_sub = df.groupby(['subreddit', 'sent_label']).size().unstack(fill_value=0)
    
    # Calculate percentages
    sentiment_pct = sentiment_by_sub.div(sentiment_by_sub.sum(axis=1), axis=0) * 100
    
    for subreddit in sentiment_pct.index:
        print(f"\nr/{subreddit}:")
        total = sentiment_by_sub.loc[subreddit].sum()
        print(f"   Total posts/comments: {total}")
        
        for sentiment in ['pos', 'neu', 'neg']:
            if sentiment in sentiment_pct.columns:
                count = sentiment_by_sub.loc[subreddit, sentiment]
                pct = sentiment_pct.loc[subreddit, sentiment]
                print(f"   {sentiment}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sentiment":
        view_sentiment_analysis()
    else:
        view_latest_data()