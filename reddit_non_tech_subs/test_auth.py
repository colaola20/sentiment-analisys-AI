#!/usr/bin/env python3
"""Quick Reddit auth test to debug 401 issues."""

import os
import sys
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

def test_reddit_auth():
    """Test Reddit authentication and show what's wrong."""
    print("=== Reddit Auth Debug ===")
    
    # Check env vars
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET") 
    user_agent = os.environ.get("REDDIT_USER_AGENT")
    
    print(f"REDDIT_CLIENT_ID: {'SET' if client_id else 'MISSING'}")
    if client_id:
        print(f"  Length: {len(client_id)} chars")
        print(f"  Starts with: {client_id[:4]}...")
    
    print(f"REDDIT_CLIENT_SECRET: {'SET' if client_secret else 'MISSING'}")
    if client_secret:
        print(f"  Length: {len(client_secret)} chars")
        print(f"  Starts with: {client_secret[:4]}...")
    
    print(f"REDDIT_USER_AGENT: {'SET' if user_agent else 'MISSING'}")
    if user_agent:
        print(f"  Value: {user_agent}")
    
    # Check for common issues
    issues = []
    if not client_id:
        issues.append("REDDIT_CLIENT_ID not set")
    elif len(client_id) < 10:
        issues.append(f"REDDIT_CLIENT_ID seems too short ({len(client_id)} chars)")
    
    if not client_secret:
        issues.append("REDDIT_CLIENT_SECRET not set")
    elif len(client_secret) < 20:
        issues.append(f"REDDIT_CLIENT_SECRET seems too short ({len(client_secret)} chars)")
    
    if not user_agent:
        issues.append("REDDIT_USER_AGENT not set")
    elif "by u/" not in user_agent:
        issues.append("REDDIT_USER_AGENT should include 'by u/yourusername'")
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nFix these and try again.")
        return False
    
    # Try actual auth
    print("\nTesting Reddit connection...")
    try:
        import praw
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            ratelimit_seconds=5,
        )
        reddit.read_only = True
        
        # Test with a simple call
        subreddit = reddit.subreddit("test")
        post = next(subreddit.hot(limit=1), None)
        if post:
            print(f"SUCCESS! Got post: {post.title[:50]}...")
            return True
        else:
            print("Connected but no posts found")
            return True
            
    except Exception as e:
        print(f"Auth failed: {e}")
        if "401" in str(e):
            print("\n401 means:")
            print("  - Wrong client_id or client_secret")
            print("  - App type mismatch (need 'script' app)")
            print("  - Secret doesn't match the app")
        return False

if __name__ == "__main__":
    success = test_reddit_auth()
    sys.exit(0 if success else 1)