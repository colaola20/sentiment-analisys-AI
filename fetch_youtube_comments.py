import os
import time
import logging
from typing import List, Dict, Iterable, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import urllib.parse
from dotenv import load_dotenv

from reddit_non_tech_subs.text_utils import clean_text
from reddit_non_tech_subs.sentiment import score_vader_batch, hf_label_to_vader_like

load_dotenv()

api_key = os.getenv("YOUTUBE_API_KEY")

def yt_client(api_key: str):
    return build('youtube', 'v3', developerKey=api_key, cache_discovery=False)


def search_videos(client, query: str, max_results=50) -> List[str]:
    videos =[]

    # Search videos
    req = client.search().list(
        q=query,
        part="id",
        type="video",
        maxResults=min(max_results, 50),   # max per call unless pagination is used
    )
    resp = req.execute()
    for item in resp.get("items", []):
        vid = item["id"].get("videoId")
        if vid:
            videos.append(vid)

    return videos

def iter_comments_for_video(client, video_id: str, per_video_limit: int = 100) -> Iterable[Dict]:
    """Yield top-level comments (and optionally replies)."""
    collected = 0
    page_token = None
    while collected < per_video_limit:
        try:
            req = client.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,
                pageToken=page_token,
                textFormat="plainText",
            )
            resp = req.execute()
        except HttpError as e:
            # If comments are disabled for this video, skip it quietly
            try:
                status = e.resp.status
            except Exception:
                status = None

            # Attempt to read the API error reason from the response content if present
            reason = None
            try:
                content = e.content
                # e.content is bytes; decode and check for reason text
                if content:
                    txt = content.decode("utf-8", errors="ignore")
                    if "commentsDisabled" in txt:
                        reason = "commentsDisabled"
                    elif "quotaExceeded" in txt:
                        reason = "quotaExceeded"
            except Exception:
                txt = None

            if status == 403 and reason == "commentsDisabled":
                logging.info("Comments are disabled for video %s, skipping.", video_id)
                return  # stop iterator for this video
            # For quota or other transient 5xx errors, re-raise so outer retry/backoff can handle
            logging.warning("YouTube API error for video %s: %s", video_id, e)
            raise


        for item in resp.get("items", []):
            top =item["snippet"]["topLevelComment"]["snippet"]
            yield {
                "video_id": video_id,
                "comment_id": item["snippet"]["topLevelComment"]["id"],
                "parent_id": None,
                "author": top.get("authorDisplayName"),
                "published_at": top.get("publishedAt"),
                "like_count": top.get("likeCount"),
                "reply_count": item["snippet"].get("totalReplyCount", 0),
                "content": top.get("textDisplay"),
            }
            collected += 1
            if collected >= per_video_limit:
                break

            # yield replies if present and you want them
            replies = item.get("replies", {}).get("comments", [])
            for r in replies:
                rs = r["snippet"]
                yield {
                    "video_id": video_id,
                    "comment_id": r["id"],
                    "parent_id": item["snippet"]["topLevelComment"]["id"],
                    "author": rs.get("authorDisplayName"),
                    "published_at": rs.get("publishedAt"),
                    "like_count": rs.get("likeCount"),
                    "reply_count": None,
                    "content": rs.get("textDisplay"),
                }
                collected += 1
                if collected >= per_video_limit:
                    break

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

def collect_for_query(client, query, cfg, ai_filter_fn):
    videos = search_videos(client, query, max_results=cfg.videos_per_query)
    rows = []
    for vid in videos:
        # Optionally fetch video title and description via videos().list(part='snippet')
        for c in iter_comments_for_video(client, vid, per_video_limit=cfg.comments_per_video):
            cleaned = clean_text(c["content"])
            if cfg.search_ai and not ai_filter_fn(cleaned, True):
                continue
            rows.append({
                "video_id": c["video_id"],
                "id": c["comment_id"],
                "parent_id": c["parent_id"],
                "author": c["author"],
                "created_at": c["published_at"],
                "likes": c["like_count"],
                "content": cleaned,
            })

    return rows


def with_retries(fn, max_attempts=5, base_sleep=1):
    attempt = 0
    while True:
        try:
            return fn()
        except HttpError as e:
            attempt += 1
            if attempt >= max_attempts:
                raise
            sleep = min(base_sleep * 2**(attempt-1), 60)
            logging.info("HttpError, sleeping %.1fs and retrying (%d/%d)", sleep, attempt, max_attempts)
            time.sleep(sleep)


# --- minimal CLI / runner (paste at end of file) ---

from dataclasses import dataclass
import argparse
import pandas as pd
from pathlib import Path


@dataclass
class YTConfig:
    queries: List[str]
    videos_per_query: int = 10
    comments_per_video: int = 100
    search_ai: bool = True
    outdir: str = "youtube_data"

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fetch YouTube comments (minimal)")
    p.add_argument("--queries", type=str, default="ChatGPT,generative AI", help="Comma-separated queries")
    p.add_argument("--videos-per-query", type=int, default=5)
    p.add_argument("--comments-per-video", type=int, default=50)
    p.add_argument("--no-ai-filter", action="store_true")
    p.add_argument("--outdir", type=str, default="youtube_data")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    cfg = YTConfig(
        queries=[q.strip() for q in args.queries.split(",") if q.strip()],
        videos_per_query=args.videos_per_query,
        comments_per_video=args.comments_per_video,
        search_ai=not args.no_ai_filter,
        outdir=args.outdir,
    )

    if not api_key:
        raise SystemExit("Missing YOUTUBE_API_KEY (set in environment or .env)")

    client = yt_client(api_key)
    all_rows = []
    for q in cfg.queries:
        rows = collect_for_query(client, q, cfg, lambda txt, flag: txt and "chatgpt" in txt.lower() if flag else True)
        all_rows.extend(rows)

    if not all_rows:
        print("No rows collected.")
        return 0

    Path(cfg.outdir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    ts = time.strftime("%Y%m%d_%H%M")
    csv_path = Path(cfg.outdir) / f"youtube_comments_{ts}.csv"
    df.to_csv(csv_path, index=True)
    print(f"Wrote {len(df)} rows to {csv_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

# python fetch_youtube_comments.py --queries "ChatGPT developer tutorial, AI coding tools, GitHub Copilot review, LLM automation workflow, AI replaces programmers, AI tools productivity" --videos-per-query 50 --comments-per-video 100


# --queries "ChatGPT developer tutorial, AI coding tools, GitHub Copilot review, LLM automation workflow, AI replaces programmers, AI tools productivity"

#export YOUTUBE_API_KEY="YOUR_KEY"
# python fetch_youtube_comments.py --queries "ChatGPT" --videos-per-query 5 --comments-per-video 20 --outdir youtube_test