from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import praw
import prawcore

from .config import DEFAULT_SUBREDDITS, RuntimeDefaults
from .io_utils import append_run_log, load_env, write_outputs, ensure_dir
from .sentiment import score_vader_batch, score_hf_batch, hf_label_to_vader_like
from .text_utils import clean_text


LOGGER = logging.getLogger("reddit_non_tech_subs")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@dataclass
class FetchConfig:
    subreddits: List[str]
    mode: str
    time_filter: str
    posts_per_sub: int
    top_level_comment_limit: int
    use_hf: bool
    outdir: str
    search_ai: bool = True
    dry_run: bool = False


def _reddit_client() -> praw.Reddit:
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT")
    if not client_id or not client_secret or not user_agent:
        raise RuntimeError(
            "Missing Reddit credentials. Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT."
        )
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        ratelimit_seconds=5,
    )
    # Ensure read-only usage
    reddit.read_only = True
    return reddit


def _iter_posts(subreddit: praw.models.Subreddit, mode: str, time_filter: str, limit: int, search_terms: list = None):
    if search_terms:
        # Search for AI-related posts
        query = " OR ".join(search_terms)
        return subreddit.search(query, sort="top", time_filter=time_filter, limit=limit)
    elif mode == "new":
        return subreddit.new(limit=limit)
    # default: top
    return subreddit.top(time_filter=time_filter, limit=limit)


def _retry_sleep(attempt: int) -> None:
    # exponential backoff with cap
    sleep_s = min(2 ** attempt, 60)
    LOGGER.info("Rate limited or transient error, sleeping %.1fs", sleep_s)
    time.sleep(sleep_s)


def _contains_ai_terms(text: str, search_ai: bool) -> bool:
    """Check if text contains AI-related terms."""
    if not search_ai:
        return True  # No filter if search_ai is False
    
    if not text:
        return False
    
    text_lower = text.lower()
    ai_keywords = [
        " ai ", "artificial intelligence", "chatgpt", "gpt", "machine learning",
        "automation", "robot", "algorithm", "neural network", "deep learning",
        "llm", "large language model", "openai", "claude", "gemini",
        "automated", "ai-generated", "ai tool", "ai system", "ai model",
        "chat bot", "chatbot", "copilot", "midjourney", "dalle", "stable diffusion",
        "ai assistant", "ai writing", "ai art", "ai video", "ai music",
        "job displacement", "automation", "ai replacing", "ai taking over",
    ]
    
    # Check if any keyword appears in the text
    for keyword in ai_keywords:
        if keyword in text_lower:
            return True
    
    return False


def fetch(
    cfg: FetchConfig,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Fetch posts and top-level comments.

    Returns (posts, comments)
    """
    reddit = _reddit_client()
    posts: List[Dict[str, object]] = []
    comments: List[Dict[str, object]] = []

    for sub in cfg.subreddits:
        LOGGER.info("Subreddit: r/%s", sub)
        sr = reddit.subreddit(sub)
        attempt = 0
        max_attempts = 5
        posts_checked = 0
        posts_filtered = 0
        while True:
            try:
                # AI search terms for filtering posts
                # Fetch 2x limit to account for filtering
                fetch_limit = cfg.posts_per_sub * 2 if cfg.search_ai else cfg.posts_per_sub
                ai_terms = ["AI", "artificial intelligence", "ChatGPT", "GPT", "machine learning", "automation", "robot", "algorithm"] if cfg.search_ai else None
                for p in _iter_posts(sr, cfg.mode, cfg.time_filter, fetch_limit, ai_terms):
                    # Skip stickied posts
                    if getattr(p, "stickied", False):
                        continue
                    
                    # Stop if we have enough posts (after filtering)
                    if len([p for p in posts if p.get("subreddit") == sub]) >= cfg.posts_per_sub:
                        break
                    
                    # Get post content
                    title = str(getattr(p, "title", "") or "")
                    selftext = str(getattr(p, "selftext", "") or "")
                    combined_text = f"{title} {selftext}".strip()
                    posts_checked += 1
                    
                    # Filter: Only include posts that actually mention AI-related terms
                    if cfg.search_ai and not _contains_ai_terms(combined_text, cfg.search_ai):
                        posts_filtered += 1
                        continue
                    
                    post_entry = {
                        "subreddit": str(sub),
                        "type": "post",
                        "id": str(p.id),
                        "parent_id": None,
                        "created_utc": float(getattr(p, "created_utc", 0.0) or 0.0),
                        "score": int(getattr(p, "score", 0) or 0),
                        "num_comments": int(getattr(p, "num_comments", 0) or 0),
                        "url": f"https://www.reddit.com{getattr(p, 'permalink', '')}",
                        "title": title,
                        "selftext": selftext,
                    }
                    posts.append(post_entry)

                    # Load comments and fetch top-level up to limit
                    try:
                        p.comment_sort = "top"
                        p.comment_limit = cfg.top_level_comment_limit
                        p.comments.replace_more(limit=0)
                        cnt = 0
                        for c in p.comments:
                            if cnt >= cfg.top_level_comment_limit:
                                break
                            if getattr(c, "is_submitter", False) is None:
                                pass
                            comment_body = getattr(c, "body", "") or ""
                            if not comment_body:
                                continue
                            
                            # Filter: Only include comments that mention AI-related terms
                            if cfg.search_ai and not _contains_ai_terms(comment_body, cfg.search_ai):
                                continue
                            
                            comments.append(
                                {
                                    "subreddit": str(sub),
                                    "type": "comment",
                                    "id": str(getattr(c, "id", "")),
                                    "parent_id": str(getattr(p, "id", "")),
                                    "created_utc": float(getattr(c, "created_utc", 0.0) or 0.0),
                                    "score": int(getattr(c, "score", 0) or 0),
                                    "num_comments": None,
                                    "url": f"https://www.reddit.com{getattr(p, 'permalink', '')}",
                                    "body": comment_body,
                                }
                            )
                            cnt += 1
                    except prawcore.exceptions.TooLarge:
                        LOGGER.info("Comment forest too large, skipping some comments")
                    except prawcore.exceptions.PrawcoreException as ce:
                        LOGGER.warning("Comment fetch error: %s", ce)
                break
            except (
                prawcore.exceptions.ResponseException,
                prawcore.exceptions.ServerError,
                prawcore.exceptions.OAuthException,
                prawcore.exceptions.RequestException,
            ) as e:
                attempt += 1
                # Fast-fail on 401 Unauthorized to prompt credential fix
                if isinstance(e, prawcore.exceptions.ResponseException):
                    try:
                        status = getattr(e.response, "status_code", None)
                    except Exception:
                        status = None
                    if status == 401:
                        LOGGER.error("Reddit authentication failed (401). Check REDDIT_CLIENT_ID/SECRET/USER_AGENT.")
                        raise
                if attempt > max_attempts:
                    LOGGER.error("Max retries reached for r/%s; skipping subreddit.", sub)
                    break
                LOGGER.warning("Fetch error on r/%s: %s (attempt %d)", sub, e, attempt)
                _retry_sleep(attempt)
                continue
            except prawcore.exceptions.PrawcoreException as e:
                LOGGER.error("Unrecoverable PRAW error on r/%s: %s", sub, e)
                break
        
        # Log filtering stats for this subreddit
        sub_posts = len([p for p in posts if p.get("subreddit") == sub])
        if cfg.search_ai and posts_checked > 0:
            LOGGER.info("r/%s: Collected %d posts (checked %d, filtered %d non-AI posts)", 
                       sub, sub_posts, posts_checked, posts_filtered)

    # Deduplicate by id
    seen_posts = set()
    dedup_posts = []
    for p in posts:
        if p["id"] in seen_posts:
            continue
        seen_posts.add(p["id"])
        dedup_posts.append(p)
    seen_comments = set()
    dedup_comments = []
    for c in comments:
        if c["id"] in seen_comments:
            continue
        seen_comments.add(c["id"])
        dedup_comments.append(c)

    return dedup_posts, dedup_comments


def build_dataframe(posts: List[Dict[str, object]], comments: List[Dict[str, object]]) -> pd.DataFrame:
    # Clean and combine
    post_rows = []
    for p in posts:
        combined = " ".join([str(p.get("title", "")), str(p.get("selftext", ""))]).strip()
        content = clean_text(combined)
        if not content:
            continue
        post_rows.append(
            {
                "subreddit": p["subreddit"],
                "type": "post",
                "id": p["id"],
                "parent_id": None,
                "created_utc": p["created_utc"],
                "score": p["score"],
                "num_comments": p["num_comments"],
                "url": p["url"],
                "content": content,
            }
        )
    comment_rows = []
    for c in comments:
        content = clean_text(str(c.get("body", "")))
        if not content:
            continue
        comment_rows.append(
            {
                "subreddit": c["subreddit"],
                "type": "comment",
                "id": c["id"],
                "parent_id": c["parent_id"],
                "created_utc": c["created_utc"],
                "score": c["score"],
                "num_comments": None,
                "url": c["url"],
                "content": content,
            }
        )

    df = pd.DataFrame(post_rows + comment_rows)
    if df.empty:
        return df
    # Order columns
    cols = [
        "subreddit",
        "type",
        "id",
        "parent_id",
        "created_utc",
        "score",
        "num_comments",
        "url",
        "content",
    ]
    return df[cols]


def add_sentiment_columns(df: pd.DataFrame, use_hf: bool) -> pd.DataFrame:
    if df.empty:
        return df
    vader_scores = score_vader_batch(df["content"].tolist())
    df = df.copy()
    df["neg"] = [s.neg for s in vader_scores]
    df["neu"] = [s.neu for s in vader_scores]
    df["pos"] = [s.pos for s in vader_scores]
    df["compound"] = [s.compound for s in vader_scores]
    df["sent_label"] = [s.sent_label for s in vader_scores]

    if use_hf:
        try:
            hf_preds = score_hf_batch(df["content"].tolist())
            df["hf_label"] = [p[0] for p in hf_preds]
            df["hf_score"] = [p[1] for p in hf_preds]
        except Exception as e:
            LOGGER.error("HF sentiment failed: %s", e)
            df["hf_label"] = None
            df["hf_score"] = None
    return df


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    defaults = RuntimeDefaults()
    parser = argparse.ArgumentParser(description="Collect Reddit posts and comments sentiment (non-tech subs)")
    parser.add_argument(
        "--subs",
        type=str,
        default="",
        help="Comma-separated subreddit list to override defaults",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=defaults.mode,
        choices=["new", "top"],
        help="Fetch 'new' or 'top' posts",
    )
    parser.add_argument(
        "--time-filter",
        type=str,
        default=defaults.time_filter,
        choices=["day", "week", "month"],
        help="Time filter for top mode",
    )
    parser.add_argument("--posts", type=int, default=defaults.posts_per_sub, help="Posts per subreddit")
    parser.add_argument("--comments", type=int, default=defaults.top_level_comment_limit, help="Top-level comments per post")
    parser.add_argument("--hf", action="store_true", help="Enable Hugging Face sentiment model as well")
    parser.add_argument("--outdir", type=str, default=defaults.outdir, help="Output directory")
    parser.add_argument("--no-ai-filter", action="store_true", help="Disable AI search filter (collect all posts)")
    parser.add_argument("--dry-run", action="store_true", help="Only parse args and exit (no API calls)")
    return parser.parse_args(argv)


def subs_from_arg(subs_arg: str) -> List[str]:
    subs = [s.strip() for s in subs_arg.split(",") if s.strip()]
    return subs if subs else DEFAULT_SUBREDDITS


def main(argv: Optional[List[str]] = None) -> int:
    _setup_logging()
    load_env()
    args = parse_args(argv)
    subs = subs_from_arg(args.subs)
    cfg = FetchConfig(
        subreddits=subs,
        mode=args.mode,
        time_filter=args.time_filter,
        posts_per_sub=args.posts,
        top_level_comment_limit=args.comments,
        use_hf=bool(args.hf),
        outdir=args.outdir,
        search_ai=not bool(args.no_ai_filter),
        dry_run=bool(args.dry_run),
    )
    ensure_dir(cfg.outdir)
    LOGGER.info(
        "Starting collection: subs=%d mode=%s tf=%s posts=%d comments=%d hf=%s",
        len(cfg.subreddits), cfg.mode, cfg.time_filter, cfg.posts_per_sub, cfg.top_level_comment_limit, cfg.use_hf,
    )

    if cfg.dry_run:
        LOGGER.info("Dry run complete: %s", cfg)
        return 0

    t0 = time.time()
    posts, comments = fetch(cfg)
    df = build_dataframe(posts, comments)
    df = add_sentiment_columns(df, cfg.use_hf)

    base_name = "reddit_ai_nontech"
    outputs = write_outputs(df, cfg.outdir, base_name)
    elapsed = time.time() - t0
    LOGGER.info("Wrote CSV to %s and Parquet to %s (%d rows) in %.1fs", outputs.get("csv"), outputs.get("parquet"), len(df), elapsed)

    append_run_log(
        cfg.outdir,
        {
            "subs_count": len(cfg.subreddits),
            "rows": len(df),
            "mode": cfg.mode,
            "time_filter": cfg.time_filter,
            "use_hf": cfg.use_hf,
            "elapsed_sec": round(elapsed, 2),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


