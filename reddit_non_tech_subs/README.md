## Reddit Non-Tech AI Sentiment Collector

Collect recent posts and top-level comments from curated non-technical subreddits, clean text, and compute sentiment (VADER baseline, optional Hugging Face) for downstream analysis.

### Quickstart

1) Install dependencies:

```bash
pip install -r reddit_non_tech_subs/requirements.txt
```

2) Set environment variables (or copy `.env.example` to `.env` and fill):

```
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=ai-nontech-sentiment/0.1 by u/<yourname>
```

3) Run the collector:

```bash
python -m reddit_non_tech_subs.collector --mode top --time-filter week --posts 100 --comments 80
```

Outputs are written to `reddit_non_tech_subs/data/` as timestamped CSV and Parquet with columns:
`subreddit, type, id, parent_id, created_utc, score, num_comments, url, content, neg, neu, pos, compound, sent_label[, hf_label, hf_score]`.

Enable the Hugging Face model with `--hf` (uses `cardiffnlp/twitter-roberta-base-sentiment-latest`).

### CLI Reference

- `--subs`: comma-separated subreddits to override defaults
- `--mode`: `new|top` (default: `top`)
- `--time-filter`: `day|week|month` (default: `week`, only for `top`)
- `--posts`: posts per subreddit (default: 150)
- `--comments`: top-level comments per post (default: 100)
- `--hf`: enable Hugging Face scoring
- `--outdir`: output dir (default: `reddit_non_tech_subs/data`)
- `--dry-run`: validate args and exit without API calls

### Default Subreddits (editable in `config.py`)

```
["writing", "Screenwriting", "writers", "freelance",
 "artbusiness", "artstore", "artists", "Design",
 "photography", "videography", "Journalism", "blogging",
 "jobs", "recruitinghell", "resumes",
 "teachers", "Professors", "gradadmissions",
 "smallbusiness", "EntrepreneurRideAlong", "marketing",
 "LegalAdviceUK", "PersonalFinance", "acting", "voiceacting",
 "music", "WeAreTheMusicMakers", "books"]
```

Suggested additional candidates you may consider (not included by default): `careerguidance`, `productivity`, `antiwork`, `WorkOnline`, `writingprompts`, `AskHR`, `selfimprovement`, `socialmedia`, `copywriting`, `EtsySellers`.

### Reproducibility & Logs

- INFO-level logging shows progress and timing.
- Each run appends a line to `RUN_LOG.md` with summary stats.

### Notes & Safeguards

- Uses official Reddit API via PRAW. No scraping or Pushshift.
- Do not store usernames or PII; only aggregate text and metadata.
- Respect rate limits; backoff on 429 and transient errors.

### Tests

Run minimal unit tests:

```bash
python -m pytest reddit_non_tech_subs/tests -q
```


