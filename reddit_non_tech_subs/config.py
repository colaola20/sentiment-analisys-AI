"""Configuration for subreddit sentiment collection.

Edit `DEFAULT_SUBREDDITS` to curate your list. Keep it non-technical focused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


DEFAULT_SUBREDDITS: List[str] = [
    "writing", "Screenwriting", "writers", "freelance",
    "artbusiness", "artstore", "artists", "Design",
    "photography", "videography", "Journalism", "blogging",
    "jobs", "recruitinghell", "resumes",
    "teachers", "Professors", "gradadmissions",
    "smallbusiness", "EntrepreneurRideAlong", "marketing",
    "LegalAdviceUK", "PersonalFinance", "acting", "voiceacting",
    "music", "WeAreTheMusicMakers", "books",
]


@dataclass(frozen=True)
class RuntimeDefaults:
    mode: str = "top"  # {"new","top"}
    time_filter: str = "week"  # {"day","week","month"} when mode == "top"
    posts_per_sub: int = 150
    top_level_comment_limit: int = 100
    use_hf: bool = False
    outdir: str = "reddit_non_tech_subs/data"


