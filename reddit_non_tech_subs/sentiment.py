from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict

import nltk


_VADER_RESOURCE = "vader_lexicon"


def _ensure_vader_downloaded() -> None:
    try:
        nltk.data.find(f"sentiment/{_VADER_RESOURCE}")
    except LookupError:
        nltk.download(_VADER_RESOURCE, quiet=True)


@dataclass
class VaderScores:
    neg: float
    neu: float
    pos: float
    compound: float
    sent_label: str


def _compound_to_label(compound: float) -> str:
    if compound >= 0.05:
        return "pos"
    if compound <= -0.05:
        return "neg"
    return "neu"


def score_vader_batch(texts: Iterable[str]) -> List[VaderScores]:
    """Compute VADER scores for each text."""
    _ensure_vader_downloaded()
    from nltk.sentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    results: List[VaderScores] = []
    for t in texts:
        if not t:
            results.append(VaderScores(neg=0.0, neu=1.0, pos=0.0, compound=0.0, sent_label="neu"))
            continue
        s = sia.polarity_scores(t)
        label = _compound_to_label(s.get("compound", 0.0))
        results.append(
            VaderScores(
                neg=float(s.get("neg", 0.0)),
                neu=float(s.get("neu", 0.0)),
                pos=float(s.get("pos", 0.0)),
                compound=float(s.get("compound", 0.0)),
                sent_label=label,
            )
        )
    return results


def score_hf_batch(texts: List[str], device: Optional[int] = None, batch_size: int = 16) -> List[Tuple[str, float]]:
    """Compute sentiment using cardiffnlp/twitter-roberta-base-sentiment-latest.

    Returns list of (label, score) where label is one of {negative, neutral, positive}.
    """
    from transformers import pipeline

    pipe = pipeline(
        task="sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device if device is not None else -1,
        truncation=True,
    )
    outputs: List[Tuple[str, float]] = []
    # Process in batches for efficiency
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        preds = pipe(batch, truncation=True)
        for p in preds:
            outputs.append((str(p["label"]).lower(), float(p["score"])))
    return outputs


def hf_label_to_vader_like(label: str) -> str:
    label_lower = label.lower()
    if "pos" in label_lower or label_lower == "positive":
        return "pos"
    if "neg" in label_lower or label_lower == "negative":
        return "neg"
    return "neu"


