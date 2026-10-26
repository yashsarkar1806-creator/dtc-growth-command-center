"""Review intelligence: sentiment (Hugging Face model with offline fallback) + theme tagging."""
from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

HF_SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

THEMES = {
    "Efficacy & skin feel": ["skin", "hydrated", "soft", "effective", "difference", "gentle", "broke me out"],
    "Shipping & delivery": ["shipping", "delivery", "arrived", "package", "damaged", "leaked"],
    "Price & value": ["expensive", "value", "price", "cheap", "small bottle"],
    "Packaging & product quality": ["pump", "packaging", "bottle", "leaked", "stopped working"],
    "Customer service": ["support", "service", "emails", "answered", "cancel"],
    "Scent & texture": ["scent", "smells", "absorbed", "texture", "sticky"],
    "Subscription": ["subscription", "cancel", "reorder"],
}

_POS_WORDS = {"love", "amazing", "best", "great", "beautiful", "favourite", "favorite", "gentle",
              "effective", "soft", "hydrated", "fast", "reorder", "quickly"}
_NEG_WORDS = {"disappointed", "broke", "expensive", "stopped", "damaged", "leaked", "frustrating",
              "hard", "nobody", "weeks", "strong", "small"}


def lexicon_sentiment(text: str) -> tuple[str, float]:
    """Tiny rule-based fallback. Returns (label, score in [-1, 1])."""
    toks = re.findall(r"[a-z']+", text.lower())
    pos = sum(t in _POS_WORDS for t in toks)
    neg = sum(t in _NEG_WORDS for t in toks)
    if pos == neg:
        return "NEUTRAL", 0.0
    score = (pos - neg) / (pos + neg)
    return ("POSITIVE" if score > 0 else "NEGATIVE"), float(score)


@lru_cache(maxsize=1)
def _load_pipeline(model_name: str = HF_SENTIMENT_MODEL):
    from transformers import pipeline  # imported lazily so the app works without transformers

    return pipeline("sentiment-analysis", model=model_name)


def score_reviews(texts: list[str], use_model: bool = True, neutral_band: float = 0.75) -> tuple[pd.DataFrame, str]:
    """Score texts. Returns (frame with label/score, engine-used string).

    With the HF model, predictions whose confidence < neutral_band are labelled NEUTRAL
    (SST-2 is binary, so this gives a pragmatic 3-way view).
    """
    if use_model:
        try:
            clf = _load_pipeline()
            preds = clf(list(texts), truncation=True, batch_size=32)
            labels, scores = [], []
            for p in preds:
                signed = p["score"] if p["label"].upper().startswith("POS") else -p["score"]
                labels.append("NEUTRAL" if p["score"] < neutral_band else p["label"].upper())
                scores.append(float(signed))
            return pd.DataFrame({"label": labels, "score": scores}), f"Hugging Face: {HF_SENTIMENT_MODEL}"
        except Exception as exc:  # no internet, no torch, etc.
            engine = f"Lexicon fallback (model unavailable: {type(exc).__name__})"
    else:
        engine = "Lexicon fallback (model disabled)"
    res = [lexicon_sentiment(t) for t in texts]
    return pd.DataFrame(res, columns=["label", "score"]), engine


def tag_themes(text: str) -> list[str]:
    t = text.lower()
    found = [name for name, kws in THEMES.items() if any(k in t for k in kws)]
    return found or ["Other"]


def analyse(reviews: pd.DataFrame, use_model: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return (per-review frame, theme summary, engine)."""
    scored, engine = score_reviews(reviews["text"].tolist(), use_model=use_model)
    df = reviews.reset_index(drop=True).join(scored)
    df["themes"] = df["text"].map(tag_themes)
    ex = df.explode("themes")
    summary = ex.groupby("themes").agg(
        mentions=("review_id", "count"),
        negative_share=("label", lambda s: (s == "NEGATIVE").mean()),
        avg_sentiment=("score", "mean"),
        avg_stars=("stars", "mean"),
    )
    # Priority = volume x negativity: where fixing CX moves the needle most.
    summary["priority"] = summary["mentions"] * summary["negative_share"]
    return df, summary.sort_values("priority", ascending=False).round(3), engine
