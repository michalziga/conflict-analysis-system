"""
news_scraper/models.py
----------------------
Pure-Python dataclasses that form the schema for every article and session.
Designed to map 1-to-1 with a search index (Elasticsearch, Typesense, etc.)
or a vector store (Pinecone, Weaviate, Chroma).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─────────────────────────────────────────────
#  Core article record — the indexation unit
# ─────────────────────────────────────────────

@dataclass
class ArticleRecord:
    # ── Identity ──────────────────────────────────────────────────
    url_hash:   str = ""          # SHA-256 of URL — dedup key
    url:        str = ""
    source_feed: str = ""
    scraped_at: str = ""          # ISO-8601 UTC

    # ── Content ───────────────────────────────────────────────────
    title:      str = ""
    authors:    list[str] = field(default_factory=list)
    publish_date: Optional[str] = None   # ISO-8601 or None
    full_text:  str = ""
    summary:    str = ""          # newspaper3k NLP summary
    top_image:  str = ""

    # ── NLP / ML features ─────────────────────────────────────────
    extracted_keywords: list[str] = field(default_factory=list)   # newspaper3k
    matched_keywords:   list[str] = field(default_factory=list)   # our keyword set
    relevance_score:    float = 0.0    # 0–1, higher = more keyword hits
    word_count:         int   = 0
    language:           str   = "unknown"

    # ── Prediction placeholders ───────────────────────────────────
    # These fields are intentionally empty at scrape time — downstream
    # NLP / ML models should populate them.
    entities:           dict  = field(default_factory=dict)
    # e.g. {"countries": ["Iran", "Israel"], "organizations": ["IRGC"]}

    sentiment:          dict  = field(default_factory=dict)
    # e.g. {"label": "negative", "score": 0.82}

    topic_tags:         list[str] = field(default_factory=list)
    # e.g. ["military-conflict", "diplomacy", "sanctions"]

    embedding_model:    str   = ""    # name of model used to embed
    embedding_vector:   list  = field(default_factory=list)  # float32[]

    prediction:         dict  = field(default_factory=dict)
    # e.g. {"escalation_risk": 0.74, "conflict_class": "armed-conflict"}

    def to_dict(self) -> dict:
        return asdict(self)

    def to_index_record(self) -> dict:
        """Lightweight dict for search index (drops embedding vector)."""
        d = self.to_dict()
        d.pop("embedding_vector", None)
        d.pop("full_text", None)
        return d

    def to_jsonl_line(self) -> str:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ─────────────────────────────────────────────
#  Per-feed result wrapper
# ─────────────────────────────────────────────

@dataclass
class FeedResult:
    feed_url: str = ""
    articles: list[ArticleRecord] = field(default_factory=list)
    error:    Optional[str] = None

    @property
    def count(self) -> int:
        return len(self.articles)


# ─────────────────────────────────────────────
#  Session — one full scrape run
# ─────────────────────────────────────────────

@dataclass
class ScrapeSession:
    session_id:      str = ""
    started_at:      str = ""
    finished_at:     str = ""
    total_scraped:   int = 0
    config_snapshot: dict = field(default_factory=dict)

    articles:        list[ArticleRecord] = field(default_factory=list)
    feed_results:    list[FeedResult]    = field(default_factory=list)

    def summary(self) -> dict:
        by_feed = {fr.feed_url: fr.count for fr in self.feed_results}
        top = sorted(self.articles, key=lambda a: a.relevance_score, reverse=True)[:5]
        return {
            "session_id":    self.session_id,
            "started_at":    self.started_at,
            "finished_at":   self.finished_at,
            "total_articles": self.total_scraped,
            "articles_by_feed": by_feed,
            "top_articles":  [
                {"title": a.title, "score": a.relevance_score, "url": a.url}
                for a in top
            ],
            "errors": [
                {"feed": fr.feed_url, "error": fr.error}
                for fr in self.feed_results if fr.error
            ],
        }
