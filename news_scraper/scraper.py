"""
news_scraper/scraper.py
-----------------------
Enhanced news scraper with structured output for indexation and prediction pipelines.
"""
from __future__ import annotations

import feedparser
import requests
from newspaper import Article
import time
import json
import hashlib
import logging
import re
import sys
import os
import nltk
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup

# ── make sibling modules importable whether run as script or notebook ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ScraperConfig
from models import ArticleRecord, FeedResult, ScrapeSession

# ── nltk data required by newspaper3k .nlp() ──────────────────────────
def _ensure_nltk_data() -> None:
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Keyword matching
# ─────────────────────────────────────────────

def score_article(text: str, title: str, keywords: list[str]) -> tuple[bool, list[str], float]:
    """
    Returns (matches, matched_keywords, relevance_score).
    Score = weighted hit count / total keywords (title hits worth 2x).
    """
    matched = set()
    title_lower = (title or "").lower()
    text_lower  = (text  or "").lower()

    for kw in keywords:
        kw_lower = kw.lower()
        in_title = kw_lower in title_lower
        in_text  = kw_lower in text_lower
        if in_title or in_text:
            matched.add(kw)

    if not matched:
        return False, [], 0.0

    title_hits = sum(1 for kw in matched if kw.lower() in title_lower)
    text_hits  = len(matched) - title_hits
    raw_score  = (title_hits * 2 + text_hits) / (len(keywords) * 2)
    score      = round(min(raw_score, 1.0), 4)
    return True, sorted(matched), score


# ─────────────────────────────────────────────
#  Article extraction
# ─────────────────────────────────────────────

def _extract_with_newspaper(url: str, timeout: int) -> Optional[dict]:
    _ensure_nltk_data()
    art = Article(url, request_timeout=timeout)
    art.download()
    art.parse()
    art.nlp()
    return {
        "title":    art.title,
        "text":     art.text,
        "authors":  art.authors,
        "summary":  art.summary,
        "keywords": art.keywords,
        "top_image": art.top_image,
        "publish_date": art.publish_date,
    }


def _extract_with_bs4(url: str, timeout: int) -> Optional[dict]:
    """Fallback extractor using requests + BeautifulSoup."""
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title = soup.find("title")
    title_text = title.get_text(strip=True) if title else ""

    # grab <p> tags as body text
    paragraphs = soup.find_all("p")
    body = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
    body = re.sub(r"\s{2,}", " ", body).strip()

    return {
        "title":       title_text,
        "text":        body,
        "authors":     [],
        "summary":     body[:500] if body else "",
        "keywords":    [],
        "top_image":   "",
        "publish_date": None,
    }


def get_article_text(url: str, timeout: int = 15) -> Optional[dict]:
    for extractor in (_extract_with_newspaper, _extract_with_bs4):
        try:
            data = extractor(url, timeout)
            if data and data.get("text"):
                return data
        except Exception as e:
            logger.debug("Extractor %s failed for %s: %s", extractor.__name__, url, e)
    logger.warning("All extractors failed for %s", url)
    return None


# ─────────────────────────────────────────────
#  Feed parsing
# ─────────────────────────────────────────────

def _parse_single_feed(feed_url: str, cfg: ScraperConfig) -> FeedResult:
    result = FeedResult(feed_url=feed_url)
    seen_in_feed: set[str] = set()

    try:
        feed = feedparser.parse(feed_url)
        entries = feed.entries[: cfg.max_entries_per_feed]
        logger.info("Feed %s — %d entries", feed_url, len(entries))

        for entry in entries:
            title   = entry.get("title", "")
            link    = entry.get("link", "")
            summary = entry.get("summary", "")

            if not link:
                continue

            url_hash = hashlib.sha256(link.encode()).hexdigest()
            if url_hash in seen_in_feed:
                continue
            seen_in_feed.add(url_hash)

            matches, matched_kws, score = score_article(
                summary, title, cfg.keywords
            )
            if not matches:
                continue

            logger.info("  Matched: %s (score=%.2f)", title[:80], score)
            article_data = get_article_text(link, timeout=cfg.request_timeout)

            if article_data:
                # re-score with full text
                _, matched_kws, score = score_article(
                    article_data["text"], title, cfg.keywords
                )

                pub_date = article_data.get("publish_date")
                if isinstance(pub_date, datetime):
                    pub_date = pub_date.isoformat()

                record = ArticleRecord(
                    url_hash        = url_hash,
                    url             = link,
                    source_feed     = feed_url,
                    scraped_at      = datetime.now(timezone.utc).isoformat(),
                    title           = article_data["title"] or title,
                    authors         = article_data["authors"],
                    publish_date    = pub_date,
                    full_text       = article_data["text"],
                    summary         = article_data["summary"],
                    extracted_keywords = article_data["keywords"],
                    top_image       = article_data.get("top_image", ""),
                    matched_keywords = matched_kws,
                    relevance_score = score,
                    word_count      = len(article_data["text"].split()),
                    language        = _detect_language(article_data["text"]),
                )
                result.articles.append(record)
                time.sleep(cfg.delay_between_requests)

    except Exception as e:
        logger.error("Feed parse error [%s]: %s", feed_url, e)
        result.error = str(e)

    return result


def _detect_language(text: str) -> str:
    """Naive language tag — extend with langdetect if needed."""
    try:
        from langdetect import detect
        return detect(text[:500])
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────
#  Main scrape orchestrator
# ─────────────────────────────────────────────

def scrape(cfg: ScraperConfig) -> ScrapeSession:
    session = ScrapeSession(
        session_id  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        started_at  = datetime.now(timezone.utc).isoformat(),
        config_snapshot = cfg.to_dict(),
    )

    global_seen: set[str] = set()

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = {
            pool.submit(_parse_single_feed, url, cfg): url
            for url in cfg.feeds
        }
        for future in as_completed(futures):
            feed_result = future.result()
            session.feed_results.append(feed_result)

            for article in feed_result.articles:
                if article.url_hash not in global_seen:
                    global_seen.add(article.url_hash)
                    session.articles.append(article)
                else:
                    logger.debug("Duplicate skipped: %s", article.url)

    session.finished_at   = datetime.now(timezone.utc).isoformat()
    session.total_scraped = len(session.articles)
    return session
