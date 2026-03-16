"""
news_scraper/analysis.py
------------------------
Text analysis module: frequency counting, phrase matching, and structured results.
Reads from the scraper's JSONL output.

Usage (CLI):
    python analysis.py
    python analysis.py --input output/index/master_index.jsonl
    python analysis.py --enrich   # also writes per-article enriched JSONL

Usage (Colab / notebook):
    from analysis import load_articles_from_jsonl, analyze_corpus, print_report
    articles = load_articles_from_jsonl(Path("output/index/master_index.jsonl"))
    results  = analyze_corpus(articles)
    print_report(results)
"""
from __future__ import annotations

import json
import re
import logging
import sys
import os
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Stopwords — comprehensive, no NLTK needed
# ─────────────────────────────────────────────

STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "not", "no", "nor", "so", "yet",
    "both", "either", "neither", "than", "that", "this", "these", "those", "it",
    "its", "we", "he", "she", "they", "them", "their", "our", "your", "my", "his",
    "her", "who", "which", "what", "when", "where", "how", "all", "each", "every",
    "any", "some", "such", "into", "through", "about", "between", "after", "before",
    "said", "says", "also", "more", "other", "new", "one", "two", "first", "last",
    "over", "under", "up", "out", "i", "you", "me", "him", "us", "am", "s", "re",
    "just", "been", "like", "very", "even", "well", "back", "still", "way", "get",
    "make", "go", "come", "take", "see", "know", "think", "use", "give", "most",
    "own", "while", "however", "since", "including", "according", "among", "against",
}


# ─────────────────────────────────────────────
#  Keyword taxonomy — multi-word phrases
# ─────────────────────────────────────────────

KEYWORDS_TECH: list[str] = [
    # missiles & rockets
    "ballistic missile", "cruise missile", "hypersonic missile",
    "anti-ship missile", "surface-to-air missile", "rocket launcher",
    # aircraft
    "fighter jet", "stealth aircraft", "stealth bomber",
    "attack helicopter", "armed drone", "combat drone", "uav", "ucav",
    # air defence
    "air defense", "air defence", "missile defense",
    "iron dome", "patriot system", "s-300", "s-400",
    # munitions
    "precision munition", "guided bomb", "bunker buster", "cluster munition",
    # naval
    "aircraft carrier", "naval strike", "submarine",
    # other
    "electronic warfare", "cyber attack", "interceptor",
]

KEYWORDS_LEADERS: list[str] = [
    # Iran
    "ali khamenei", "mojtaba khamenei", "masoud pezeshkian",
    "abbas araghchi", "mohammad pakpour", "esmail qaani",
    # US
    "donald trump", "pete hegseth", "marco rubio",
    "lindsey graham", "mike waltz",
    # Israel
    "benjamin netanyahu", "yoav gallant", "benny gantz",
    # others
    "mark carney", "emmanuel macron", "tamim bin hamad",
    "vladimir putin", "xi jinping", "olaf scholz", "rishi sunak",
]

KEYWORDS_ORGS: list[str] = [
    "irgc", "revolutionary guard", "quds force",
    "hezbollah", "hamas", "islamic jihad",
    "idf", "mossad", "cia", "pentagon",
    "nato", "un security council", "iaea", "centcom",
]

KEYWORDS_EVENTS: list[str] = [
    "airstrike", "air strike", "ceasefire", "cease-fire",
    "nuclear deal", "nuclear talks", "sanctions", "arms deal",
    "peace talks", "military exercise", "naval blockade",
    "ground invasion", "drone attack", "missile strike",
    "assassination", "prisoner exchange",
]


# ─────────────────────────────────────────────
#  I/O helpers
# ─────────────────────────────────────────────

def load_articles_from_jsonl(jsonl_path: Path) -> list[dict]:
    """Load all article records from a JSONL file."""
    articles = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                articles.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed line %d: %s", line_no, e)
    logger.info("Loaded %d articles from %s", len(articles), jsonl_path)
    return articles


def find_latest_jsonl(output_dir: Path = Path("output")) -> Optional[Path]:
    """Auto-discover the most recent JSONL — master index first, then latest session."""
    master = output_dir / "index" / "master_index.jsonl"
    if master.exists():
        return master
    sessions = sorted(
        (output_dir / "sessions").glob("*/articles.jsonl"),
        key=lambda p: p.parent.name,
        reverse=True,
    )
    return sessions[0] if sessions else None


def combine_text(articles: list[dict]) -> str:
    """Concatenate title + full_text from all articles."""
    parts = []
    for art in articles:
        title = art.get("title", "")
        body  = art.get("full_text", "") or art.get("summary", "")
        if title: parts.append(title)
        if body:  parts.append(body)
    return " ".join(parts)


# ─────────────────────────────────────────────
#  Analysis functions
# ─────────────────────────────────────────────

def tokenize(text: str, stopwords: set[str] = STOPWORDS) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords, drop 1-char tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return [t for t in text.split() if t not in stopwords and len(t) > 1]


def count_phrases(text: str, phrases: list[str]) -> list[tuple[str, int]]:
    """
    Case-insensitive substring count for each phrase.
    Returns only phrases with count > 0, sorted descending.
    """
    text_lower = text.lower()
    counts = {
        phrase: text_lower.count(phrase.lower())
        for phrase in phrases
        if text_lower.count(phrase.lower()) > 0
    }
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)


def top_unigrams(text: str, n: int = 30) -> list[tuple[str, int]]:
    """Most frequent single words after stopword removal."""
    return Counter(tokenize(text)).most_common(n)


# ─────────────────────────────────────────────
#  Per-article enrichment — for indexation
# ─────────────────────────────────────────────

def enrich_article(article: dict) -> dict:
    """
    Add an analysis{} block to a single article dict in-place.
    Safe to call on any ArticleRecord dict from the scraper.
    """
    text = (article.get("full_text", "") or "") + " " + (article.get("summary", "") or "")
    article["analysis"] = {
        "tech_mentions":   dict(count_phrases(text, KEYWORDS_TECH)),
        "leader_mentions": dict(count_phrases(text, KEYWORDS_LEADERS)),
        "org_mentions":    dict(count_phrases(text, KEYWORDS_ORGS)),
        "event_mentions":  dict(count_phrases(text, KEYWORDS_EVENTS)),
        "analyzed_at":     datetime.now(timezone.utc).isoformat(),
    }
    return article


# ─────────────────────────────────────────────
#  Corpus-level analysis
# ─────────────────────────────────────────────

def analyze_corpus(articles: list[dict]) -> dict:
    """
    Full frequency analysis over all articles.
    Returns a structured dict suitable for JSON serialisation or DataFrame loading.
    """
    text = combine_text(articles)
    return {
        "corpus_stats": {
            "total_articles": len(articles),
            "total_words":    len(text.split()),
            "analyzed_at":    datetime.now(timezone.utc).isoformat(),
        },
        "military_tech":     count_phrases(text, KEYWORDS_TECH)[:20],
        "political_leaders": count_phrases(text, KEYWORDS_LEADERS)[:20],
        "organizations":     count_phrases(text, KEYWORDS_ORGS)[:20],
        "event_types":       count_phrases(text, KEYWORDS_EVENTS)[:20],
        "top_unigrams":      top_unigrams(text, n=30),
    }


# ─────────────────────────────────────────────
#  Output
# ─────────────────────────────────────────────

def save_analysis(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Analysis saved → %s", output_path)


def print_report(results: dict) -> None:
    cs = results["corpus_stats"]
    print(f"\n{'='*55}")
    print(f"  CORPUS: {cs['total_articles']} articles  |  {cs['total_words']:,} words")
    print(f"{'='*55}")
    sections = [
        ("Military technology",   "military_tech"),
        ("Political leaders",     "political_leaders"),
        ("Organizations",         "organizations"),
        ("Event types",           "event_types"),
    ]
    for title, key in sections:
        items = results.get(key, [])
        if not items:
            continue
        print(f"\n  {title}")
        print(f"  {'-'*40}")
        for phrase, count in items[:15]:
            bar = "█" * min(count, 30)
            print(f"  {phrase:<30} {count:>5}  {bar}")
    print(f"\n  Top words (after stopwords)")
    print(f"  {'-'*40}")
    for word, count in results.get("top_unigrams", [])[:20]:
        bar = "█" * min(count // 5, 30)
        print(f"  {word:<30} {count:>5}  {bar}")
    print()


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Analyse scraped news corpus.")
    parser.add_argument("--input",  type=str, help="Path to JSONL (auto-detected if omitted).")
    parser.add_argument("--output", type=str, default="output/analysis", help="Output dir.")
    parser.add_argument("--enrich", action="store_true",
                        help="Write per-article enriched JSONL.")
    # parse_known_args so this is safe to import in Colab
    args, _ = parser.parse_known_args()

    jsonl_path = Path(args.input) if args.input else find_latest_jsonl()
    if not jsonl_path:
        logger.error("No JSONL file found. Run the scraper first, or pass --input <path>.")
        return

    logger.info("Reading from %s", jsonl_path)
    articles = load_articles_from_jsonl(jsonl_path)
    if not articles:
        logger.error("No articles loaded — nothing to analyse.")
        return

    results = analyze_corpus(articles)
    print_report(results)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir   = Path(args.output)
    save_analysis(results, out_dir / f"corpus_analysis_{timestamp}.json")

    if args.enrich:
        enriched_path = out_dir / f"enriched_{timestamp}.jsonl"
        enriched_path.parent.mkdir(parents=True, exist_ok=True)
        with open(enriched_path, "w", encoding="utf-8") as f:
            for art in articles:
                f.write(json.dumps(enrich_article(art), ensure_ascii=False) + "\n")
        logger.info("Enriched JSONL saved → %s", enriched_path)


if __name__ == "__main__":
    main()
