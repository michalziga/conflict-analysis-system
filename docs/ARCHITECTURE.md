# Architecture

## Overview

```
RSS Feeds (22 sources)
        │
        ▼
  scraper.py  ──── parallel fetch (ThreadPoolExecutor)
        │              │
        │         feedparser → parse entries
        │         newspaper3k → extract full text + NLP
        │         BeautifulSoup → fallback extractor
        │
        ▼
  ArticleRecord  ──── structured dataclass (models.py)
        │              url_hash · title · full_text · summary
        │              matched_keywords · relevance_score
        │              entities · sentiment · embedding_vector  ← empty, filled downstream
        │
        ▼
  writer.py  ──── output layer
        │
        ├── output/sessions/<id>/session_meta.json   ← run summary
        ├── output/sessions/<id>/articles_full.json  ← all records with full text
        ├── output/sessions/<id>/articles.jsonl      ← ML corpus (one record per line)
        └── output/index/master_index.jsonl          ← cumulative, append-only index
                                │
                                ▼
                         analysis.py  ──── phrase frequency analysis
                                │          enrich_article() → per-article analysis{}
                                │          analyze_corpus() → corpus-level aggregation
                                │
                                ▼
                         dashboard.html  ──── standalone HTML dashboard
                                             loads any output file via drag-and-drop
```

## Data Flow

### Scrape phase

1. `ScraperConfig` defines keywords, feeds, and crawl behaviour
2. `scrape()` submits each feed to a `ThreadPoolExecutor` (default 2–4 workers)
3. For each feed, `_parse_single_feed()` iterates entries and filters by keyword match
4. Matching entries are downloaded via `newspaper3k` (primary) or `BeautifulSoup` (fallback)
5. Each article is re-scored against the full text and stored as an `ArticleRecord`
6. Duplicates are caught by SHA-256 hash of the URL, both within and across feeds

### Analysis phase

Two modes:

- **Corpus-level**: `analyze_corpus()` aggregates phrase counts across all articles. Output is a dict with `military_tech`, `political_leaders`, `organizations`, `event_types`, and `top_unigrams`.
- **Per-article**: `enrich_article()` adds an `analysis{}` block directly to each record with per-category phrase counts and a timestamp.

### Prediction slots

`ArticleRecord` reserves fields for downstream models:

| Field | Populated by |
|---|---|
| `entities` | spaCy NER / GLiNER |
| `sentiment` | classifier (cardiffnlp, etc.) |
| `topic_tags` | zero-shot classifier |
| `embedding_vector` | sentence-transformers |
| `prediction` | custom escalation model |

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | All settings. Load from JSON or override programmatically |
| `models.py` | Dataclasses: `ArticleRecord`, `FeedResult`, `ScrapeSession` |
| `scraper.py` | Feed parsing, article extraction, deduplication, scoring |
| `writer.py` | Output: JSON, JSONL, master index |
| `analysis.py` | Phrase counting, corpus aggregation, per-article enrichment |
| `__main__.py` | CLI entry point; also exposes `run()` for notebook use |

## Output Schema

Every `ArticleRecord` carries these fields:

```json
{
  "url_hash":           "sha256 of url — dedup key",
  "url":                "https://...",
  "source_feed":        "https://feeds.bbci.co.uk/news/world/rss.xml",
  "scraped_at":         "2026-03-16T14:22:01Z",
  "title":              "Article title",
  "authors":            ["Author Name"],
  "publish_date":       "2026-03-16T10:00:00",
  "full_text":          "Full article body...",
  "summary":            "newspaper3k NLP summary",
  "top_image":          "https://...",
  "extracted_keywords": ["keyword1", "keyword2"],
  "matched_keywords":   ["Iran", "missile"],
  "relevance_score":    0.74,
  "word_count":         842,
  "language":           "en",
  "entities":           {},
  "sentiment":          {},
  "topic_tags":         [],
  "embedding_model":    "",
  "embedding_vector":   [],
  "prediction":         {}
}
```

After enrichment, each record also carries:

```json
{
  "analysis": {
    "tech_mentions":   { "ballistic missile": 4, "drone attack": 2 },
    "leader_mentions": { "ali khamenei": 3 },
    "org_mentions":    { "irgc": 7, "iaea": 2 },
    "event_mentions":  { "sanctions": 5, "nuclear deal": 3 },
    "analyzed_at":     "2026-03-16T14:25:00Z"
  }
}
```
