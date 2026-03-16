# Conflict Analysis System (CAS)

A production-grade geopolitical news scraper, analysis pipeline, and intelligence dashboard — built for indexation and escalation prediction.

![CI](https://github.com/michalziga/conflict-analysis-system)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What it does

- Scrapes **22 open-access RSS feeds** — think-tanks (RAND, CSIS, Carnegie, Brookings, Chatham House, IISS), wire services (Reuters, AP, BBC), regional outlets (Al Jazeera, France 24, DW), and specialist defence sources (ISW, War on the Rocks, DoD)
- Extracts full article text with `newspaper3k` + BeautifulSoup fallback
- Scores and deduplicates articles by keyword relevance
- Outputs structured **JSONL** designed to feed directly into search indexes and ML models
- Runs **phrase-frequency analysis** across four taxonomies: military technology, political leaders, organizations, and event types
- Ships a **standalone HTML dashboard** with a D3 world map, escalation index, frequency charts, and live intel feed — no server required

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/conflict-analysis-system.git
cd conflict-analysis-system

# 2. Install
pip install -r requirements.txt
# Or: pip install -e ".[nlp]"  to include spaCy, sentence-transformers

# 3. Run
python -m news_scraper                           # default config
python -m news_scraper --config config.json      # custom config
python -m news_scraper --dry-run                 # parse feeds, no output saved
python -m news_scraper --keywords war Iran IRGC  # override keywords
```

### Colab

Open `news_scraper_colab.ipynb` directly in Google Colab — all source files are written inline, no upload needed.

---

## Repository layout

```
conflict-analysis-system/
│
├── news_scraper/               ← Python package
│   ├── __init__.py
│   ├── __main__.py             ← CLI entry point + run() for notebooks
│   ├── config.py               ← ScraperConfig dataclass
│   ├── models.py               ← ArticleRecord, FeedResult, ScrapeSession
│   ├── scraper.py              ← Feed parsing, extraction, deduplication
│   ├── writer.py               ← JSON / JSONL / master index output
│   └── analysis.py             ← Phrase analysis, corpus aggregation, enrichment
│
├── dashboard/
│   └── dashboard.html          ← Standalone intelligence dashboard (no server)
│
├── tests/
│   └── test_pipeline.py        ← Unit tests (pytest)
│
├── docs/
│   ├── ARCHITECTURE.md         ← Data flow and module responsibilities
│   └── PIPELINE.md             ← Downstream NLP, embeddings, Elasticsearch, Claude API
│
├── .github/
│   └── workflows/
│       ├── ci.yml              ← Lint + test on push/PR
│       └── scheduled_scrape.yml ← Scrape every 6h, upload artifact
│
├── news_scraper_colab.ipynb    ← Colab trial run notebook
├── config.example.json         ← Template — copy to config.json and edit
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Configuration

Copy `config.example.json` to `config.json` and edit:

```json
{
  "keywords": ["war", "Iran", "IRGC", "missile", "sanctions"],
  "feeds":    ["https://feeds.bbci.co.uk/news/world/rss.xml"],
  "max_entries_per_feed":   50,
  "delay_between_requests": 2.0,
  "max_workers":            4,
  "output_dir":             "output",
  "min_word_count":         80
}
```

All settings can also be passed as CLI flags:

```bash
python -m news_scraper \
  --keywords war Iran IRGC \
  --output /data/output \
  --workers 4 \
  --verbose
```

---

## Output

```
output/
  sessions/
    20260316_142301/
      session_meta.json     ← run summary + config snapshot
      articles_full.json    ← all ArticleRecord objects with full text
      articles.jsonl        ← ML corpus — one JSON record per line
  index/
    master_index.jsonl      ← cumulative, append-only across all runs
  analysis/
    corpus_analysis_*.json  ← frequency analysis results
    enriched_*.jsonl        ← articles with analysis{} blocks added
```

### ArticleRecord fields

| Field | Description |
|---|---|
| `url_hash` | SHA-256 of URL — use as document ID |
| `title`, `full_text`, `summary` | Content |
| `matched_keywords` | Which config keywords triggered this article |
| `relevance_score` | 0–1 float — title hits weighted 2× |
| `word_count`, `language` | Quality / filtering features |
| `entities` | Empty at scrape time — populate with spaCy |
| `sentiment` | Empty — populate with a classifier |
| `embedding_vector` | Empty — populate with sentence-transformers |
| `prediction` | Empty — populate with your model |

---

## Analysis

```bash
# Corpus-level analysis (reads master_index.jsonl automatically)
python -m news_scraper.analysis

# Per-article enrichment — adds analysis{} block to each record
python -m news_scraper.analysis --enrich

# Specify a file explicitly
python -m news_scraper.analysis --input output/index/master_index.jsonl
```

Or use the API:

```python
from news_scraper.analysis import load_articles_from_jsonl, analyze_corpus, print_report

articles = load_articles_from_jsonl("output/index/master_index.jsonl")
results  = analyze_corpus(articles)
print_report(results)
```

---

## Dashboard

Open `dashboard/dashboard.html` in any browser — no server, no install.

- **World map** — D3 + TopoJSON with real country geometry. Conflict actors colour-coded by threat level. Animated great-circle arcs show relationships between Iran, Israel, Hezbollah, Houthis, US, Russia, China. Hover any country for escalation score and signal tags.
- **Middle East zoom** — toggle between world and theatre view
- **Escalation Prediction Index** — per-actor score cards (0–100) with delta, bar, and signal tags
- **Frequency tabs** — Military Tech / Leaders / Organizations / Events
- **Intel stream** — recent articles with relevance scores and keyword tags
- **File loading** — drag-and-drop any output file to replace demo data:
  - `master_index.jsonl` — full article corpus
  - `enriched_*.jsonl` — with per-article analysis blocks
  - `articles_full.json` — session articles
  - `corpus_analysis_*.json` — pre-aggregated analysis

---

## Downstream pipeline

See [`docs/PIPELINE.md`](docs/PIPELINE.md) for ready-to-run code for:

- spaCy NER / GLiNER entity extraction
- sentence-transformers embeddings
- Elasticsearch bulk indexing
- Sentiment classification (cardiffnlp)
- Escalation score baseline model
- Claude API batch summarisation (~$0.003/article on Sonnet 4.6 with Batch API)

---

## Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=news_scraper
```

The test suite covers: config round-trips, `score_article` edge cases, tokenisation, phrase counting, corpus analysis, per-article enrichment, and filesystem output.

---

## GitHub Actions

- **CI** (`ci.yml`) — linting (ruff) + full test suite on Python 3.9–3.12, triggered on push and PR
- **Scheduled scrape** (`scheduled_scrape.yml`) — runs every 6 hours on `main`, uploads output as a GitHub Actions artifact (retained 30 days)

To enable scheduled scraping: push to `main` and the workflow activates automatically.

---

## Feed sources

| Category | Sources |
|---|---|
| US think-tanks | CSIS, RAND (Commentary + Reports), Carnegie Endowment, Brookings |
| UK/EU think-tanks | Chatham House, IISS |
| Foreign policy journals | Foreign Affairs (CFR), Foreign Policy, War on the Rocks, ISW |
| Wire / broadcast | BBC World, Reuters, AP, The Guardian |
| Regional | Al Jazeera, Times of India, France 24, Deutsche Welle |
| Defence specialist | Spacewar, US DoD News |

All feeds are open-access. No paywalled sources (FT, Economist, Bloomberg, NYT, WaPo removed).

---

## License

MIT — see [LICENSE](LICENSE).
