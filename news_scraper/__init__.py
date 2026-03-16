"""
news_scraper — Geopolitical conflict news scraper and analysis pipeline.

Usage:
    python -m news_scraper                     # run with default config
    python -m news_scraper --config cfg.json   # custom config
    python -m news_scraper --dry-run           # parse feeds, no output saved

Programmatic:
    from news_scraper.config   import ScraperConfig
    from news_scraper.scraper  import scrape
    from news_scraper.writer   import write_outputs
    from news_scraper.analysis import analyze_corpus
"""

__version__ = "1.0.0"
__author__  = "Conflict Analysis System"
