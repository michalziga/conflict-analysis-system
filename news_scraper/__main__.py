"""
news_scraper/__main__.py
------------------------
CLI entry point — also importable as a function for Colab / notebook use.

Usage (terminal):
    python -m news_scraper
    python -m news_scraper --config my_config.json
    python -m news_scraper --keywords war Iran --feeds https://...

Usage (Colab / notebook):
    from news_scraper.__main__ import run
    run()
"""
from __future__ import annotations

import argparse
import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ScraperConfig
from scraper import scrape
from writer import write_outputs


# ─────────────────────────────────────────────
#  Logging setup
# ─────────────────────────────────────────────

def setup_logging(verbose: bool = False, log_dir: str = ".") -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    log_path = Path(log_dir) / "scraper.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # avoid duplicate handlers when re-run in a notebook
    root = logging.getLogger()
    if root.handlers:
        root.handlers.clear()
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_path), encoding="utf-8"),
        ],
    )


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Structured news scraper for indexation and prediction pipelines."
    )
    p.add_argument("--config",    type=str, help="Path to JSON config file.")
    p.add_argument("--keywords",  nargs="+", help="Override keywords list.")
    p.add_argument("--feeds",     nargs="+", help="Override RSS feed URLs.")
    p.add_argument("--output",    type=str,  default="output", help="Output directory.")
    p.add_argument("--workers",   type=int,  default=2,        help="Parallel feed workers.")
    p.add_argument("--verbose",   action="store_true",         help="Debug logging.")
    p.add_argument("--dry-run",   action="store_true",         help="Parse feeds but don't save.")
    return p


# ─────────────────────────────────────────────
#  Entry point — callable from CLI or notebook
# ─────────────────────────────────────────────

def run(
    config:   str  = None,
    keywords: list = None,
    feeds:    list = None,
    output:   str  = "output",
    workers:  int  = 2,
    verbose:  bool = False,
    dry_run:  bool = False,
) -> None:
    """
    Programmatic entry point — use this in Colab cells instead of CLI.
    All parameters mirror the CLI flags.
    """
    setup_logging(verbose, log_dir=output)
    log = logging.getLogger("main")

    # ── Load config ───────────────────────────────────────────────
    if config:
        cfg = ScraperConfig.from_json(config)
        log.info("Loaded config from %s", config)
    else:
        cfg = ScraperConfig()

    # ── Override with explicit args ───────────────────────────────
    if keywords:
        cfg.keywords = keywords
    if feeds:
        cfg.feeds = feeds
    cfg.output_dir  = output
    cfg.max_workers = workers

    log.info("=" * 60)
    log.info("NEWS SCRAPER starting")
    log.info("Keywords : %s", ", ".join(cfg.keywords))
    log.info("Feeds    : %d configured", len(cfg.feeds))
    log.info("Workers  : %d", cfg.max_workers)
    log.info("=" * 60)

    session = scrape(cfg)

    summary = session.summary()
    log.info("")
    log.info("=" * 60)
    log.info("SCRAPE COMPLETE — %d articles found", summary["total_articles"])
    for feed_url, count in summary["articles_by_feed"].items():
        log.info("  %-55s %d", feed_url, count)
    if summary["errors"]:
        log.warning("Feed errors:")
        for err in summary["errors"]:
            log.warning("  %s → %s", err["feed"], err["error"])
    log.info("")
    log.info("Top articles by relevance:")
    for item in summary["top_articles"]:
        log.info("  [%.2f] %s", item["score"], item["title"])
    log.info("=" * 60)

    if not dry_run:
        written = write_outputs(session, cfg)
        log.info("")
        log.info("Output files:")
        for label, path in written.items():
            log.info("  %-10s → %s", label, path)
    else:
        log.info("Dry run — no files written.")

    return session


def main() -> None:
    # parse_known_args ignores Colab's extra argv flags (--ip, --stdin, etc.)
    args, _ = build_parser().parse_known_args()
    run(
        config   = args.config,
        keywords = args.keywords,
        feeds    = args.feeds,
        output   = args.output,
        workers  = args.workers,
        verbose  = args.verbose,
        dry_run  = args.dry_run,
    )


if __name__ == "__main__":
    main()
