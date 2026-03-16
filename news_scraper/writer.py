"""
news_scraper/writer.py
----------------------
Handles all output: structured JSON, JSONL corpus, and a lightweight index.
"""
from __future__ import annotations

import json
import logging
import sys
import os
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ScraperConfig
from models import ScrapeSession

logger = logging.getLogger(__name__)


def write_outputs(session: ScrapeSession, cfg: ScraperConfig) -> dict[str, Path]:
    """
    Writes all configured output files and returns a dict of {type: path}.

    Output structure:
        output/
          sessions/
            <session_id>/
              session_meta.json      ← summary + config snapshot
              articles_full.json     ← all ArticleRecord dicts (with full text)
              articles.jsonl         ← one JSON line per article (ML-friendly)
          index/
              master_index.jsonl     ← cumulative lightweight index (appended)
    """
    out_root = Path(cfg.output_dir)
    session_dir = out_root / "sessions" / session.session_id
    index_dir   = out_root / "index"

    session_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    # ── Filter stubs ──────────────────────────────────────────────
    articles = [
        a for a in session.articles
        if a.word_count >= cfg.min_word_count
    ]
    logger.info("Writing %d articles (filtered from %d)", len(articles), len(session.articles))

    # ── session_meta.json ─────────────────────────────────────────
    meta_path = session_dir / "session_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(session.summary(), f, indent=2, ensure_ascii=False, default=str)
    written["meta"] = meta_path
    logger.info("Saved session meta → %s", meta_path)

    # ── articles_full.json ────────────────────────────────────────
    if cfg.save_json:
        full_path = session_dir / "articles_full.json"
        payload = {
            "session_id": session.session_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total": len(articles),
            "articles": [a.to_dict() for a in articles],
        }
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        written["json"] = full_path
        logger.info("Saved full JSON → %s", full_path)

    # ── articles.jsonl ────────────────────────────────────────────
    if cfg.save_jsonl:
        jsonl_path = session_dir / "articles.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for a in articles:
                f.write(a.to_jsonl_line() + "\n")
        written["jsonl"] = jsonl_path
        logger.info("Saved JSONL corpus → %s", jsonl_path)

    # ── master_index.jsonl (append) ───────────────────────────────
    if cfg.save_index:
        index_path = index_dir / "master_index.jsonl"
        with open(index_path, "a", encoding="utf-8") as f:
            for a in articles:
                f.write(json.dumps(a.to_index_record(), ensure_ascii=False, default=str) + "\n")
        written["index"] = index_path
        logger.info("Appended to master index → %s", index_path)

    return written
