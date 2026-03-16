"""
news_scraper/config.py
----------------------
All scraper settings live here — load from JSON/YAML or override programmatically.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ScraperConfig:
    # ── What to look for ──────────────────────────────────────────
    keywords: list[str] = field(default_factory=lambda: [
        "war", "Iran", "IRGC", "US", "Israel",
        "missile", "sanctions", "nuclear", "ceasefire", "airstrike",
    ])

    # ── Where to look ─────────────────────────────────────────────
    feeds: list[str] = field(default_factory=lambda: [

        # ── US think-tanks & policy institutes (all open access) ──
        "https://www.csis.org/rss.xml",                          # CSIS
        "https://www.rand.org/pubs/commentary.xml",              # RAND Commentary
        "https://www.rand.org/pubs/research_reports.xml",        # RAND Research Reports
        "https://carnegieendowment.org/rss",                     # Carnegie Endowment
        "https://www.brookings.edu/feed/?post_type=research",    # Brookings Institution

        # ── UK / European think-tanks ─────────────────────────────
        "https://www.chathamhouse.org/rss.xml",                  # Chatham House (RIIA)
        "https://www.iiss.org/rss",                              # IISS — military & strategic affairs

        # ── Foreign policy & security journals ────────────────────
        "https://foreignaffairs.com/rss.xml",                    # Foreign Affairs (CFR) — free articles
        "https://foreignpolicy.com/feed",                        # Foreign Policy magazine
        "https://warontherocks.com/feed",                        # War on the Rocks — defence analysis
        "https://iswresearch.blogspot.com/feeds/posts/default",  # ISW — Institute for the Study of War

        # ── Anglophone wire / broadcast ───────────────────────────
        "https://feeds.bbci.co.uk/news/world/rss.xml",          # BBC World
        "https://feeds.reuters.com/reuters/worldNews",           # Reuters World
        "https://apnews.com/rss/apf-intlnews",                   # Associated Press
        "https://www.theguardian.com/world/rss",                 # The Guardian

        # ── Regional / Global South ───────────────────────────────
        "https://www.aljazeera.com/xml/rss/all.xml",             # Al Jazeera
        "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms", # Times of India
        "https://www.france24.com/en/rss",                       # France 24 (EN)
        "https://feeds.dw.com/rw/rss/rss.xml",                   # Deutsche Welle

        # ── Defence / security specialist ─────────────────────────
        "https://www.spacewar.com/Military_Technology.xml",      # Spacewar — defence tech
        "https://www.defense.gov/DesktopModules/ArticleCS/RSS.ashx?ContentType=1&Site=945&max=10",  # US DoD News
    ])

    # ── Crawl behaviour ───────────────────────────────────────────
    max_entries_per_feed:    int   = 50
    delay_between_requests:  float = 2.0   # seconds, per-article within a feed
    request_timeout:         int   = 15    # seconds
    max_workers:             int   = 4     # parallel feed fetchers

    # ── Output ────────────────────────────────────────────────────
    output_dir:     str = "output"
    save_json:      bool = True
    save_jsonl:     bool = True   # one article per line — great for streaming / ML
    save_index:     bool = True   # lightweight index file for fast lookups
    min_word_count: int  = 80     # drop stub articles

    # ── Prediction helpers ────────────────────────────────────────
    # Entity categories for downstream NER tagging hints
    entity_hints: dict[str, list[str]] = field(default_factory=lambda: {
        "countries":       ["Iran", "Israel", "US", "USA", "United States", "Russia", "China"],
        "organizations":   ["IRGC", "NATO", "UN", "CIA", "Mossad", "IDF", "Pentagon"],
        "event_types":     ["airstrike", "ceasefire", "sanctions", "nuclear", "missile", "invasion"],
    })

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "ScraperConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def save_json_file(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
