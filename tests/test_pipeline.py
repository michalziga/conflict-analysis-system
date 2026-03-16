"""
tests/test_pipeline.py
----------------------
Unit tests for the news scraper pipeline.
Run with: pytest tests/ -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# ensure package importable from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from news_scraper.config   import ScraperConfig
from news_scraper.models   import ArticleRecord, FeedResult, ScrapeSession
from news_scraper.scraper  import score_article
from news_scraper.analysis import (
    tokenize, count_phrases, analyze_corpus,
    enrich_article,
)


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

class TestScraperConfig:
    def test_default_keywords_non_empty(self):
        cfg = ScraperConfig()
        assert len(cfg.keywords) > 0

    def test_default_feeds_non_empty(self):
        cfg = ScraperConfig()
        assert len(cfg.feeds) > 0

    def test_all_feeds_are_urls(self):
        cfg = ScraperConfig()
        for feed in cfg.feeds:
            assert feed.startswith("http"), f"Not a URL: {feed}"

    def test_to_dict_roundtrip(self):
        cfg  = ScraperConfig()
        d    = cfg.to_dict()
        cfg2 = ScraperConfig(**d)
        assert cfg.keywords    == cfg2.keywords
        assert cfg.feeds       == cfg2.feeds
        assert cfg.max_workers == cfg2.max_workers

    def test_from_json(self, tmp_path):
        data = {"keywords": ["test"], "feeds": ["https://example.com/rss"]}
        p    = tmp_path / "cfg.json"
        p.write_text(json.dumps(data))
        cfg  = ScraperConfig.from_json(p)
        assert cfg.keywords == ["test"]
        assert cfg.feeds    == ["https://example.com/rss"]


# ─────────────────────────────────────────────
#  Models
# ─────────────────────────────────────────────

class TestArticleRecord:
    def test_default_fields(self):
        a = ArticleRecord()
        assert a.url_hash == ""
        assert a.relevance_score == 0.0
        assert isinstance(a.authors, list)
        assert isinstance(a.entities, dict)

    def test_to_dict_serialisable(self):
        a = ArticleRecord(title="Test", url="https://example.com", word_count=200)
        d = a.to_dict()
        assert json.dumps(d)  # must be JSON-serialisable
        assert d["title"]      == "Test"
        assert d["word_count"] == 200

    def test_to_index_record_drops_embedding_and_full_text(self):
        a = ArticleRecord(
            full_text="long article body...",
            embedding_vector=[0.1, 0.2, 0.3],
        )
        idx = a.to_index_record()
        assert "embedding_vector" not in idx
        assert "full_text"        not in idx

    def test_to_jsonl_line_is_valid_json(self):
        a    = ArticleRecord(title="IRGC missile test", url="https://example.com")
        line = a.to_jsonl_line()
        parsed = json.loads(line)
        assert parsed["title"] == "IRGC missile test"


class TestScrapeSession:
    def test_summary_empty_session(self):
        s = ScrapeSession(session_id="test_001", started_at="2026-01-01T00:00:00Z")
        summary = s.summary()
        assert summary["session_id"]     == "test_001"
        assert summary["total_articles"] == 0
        assert summary["errors"]         == []

    def test_summary_top_articles_sorted(self):
        arts = [
            ArticleRecord(title="A", relevance_score=0.9),
            ArticleRecord(title="B", relevance_score=0.4),
            ArticleRecord(title="C", relevance_score=0.7),
        ]
        s = ScrapeSession(articles=arts, total_scraped=3)
        top = s.summary()["top_articles"]
        scores = [t["score"] for t in top]
        assert scores == sorted(scores, reverse=True)


# ─────────────────────────────────────────────
#  Scraper — unit tests (no network)
# ─────────────────────────────────────────────

class TestScoreArticle:
    def test_no_match_returns_false(self):
        matched, kws, score = score_article("weather today", "sunshine", ["Iran", "missile"])
        assert matched is False
        assert kws    == []
        assert score  == 0.0

    def test_title_hit_scores_higher_than_body_only(self):
        kws = ["Iran", "missile"]
        _, _, score_title = score_article("Iran fired a missile", "Iran missile strike", kws)
        _, _, score_body  = score_article("Iran fired a missile", "Unrelated headline",  kws)
        assert score_title > score_body

    def test_score_capped_at_one(self):
        kws  = ["Iran"]
        _, _, score = score_article("Iran Iran Iran", "Iran Iran", kws)
        assert score <= 1.0

    def test_case_insensitive(self):
        matched, kws, _ = score_article("IRAN launched missiles", "title", ["iran", "missile"])
        assert matched is True
        assert "iran" in [k.lower() for k in kws]

    def test_partial_word_no_match(self):
        """'Iraq' should not match keyword 'Iran'"""
        matched, _, _ = score_article("Iraq situation", "Iraq update", ["Iran"])
        assert matched is False

    def test_multiple_keywords_all_matched(self):
        _, kws, _ = score_article(
            "Iran launched a ballistic missile at Israel",
            "Iran missile strike on Israel",
            ["Iran", "missile", "Israel"],
        )
        assert set(kws) == {"Iran", "missile", "Israel"}


# ─────────────────────────────────────────────
#  Analysis
# ─────────────────────────────────────────────

class TestTokenize:
    def test_removes_stopwords(self):
        tokens = tokenize("the war in Iran")
        assert "the" not in tokens
        assert "in"  not in tokens
        assert "war" in tokens
        assert "Iran".lower() in tokens

    def test_removes_punctuation(self):
        tokens = tokenize("Iran's IRGC launched missiles!")
        assert all(t.isalpha() for t in tokens)

    def test_empty_string_returns_empty(self):
        assert tokenize("") == []

    def test_single_char_tokens_removed(self):
        tokens = tokenize("a b c Iran")
        assert "a" not in tokens
        assert "b" not in tokens


class TestCountPhrases:
    def test_finds_phrase(self):
        results = dict(count_phrases("Iran fired a ballistic missile", ["ballistic missile"]))
        assert results.get("ballistic missile", 0) >= 1

    def test_case_insensitive(self):
        results = dict(count_phrases("BALLISTIC MISSILE launch", ["ballistic missile"]))
        assert "ballistic missile" in results

    def test_zero_count_excluded(self):
        results = count_phrases("nothing relevant here", ["ballistic missile", "ceasefire"])
        assert len(results) == 0

    def test_sorted_descending(self):
        text = "airstrike airstrike airstrike ceasefire"
        results = count_phrases(text, ["airstrike", "ceasefire"])
        counts = [c for _, c in results]
        assert counts == sorted(counts, reverse=True)

    def test_multi_word_not_split(self):
        """'Iran' alone should not count as 'Iran nuclear'"""
        results = dict(count_phrases("Iran was mentioned", ["Iran nuclear"]))
        assert "Iran nuclear" not in results


class TestAnalyzeCorpus:
    def _make_articles(self, n: int = 5) -> list[dict]:
        return [
            {
                "title":      f"Iran missile strike {i}",
                "full_text":  "Iran launched a ballistic missile. The IRGC confirmed the airstrike.",
                "summary":    "Iran IRGC ballistic missile airstrike",
                "word_count": 50,
            }
            for i in range(n)
        ]

    def test_returns_all_sections(self):
        result = analyze_corpus(self._make_articles())
        for key in ["corpus_stats", "military_tech", "political_leaders",
                    "organizations", "event_types", "top_unigrams"]:
            assert key in result

    def test_corpus_stats_count(self):
        arts   = self._make_articles(7)
        result = analyze_corpus(arts)
        assert result["corpus_stats"]["total_articles"] == 7

    def test_military_tech_finds_phrases(self):
        result = analyze_corpus(self._make_articles())
        tech   = dict(result["military_tech"])
        assert "ballistic missile" in tech
        assert tech["ballistic missile"] >= 5

    def test_empty_corpus_returns_zeros(self):
        result = analyze_corpus([])
        assert result["corpus_stats"]["total_articles"] == 0
        assert result["military_tech"] == []


class TestEnrichArticle:
    def test_adds_analysis_block(self):
        art = {
            "full_text": "Iran launched a ballistic missile. The IRGC was involved.",
            "summary":   "",
        }
        enriched = enrich_article(art)
        assert "analysis" in enriched
        assert "tech_mentions"   in enriched["analysis"]
        assert "org_mentions"    in enriched["analysis"]
        assert "analyzed_at"     in enriched["analysis"]

    def test_tech_mention_counted(self):
        art      = {"full_text": "ballistic missile fired twice ballistic missile", "summary": ""}
        enriched = enrich_article(art)
        assert enriched["analysis"]["tech_mentions"].get("ballistic missile", 0) >= 2

    def test_does_not_mutate_original_when_copied(self):
        art  = {"full_text": "Iran IRGC", "summary": ""}
        copy = dict(art)
        enrich_article(copy)
        assert "analysis" not in art


# ─────────────────────────────────────────────
#  Writer — filesystem tests
# ─────────────────────────────────────────────

class TestWriter:
    def _make_session(self) -> ScrapeSession:
        arts = [
            ArticleRecord(
                url_hash   = f"hash{i}",
                url        = f"https://example.com/{i}",
                title      = f"Article {i}",
                full_text  = "Iran IRGC ballistic missile " * 30,
                word_count = 120,
                relevance_score = 0.7,
                source_feed = "https://feeds.bbci.co.uk/news/world/rss.xml",
                scraped_at  = datetime.now(timezone.utc).isoformat(),
            )
            for i in range(3)
        ]
        s = ScrapeSession(
            session_id   = "test_session",
            started_at   = datetime.now(timezone.utc).isoformat(),
            finished_at  = datetime.now(timezone.utc).isoformat(),
            total_scraped= len(arts),
            articles     = arts,
        )
        s.feed_results.append(FeedResult(feed_url="https://example.com/rss", articles=arts))
        return s

    def test_write_outputs_creates_files(self, tmp_path):
        from news_scraper.writer  import write_outputs
        cfg          = ScraperConfig(output_dir=str(tmp_path))
        session      = self._make_session()
        written      = write_outputs(session, cfg)

        assert written["meta"].exists()
        assert written["json"].exists()
        assert written["jsonl"].exists()
        assert written["index"].exists()

    def test_jsonl_round_trip(self, tmp_path):
        from news_scraper.writer import write_outputs
        cfg     = ScraperConfig(output_dir=str(tmp_path))
        session = self._make_session()
        written = write_outputs(session, cfg)

        lines = [
            json.loads(row)
            for row in written["jsonl"].read_text().splitlines()
            if row.strip()
        ]
        assert len(lines) == 3
        assert all("url_hash" in rec for rec in lines)

    def test_master_index_appends(self, tmp_path):
        from news_scraper.writer import write_outputs
        cfg = ScraperConfig(output_dir=str(tmp_path))

        write_outputs(self._make_session(), cfg)
        write_outputs(self._make_session(), cfg)

        index_path = tmp_path / "index" / "master_index.jsonl"
        lines      = [row for row in index_path.read_text().splitlines() if row.strip()]
        # Two sessions × 3 articles each = 6 lines
        assert len(lines) == 6

    def test_stub_articles_filtered(self, tmp_path):
        from news_scraper.writer import write_outputs
        cfg = ScraperConfig(output_dir=str(tmp_path), min_word_count=100)
        session = self._make_session()
        # make one article a stub (< min_word_count)
        session.articles[0].word_count = 10
        written = write_outputs(session, cfg)

        lines = [
            json.loads(row)
            for row in written["jsonl"].read_text().splitlines()
            if row.strip()
        ]
        assert len(lines) == 2  # only the two non-stub articles
