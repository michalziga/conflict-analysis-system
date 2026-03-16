# Downstream Pipeline Guide

How to connect the scraper output to NER, embeddings, search indexing, and prediction models.

## 1. NER — Entity Extraction (spaCy)

```python
import spacy, json
from pathlib import Path

nlp = spacy.load("en_core_web_trf")

enriched = []
with open("output/index/master_index.jsonl") as f:
    for line in f:
        art = json.loads(line)
        doc = nlp(art["full_text"][:5000])  # cap for speed
        art["entities"] = {
            "GPE":    [e.text for e in doc.ents if e.label_ == "GPE"],
            "ORG":    [e.text for e in doc.ents if e.label_ == "ORG"],
            "PERSON": [e.text for e in doc.ents if e.label_ == "PERSON"],
            "EVENT":  [e.text for e in doc.ents if e.label_ == "EVENT"],
        }
        enriched.append(art)
```

## 2. Embeddings (sentence-transformers)

```python
from sentence_transformers import SentenceTransformer
import json, numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("output/index/master_index.jsonl") as f:
    articles = [json.loads(l) for l in f if l.strip()]

# Embed title + summary for each article
texts   = [a["title"] + ". " + a["summary"] for a in articles]
vectors = model.encode(texts, batch_size=32, show_progress_bar=True)

for art, vec in zip(articles, vectors):
    art["embedding_vector"] = vec.tolist()
    art["embedding_model"]  = "all-MiniLM-L6-v2"
```

## 3. Elasticsearch Indexing

```python
from elasticsearch import Elasticsearch, helpers
import json

es = Elasticsearch("http://localhost:9200")

# Create index with mappings
es.indices.create(index="cas_articles", body={
    "mappings": {
        "properties": {
            "url_hash":        {"type": "keyword"},
            "title":           {"type": "text"},
            "full_text":       {"type": "text"},
            "summary":         {"type": "text"},
            "scraped_at":      {"type": "date"},
            "publish_date":    {"type": "date"},
            "source_feed":     {"type": "keyword"},
            "relevance_score": {"type": "float"},
            "matched_keywords":{"type": "keyword"},
            "language":        {"type": "keyword"},
            "embedding_vector":{"type": "dense_vector", "dims": 384},
        }
    }
}, ignore=400)

def gen_actions(jsonl_path):
    with open(jsonl_path) as f:
        for line in f:
            doc = json.loads(line)
            yield {
                "_index": "cas_articles",
                "_id":    doc["url_hash"],
                "_source": doc,
            }

helpers.bulk(es, gen_actions("output/index/master_index.jsonl"))
```

## 4. Sentiment Analysis

```python
from transformers import pipeline
import json

sentiment = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True, max_length=512,
)

with open("output/index/master_index.jsonl") as f:
    articles = [json.loads(l) for l in f if l.strip()]

for art in articles:
    result = sentiment(art["summary"] or art["title"])[0]
    art["sentiment"] = {
        "label": result["label"],
        "score": round(result["score"], 4),
    }
```

## 5. Escalation Prediction

The `prediction` field is intentionally empty at scrape time. A simple baseline:

```python
import json

ESCALATION_WEIGHTS = {
    "ballistic missile": 8, "nuclear deal": 6,  "airstrike": 7,
    "ceasefire": -4,        "sanctions": 3,      "drone attack": 6,
    "irgc": 5,              "idf": 5,            "ground invasion": 9,
    "nuclear talks": 4,     "missile strike": 8, "peace talks": -5,
}

def predict_escalation(art: dict) -> dict:
    analysis = art.get("analysis", {})
    score    = 0
    signals  = []

    for category in ["tech_mentions", "event_mentions", "org_mentions"]:
        for phrase, count in analysis.get(category, {}).items():
            weight = ESCALATION_WEIGHTS.get(phrase, 0)
            score += weight * count
            if weight > 0:
                signals.append(phrase)

    # Normalise to 0–100
    normalised = min(100, max(0, score))

    return {
        "escalation_risk":  normalised / 100,
        "escalation_class": "critical" if normalised > 70 else
                            "elevated" if normalised > 40 else "watch",
        "signals":          signals[:5],
        "model_version":    "baseline-v1",
        "predicted_at":     datetime.now(timezone.utc).isoformat(),
    }

# Apply to enriched JSONL
with open("output/analysis/enriched_latest.jsonl") as f:
    articles = [json.loads(l) for l in f if l.strip()]

for art in articles:
    art["prediction"] = predict_escalation(art)
```

## 6. Claude API Summaries (Batch)

```python
import anthropic, json

client = anthropic.Anthropic()

with open("output/index/master_index.jsonl") as f:
    articles = [json.loads(l) for l in f if l.strip()]

batch = client.beta.messages.batches.create(
    requests=[
        {
            "custom_id": art["url_hash"],
            "params": {
                "model": "claude-sonnet-4-6",
                "max_tokens": 200,
                "system": (
                    "You are a geopolitical analyst. Summarise this article in exactly "
                    "2 sentences focusing on: (1) the key actors and actions, "
                    "(2) the strategic / escalation significance. "
                    "Be precise and factual."
                ),
                "messages": [{
                    "role": "user",
                    "content": art["full_text"][:3000]
                }]
            }
        }
        for art in articles
        if art.get("full_text")
    ]
)

print(f"Batch submitted: {batch.id}")
print("Poll results at: client.beta.messages.batches.retrieve(batch.id)")
```

Cost at current Sonnet 4.6 pricing with Batch API (50% discount):
~$0.0026 per article. For 500 articles: ~$1.30.
