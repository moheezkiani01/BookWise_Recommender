# 📚 BookWise Recommender

A hybrid book recommendation engine that blends **collaborative filtering** (SVD matrix factorisation) with **content-based filtering** (TF-IDF + cosine similarity) into a single weighted scoring system, served over a **FastAPI** REST interface.

---

## Project Structure

```
BookWise/
├── config/
│   └── settings.yaml          ← All hyperparameters & paths
├── data/
│   ├── generate_data.py        ← Synthetic dataset generator
│   ├── books_catalog.csv       ← Generated book metadata
│   └── user_interactions.csv   ← Generated user–book ratings
├── src/
│   ├── collaborative/
│   │   └── svd_engine.py       ← SVD matrix factorisation
│   ├── content_based/
│   │   └── tfidf_engine.py     ← TF-IDF + cosine similarity
│   ├── hybrid/
│   │   └── fusion.py           ← Weighted score blending
│   ├── evaluation/
│   │   └── metrics.py          ← Precision@K, Recall@K, NDCG@K
│   └── utils/
│       └── loader.py           ← Config loader + DataStore
├── api/
│   ├── main.py                 ← FastAPI app factory
│   ├── router.py               ← Route handlers
│   └── schemas.py              ← Pydantic request/response models
├── artifacts/                  ← Serialised model files (auto-created)
├── report/
│   └── evaluation_report.md    ← Auto-generated evaluation results
├── train.py                    ← Training pipeline entry point
├── evaluate.py                 ← Offline evaluation entry point
└── requirements.txt
```

---

## Quick Start

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

### 2 · Generate data, train models

```bash
python train.py
```

This will:
- Generate `data/books_catalog.csv` and `data/user_interactions.csv`
- Train the SVD collaborative filter (with cross-validation)
- Build the TF-IDF content similarity matrix
- Save artefacts to `artifacts/`

### 3 · Run offline evaluation

```bash
python evaluate.py
```

Prints a Precision / Recall / NDCG comparison table and writes `report/evaluation_report.md`.

### 4 · Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Reference

### `GET /recommend/{user_id}`

Returns the top-N personalised book recommendations.

**Query parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_n`   | int  | 10      | Number of books to return (1–50) |

**Example request**

```
GET /recommend/42?top_n=5
```

**Example response**

```json
{
  "user_id": 42,
  "is_cold_start": false,
  "recommendations": [
    {
      "book_id": 87,
      "title": "Book Title 087",
      "author": "Nadia Osei",
      "genre": "Literary Fiction",
      "avg_rating": 4.31,
      "score": 0.8214,
      "source": "hybrid"
    },
    ...
  ],
  "total": 5
}
```

**Cold-start response** (unknown user)

```json
{
  "user_id": 9999,
  "is_cold_start": true,
  "recommendations": [
    {
      "book_id": 12,
      "title": "Book Title 012",
      "author": "Elena Voss",
      "genre": "Science Fiction",
      "avg_rating": 4.78,
      "score": 0.9100,
      "source": "content_based"
    },
    ...
  ],
  "total": 10
}
```

### `GET /health`

```json
{ "status": "ok", "version": "1.0.0", "message": "BookWise Recommender API is running" }
```

---

## System Design

### Hybrid Scoring

```
hybrid_score(u, i) = α · norm(CF_score(u, i))  +  β · norm(CB_score(u, i))
```

Where `α + β = 1.0` (defaults: α = 0.60, β = 0.40).  
Both score streams are **min-max normalised** before blending so they live on the same scale.

### Cold-Start Handling

| User type | Strategy |
|-----------|----------|
| Warm user (seen in training) | Hybrid blend (CF + CB) |
| Semi-cold (has recent interactions, not in training) | Content-based from liked books |
| Fully cold (brand new, no history) | Popularity-weighted genre similarity |

### Configuration

All hyperparameters live in `config/settings.yaml`. Key knobs:

```yaml
collaborative:
  n_factors: 50      # SVD latent dimensions
  n_epochs:  30      # training iterations

hybrid:
  collab_weight: 0.60  # α – increase for heavier CF influence
  content_weight: 0.40 # β – increase for heavier content influence
  top_n: 10

evaluation:
  relevance_threshold: 4.0  # minimum rating counted as "relevant"
```

---

## Evaluation Metrics

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Precision@K** | hits@K / K | Fraction of recommended items that are relevant |
| **Recall@K** | hits@K / \|relevant\| | Fraction of relevant items that were recommended |
| **NDCG@K** | DCG@K / IDCG@K | Ranking quality; higher-ranked relevant items score more |

---

## Reproducibility

- Random seed is set globally in `config/settings.yaml` (`data.random_seed: 42`)
- Data generation, train/test splits, and model initialisation all consume this seed
- Running `python train.py` followed by `python evaluate.py` should produce
  **identical results** on the same machine and Python/library versions

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `scikit-surprise` | SVD collaborative filtering |
| `scikit-learn` | TF-IDF vectorisation, cosine similarity |
| `scipy` | Sparse matrix operations |
| `pandas` / `numpy` | Data manipulation |
| `fastapi` / `uvicorn` | REST API framework |
| `pyyaml` | Configuration management |
| `pydantic` | Request/response schema validation |
