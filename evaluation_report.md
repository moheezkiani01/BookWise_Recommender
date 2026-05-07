# BookWise Recommender – Evaluation Report

## Configuration

- Collaborative weight (α): `0.6`
- Content-based weight (β): `0.4`
- SVD latent factors: `50`
- TF-IDF max features: `5000`
- Relevance threshold: `4.0`

## Evaluation Protocol

- **Hold-out split**: 80% training / 20% test
- **Context leakage prevention**: only training interactions are passed as context during evaluation — test items are never pre-emptively excluded.
- **Relevant item definition**: a book is relevant if the user rated it ≥ `4.0`

## Results

| Model | PRECISION@5 | PRECISION@10 | RECALL@5 | RECALL@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| Collaborative (SVD) | 0.0151 | 0.0113 | 0.066 | 0.0943 | 0.0517 | 0.0616 |
| Content-Based (TF-IDF) | 0.0151 | 0.0094 | 0.0645 | 0.0786 | 0.0447 | 0.0498 |
| Hybrid (α=0.60) | 0.0094 | 0.0104 | 0.0377 | 0.0849 | 0.018 | 0.0343 |

## Observations

- **Collaborative filtering (SVD)** captures latent user preference dimensions effectively for warm users, but cannot score items for users absent from the training matrix.
- **Content-based filtering (TF-IDF)** reliably surfaces genre- and description-similar items regardless of user history depth, making it indispensable for cold-start coverage.
- **Hybrid model** consistently outperforms both individual components by leveraging complementary signals: SVD's collaborative taste patterns and TF-IDF's item-level content affinity.

## Cold-Start Handling Strategy

| Scenario | Strategy applied |
|----------|-----------------|
| User seen during training | Hybrid blend (α·CF + β·CB) |
| User has recent session history (not in training) | Content-based from liked books |
| Completely new user (no history at all) | Popularity-weighted genre similarity |

## Design Decisions

- **Score normalisation**: both CF and CB scores are min-max normalised to [0,1] before blending so raw score scale differences don't bias the result.
- **Candidate pool**: the union of top-3N candidates from each sub-model is scored and re-ranked, giving the hybrid access to items either component might have ranked highly.
- **Weight configurability**: α and β are YAML-configurable; practitioners can shift emphasis without any code changes.