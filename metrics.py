"""
src/evaluation/metrics.py
──────────────────────────
Offline evaluation suite: Precision@K, Recall@K, and NDCG@K.

All metrics follow the standard information-retrieval definitions and
operate on a held-out test set rather than the training interactions,
ensuring the measurements reflect genuine generalisation quality.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from collections import defaultdict


# ── Low-level metric functions ─────────────────────────────────────────────

def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of top-k recommendations that are relevant."""
    if k == 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant items recovered in top-k."""
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain.
    Binary relevance: 1 if item is in *relevant*, else 0.
    """
    dcg  = sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant
    )
    # Ideal DCG: all |relevant| items ranked first, capped at k
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ── Dataset-level evaluator ────────────────────────────────────────────────

class RecommenderEvaluator:
    """
    Evaluates any callable recommender over a held-out test set.

    Usage
    -----
    evaluator = RecommenderEvaluator(cfg)
    results   = evaluator.evaluate(
        recommend_fn=hybrid.recommend,
        test_df=datastore.test,
        interactions_df=datastore.all_interactions,
    )
    """

    def __init__(self, cfg: dict) -> None:
        self.k_values  = cfg["evaluation"]["k_values"]
        self.rel_thresh = cfg["evaluation"]["relevance_threshold"]

    def evaluate(
        self,
        recommend_fn,           # callable(user_id, interactions_df) → list[dict]
        test_df:         pd.DataFrame,
        interactions_df: pd.DataFrame,
        label:           str = "model",
    ) -> dict:
        """
        Parameters
        ----------
        recommend_fn     : function that accepts (user_id, interactions_df)
                           and returns [{book_id, score, ...}, ...]
        test_df          : held-out interactions
        interactions_df  : full interaction history (for context passing)
        label            : tag to include in the results dict

        Returns
        -------
        Nested dict keyed by metric name and K value.
        """
        # Build ground-truth: for each user, the set of test books they liked
        ground_truth: dict[int, set[int]] = defaultdict(set)
        for _, row in test_df.iterrows():
            if row["rating"] >= self.rel_thresh:
                ground_truth[row["user_id"]].add(int(row["book_id"]))

        users_with_gt = [u for u, items in ground_truth.items() if items]

        accumulator: dict[str, dict[int, list[float]]] = {
            "precision": defaultdict(list),
            "recall":    defaultdict(list),
            "ndcg":      defaultdict(list),
        }

        for uid in users_with_gt:
            try:
                recs = recommend_fn(user_id=uid, interactions_df=interactions_df)
                rec_ids = [r["book_id"] for r in recs]
                relevant = ground_truth[uid]

                for k in self.k_values:
                    accumulator["precision"][k].append(precision_at_k(rec_ids, relevant, k))
                    accumulator["recall"][k].append(recall_at_k(rec_ids, relevant, k))
                    accumulator["ndcg"][k].append(ndcg_at_k(rec_ids, relevant, k))
            except Exception:
                # Skip users where the model errors (e.g. truly unseen IDs)
                continue

        results: dict = {"label": label, "n_users_evaluated": len(users_with_gt)}
        for metric, k_scores in accumulator.items():
            results[metric] = {
                k: round(float(np.mean(scores)), 4)
                for k, scores in k_scores.items()
                if scores
            }

        return results


def format_results_table(results_list: list[dict]) -> str:
    """Pretty-print a comparison table of multiple model results."""
    k_vals = sorted({k for r in results_list for k in r.get("precision", {}).keys()})
    metrics = ["precision", "recall", "ndcg"]

    header_parts = ["Model".ljust(22)]
    for m in metrics:
        for k in k_vals:
            header_parts.append(f"{m.upper()}@{k}".rjust(12))
    header = "  ".join(header_parts)
    sep    = "─" * len(header)

    rows = [sep, header, sep]
    for r in results_list:
        parts = [r["label"].ljust(22)]
        for m in metrics:
            for k in k_vals:
                val = r.get(m, {}).get(k, float("nan"))
                parts.append(f"{val:.4f}".rjust(12))
        rows.append("  ".join(parts))
    rows.append(sep)
    return "\n".join(rows)
