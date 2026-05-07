"""
src/hybrid/fusion.py
─────────────────────
Hybrid recommendation engine that blends collaborative-filtering
scores with content-based scores via a configurable weighted sum.

Blending strategy
-----------------
  hybrid_score(u, i) = α · cf_score(u, i)  +  (1-α) · cb_score(u, i)

Both score streams are min-max normalised to [0, 1] *before* blending
so they live on the same scale regardless of their raw distributions.

Cold-start handling
-------------------
When a user_id is absent from the training interactions (true cold
start), α is set to 0 and the recommendation falls back entirely to
the content-based engine using the catalogue's popularity prior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.collaborative.svd_engine import CollaborativeEngine
from src.content_based.tfidf_engine import ContentEngine


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    """Normalise a {item_id: score} dict to [0, 1]."""
    if not scores:
        return scores
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    rng = hi - lo if hi > lo else 1e-9
    return {k: (v - lo) / rng for k, v in scores.items()}


class HybridEngine:
    """
    Combines CollaborativeEngine and ContentEngine outputs into a
    single ranked recommendation list.
    """

    def __init__(
        self,
        collab_engine:  CollaborativeEngine,
        content_engine: ContentEngine,
        cfg:            dict,
    ) -> None:
        self.collab  = collab_engine
        self.content = content_engine

        self._alpha     = float(cfg["hybrid"]["collab_weight"])
        self._beta      = float(cfg["hybrid"]["content_weight"])
        self._top_n     = int(cfg["hybrid"]["top_n"])
        self._known_users: set = set()

        # Normalise weights so they always sum to 1
        total = self._alpha + self._beta
        self._alpha /= total
        self._beta  /= total

    # ── Public API ─────────────────────────────────────────────────────────
    def register_known_users(self, user_ids: set) -> None:
        """Tell the engine which users exist in the training data."""
        self._known_users = user_ids

    def recommend(
        self,
        user_id:     int,
        interactions_df: pd.DataFrame | None = None,
        top_n:       int | None = None,
    ) -> list[dict]:
        """
        Return a ranked list of recommendation dicts:
          [{"book_id": int, "score": float, "source": str}, ...]

        Parameters
        ----------
        user_id         : target user
        interactions_df : full interactions table (used to identify seen books)
        top_n           : override config top_n if provided
        """
        n = top_n or self._top_n
        is_cold_start = user_id not in self._known_users

        seen_ids: set = set()
        seed_ids: list[int] = []

        if interactions_df is not None:
            user_rows = interactions_df[interactions_df["user_id"] == user_id]
            seen_ids  = set(user_rows["book_id"].tolist())
            # Seed the content engine with books the user liked
            liked = user_rows[user_rows["rating"] >= 3.5]
            seed_ids = liked["book_id"].tolist()

        if is_cold_start:
            return self._cold_start_recommend(seen_ids, seed_ids, n)

        return self._warm_recommend(user_id, seen_ids, seed_ids, n)

    # ── Internal helpers ───────────────────────────────────────────────────
    def _warm_recommend(
        self,
        user_id:  int,
        seen_ids: set,
        seed_ids: list[int],
        n:        int,
    ) -> list[dict]:
        # --- Collaborative scores -----------------------------------------
        cf_raw = dict(
            self.collab.predict_for_user(user_id, seen_ids, top_n=n * 3)
        )

        # --- Content scores ------------------------------------------------
        cb_raw = dict(
            self.content.predict_for_user(seed_ids, seen_ids, top_n=n * 3)
        )

        # --- Candidate pool: union of both sets ----------------------------
        all_candidates = set(cf_raw.keys()) | set(cb_raw.keys())

        # Fill missing scores with 0 before normalisation
        cf_full = {c: cf_raw.get(c, 0.0) for c in all_candidates}
        cb_full = {c: cb_raw.get(c, 0.0) for c in all_candidates}

        cf_norm = _minmax(cf_full)
        cb_norm = _minmax(cb_full)

        hybrid: dict[int, float] = {
            c: self._alpha * cf_norm[c] + self._beta * cb_norm[c]
            for c in all_candidates
        }

        ranked = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:n]
        return [
            {"book_id": bid, "score": round(score, 4), "source": "hybrid"}
            for bid, score in ranked
        ]

    def _cold_start_recommend(
        self,
        seen_ids: set,
        seed_ids: list[int],
        n:        int,
    ) -> list[dict]:
        """
        Pure content-based path for users with no training history.
        If seed_ids is also empty (truly new user), falls back to
        popularity-weighted suggestions from the content engine.
        """
        cb_raw = dict(
            self.content.predict_for_user(seed_ids, seen_ids, top_n=n)
        )
        cb_norm = _minmax(cb_raw)
        ranked  = sorted(cb_norm.items(), key=lambda x: x[1], reverse=True)[:n]
        return [
            {"book_id": bid, "score": round(score, 4), "source": "content_based"}
            for bid, score in ranked
        ]

    # ── Score a single item (used by evaluator) ────────────────────────────
    def score_item(
        self,
        user_id:  int,
        book_id:  int,
        seed_ids: list[int],
    ) -> float:
        if user_id not in self._known_users:
            return self.content.score(book_id, seed_ids)

        cf_s = (self.collab.score(user_id, book_id) - 1.0) / 4.0   # scale [1-5] → [0-1]
        cb_s = self.content.score(book_id, seed_ids)
        return self._alpha * cf_s + self._beta * cb_s
