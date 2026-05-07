"""
src/content_based/tfidf_engine.py
──────────────────────────────────
Content-based filter using TF-IDF vectorisation over book metadata,
with cosine similarity for nearest-neighbour retrieval.

Design notes
------------
* Book descriptions, titles, genres, and authors are concatenated
  into a single text corpus before vectorisation.
* Item–item cosine similarity is pre-computed and stored in a sparse
  matrix so inference is just a fast row look-up.
* For new (cold-start) users, we fall back purely to this engine
  using global popularity as a prior when no seed books are available.
"""

from __future__ import annotations

import pickle
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import csr_matrix


class ContentEngine:
    """
    TF-IDF + cosine similarity content-based recommender.
    """

    def __init__(self, cfg: dict) -> None:
        c = cfg["content_based"]
        self.vectorizer = TfidfVectorizer(
            max_features=c["max_features"],
            ngram_range=tuple(c["ngram_range"]),
            stop_words="english",
            sublinear_tf=True,         # log(1+tf) dampening for rare terms
        )
        self._top_k       = c["similarity_top_k"]
        self._book_ids:   list[int]         = []
        self._id_to_idx:  dict[int, int]    = {}
        self._sim_matrix: np.ndarray | None = None
        self._books_df:   pd.DataFrame | None = None
        self._trained = False

    # ── Training ───────────────────────────────────────────────────────────
    def fit(self, books_df: pd.DataFrame, combined_text: pd.Series) -> "ContentEngine":
        """
        Parameters
        ----------
        books_df      : full book catalogue DataFrame (must contain 'book_id')
        combined_text : pre-built concatenated text Series (same index as books_df)
        """
        self._books_df  = books_df.reset_index(drop=True)
        self._book_ids  = self._books_df["book_id"].tolist()
        self._id_to_idx = {bid: idx for idx, bid in enumerate(self._book_ids)}

        tfidf_matrix = self.vectorizer.fit_transform(combined_text.fillna(""))

        # Full pairwise similarity is O(n²); for large catalogues we
        # compute it in full only when n is manageable (≤ 5 000 items).
        # Beyond that a nearest-neighbour approximation would be better,
        # but for this task the synthetic catalogue is well within range.
        self._sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
        self._trained    = True
        return self

    # ── Inference ──────────────────────────────────────────────────────────
    def similar_to(
        self,
        book_id: int,
        exclude_ids: set | None = None,
        top_n: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Return the top-n books most similar to *book_id*.
        """
        self._check_trained()
        if exclude_ids is None:
            exclude_ids = set()

        idx     = self._id_to_idx.get(book_id)
        if idx is None:
            return []

        sim_row = self._sim_matrix[idx]  # type: ignore[index]
        ranked  = np.argsort(sim_row)[::-1]

        results = []
        for i in ranked:
            cid = self._book_ids[i]
            if cid == book_id or cid in exclude_ids:
                continue
            results.append((cid, float(sim_row[i])))
            if len(results) >= top_n:
                break
        return results

    def predict_for_user(
        self,
        seed_book_ids: list[int],
        exclude_ids: set | None = None,
        top_n: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Aggregate item-level similarity scores across *seed_book_ids*
        to form a user-level content-based ranking.
        """
        self._check_trained()
        if not seed_book_ids:
            return self._popular_fallback(exclude_ids or set(), top_n)

        score_acc: dict[int, float] = {}
        for sid in seed_book_ids:
            for bid, sim in self.similar_to(book_id=sid,
                                            exclude_ids=exclude_ids,
                                            top_n=self._top_k):
                score_acc[bid] = score_acc.get(bid, 0.0) + sim

        ranked = sorted(score_acc.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def score(self, book_id: int, seed_book_ids: list[int]) -> float:
        """
        Aggregate similarity of *book_id* against a user's seed items.
        Used by the hybrid layer for a consistent single-score interface.
        """
        self._check_trained()
        if not seed_book_ids:
            return 0.0
        total = 0.0
        for sid in seed_book_ids:
            idx_seed = self._id_to_idx.get(sid)
            idx_tgt  = self._id_to_idx.get(book_id)
            if idx_seed is None or idx_tgt is None:
                continue
            total += float(self._sim_matrix[idx_seed, idx_tgt])  # type: ignore
        return total / len(seed_book_ids)

    # ── Cold-start fallback ────────────────────────────────────────────────
    def _popular_fallback(self, exclude_ids: set, top_n: int) -> list[tuple[int, float]]:
        """
        When there are no seed items, rank by the book's average rating
        (normalised to [0, 1]) as a naive popularity prior.
        """
        df = self._books_df.copy()  # type: ignore[union-attr]
        df = df[~df["book_id"].isin(exclude_ids)]
        if "avg_rating" in df.columns:
            df = df.sort_values("avg_rating", ascending=False)
            min_r, max_r = df["avg_rating"].min(), df["avg_rating"].max()
            rng = max(max_r - min_r, 1e-9)
            return [
                (row["book_id"], (row["avg_rating"] - min_r) / rng)
                for _, row in df.head(top_n).iterrows()
            ]
        return [(bid, 0.5) for bid in df["book_id"].head(top_n).tolist()]

    # ── Persistence ────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "vectorizer":  self.vectorizer,
                "sim_matrix":  self._sim_matrix,
                "book_ids":    self._book_ids,
                "id_to_idx":   self._id_to_idx,
                "books_df":    self._books_df,
            }, fh)

    def load(self, path: str) -> "ContentEngine":
        with open(path, "rb") as fh:
            p = pickle.load(fh)
        self.vectorizer   = p["vectorizer"]
        self._sim_matrix  = p["sim_matrix"]
        self._book_ids    = p["book_ids"]
        self._id_to_idx   = p["id_to_idx"]
        self._books_df    = p["books_df"]
        self._trained     = True
        return self

    def _check_trained(self) -> None:
        if not self._trained:
            raise RuntimeError("ContentEngine must be fitted before inference.")
