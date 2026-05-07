"""
src/collaborative/svd_engine.py
────────────────────────────────
Matrix-factorisation collaborative filter using scipy truncated SVD.
No scikit-surprise dependency — pure numpy/scipy, NumPy-2.x safe.

Algorithm
---------
1. Build a dense user×item matrix from sparse interactions.
2. Centre each user's observed ratings by subtracting their mean.
3. Decompose with truncated SVD: M ≈ U Σ Vᵀ (k latent factors).
4. Predict: r̂(u,i) = μ_u  +  (U·Σ)[u,:]  ·  V[:,i]
"""

from __future__ import annotations
import pickle, os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import KFold


class CollaborativeEngine:

    def __init__(self, cfg: dict) -> None:
        c = cfg["collaborative"]
        self._n_factors = int(c["n_factors"])
        self._seed      = int(cfg["data"]["random_seed"])
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._user_means:   np.ndarray | None = None
        self._uid_to_row: dict[int, int] = {}
        self._bid_to_col: dict[int, int] = {}
        self._row_to_uid: list[int] = []
        self._col_to_bid: list[int] = []
        self._trained = False

    # ── Training ───────────────────────────────────────────────────────────
    def fit(self, train_df: pd.DataFrame) -> "CollaborativeEngine":
        users = sorted(train_df["user_id"].unique())
        items = sorted(train_df["book_id"].unique())
        self._row_to_uid = users
        self._col_to_bid = items
        self._uid_to_row = {u: i for i, u in enumerate(users)}
        self._bid_to_col = {b: j for j, b in enumerate(items)}

        n_u, n_i = len(users), len(items)
        rows = train_df["user_id"].map(self._uid_to_row).values
        cols = train_df["book_id"].map(self._bid_to_col).values
        vals = train_df["rating"].values.astype(np.float64)

        # Build dense matrix for mean-centring (manageable for n≤500 users)
        R_dense = np.zeros((n_u, n_i), dtype=np.float64)
        R_dense[rows, cols] = vals

        self._user_means = np.array([
            R_dense[u, R_dense[u] != 0].mean() if (R_dense[u] != 0).any() else 0.0
            for u in range(n_u)
        ])

        # Centre observed entries
        R_c = R_dense.copy()
        for u in range(n_u):
            mask = R_c[u] != 0
            R_c[u, mask] -= self._user_means[u]

        # Truncated SVD
        k = min(self._n_factors, n_u - 1, n_i - 1)
        np.random.seed(self._seed)
        U, sigma, Vt = svds(csr_matrix(R_c), k=k)
        order = np.argsort(sigma)[::-1]
        U, sigma, Vt = U[:, order], sigma[order], Vt[order, :]

        self._user_factors = U * sigma     # (n_users, k)
        self._item_factors = Vt.T          # (n_items, k)
        self._trained = True
        return self

    # ── Inference ──────────────────────────────────────────────────────────
    def predict_for_user(self, user_id: int,
                         seen_book_ids: set | None = None,
                         top_n: int = 10) -> list[tuple[int, float]]:
        self._check_trained()
        if seen_book_ids is None:
            seen_book_ids = set()
        row = self._uid_to_row.get(user_id)
        if row is None:
            return []
        u_vec  = self._user_factors[row]
        mu     = self._user_means[row]
        scores = self._item_factors @ u_vec + mu
        cands  = [(bid, float(scores[j]))
                  for j, bid in enumerate(self._col_to_bid)
                  if bid not in seen_book_ids]
        cands.sort(key=lambda x: x[1], reverse=True)
        return cands[:top_n]

    def score(self, user_id: int, book_id: int) -> float:
        self._check_trained()
        row = self._uid_to_row.get(user_id)
        col = self._bid_to_col.get(book_id)
        if row is None or col is None:
            return 3.0
        return float(self._item_factors[col] @ self._user_factors[row]
                     + self._user_means[row])

    # ── Cross-validation ───────────────────────────────────────────────────
    def cross_validate(self, interactions_df: pd.DataFrame, cv: int = 3) -> dict:
        kf = KFold(n_splits=cv, shuffle=True, random_state=self._seed)
        rmses, maes = [], []
        mini_cfg = {"collaborative": {"n_factors": self._n_factors},
                    "data": {"random_seed": self._seed}}
        for tr_idx, val_idx in kf.split(interactions_df):
            tr, val = interactions_df.iloc[tr_idx], interactions_df.iloc[val_idx]
            tmp   = CollaborativeEngine(mini_cfg).fit(tr)
            preds = np.array([tmp.score(r["user_id"], r["book_id"])
                              for _, r in val.iterrows()])
            truth = val["rating"].values
            rmses.append(float(np.sqrt(np.mean((preds - truth) ** 2))))
            maes.append(float(np.mean(np.abs(preds - truth))))
        return {"rmse_mean": float(np.mean(rmses)),
                "mae_mean":  float(np.mean(maes))}

    # ── Persistence ────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "user_factors": self._user_factors,
                "item_factors": self._item_factors,
                "user_means":   self._user_means,
                "uid_to_row":   self._uid_to_row,
                "bid_to_col":   self._bid_to_col,
                "row_to_uid":   self._row_to_uid,
                "col_to_bid":   self._col_to_bid,
            }, fh)

    def load(self, path: str) -> "CollaborativeEngine":
        with open(path, "rb") as fh:
            p = pickle.load(fh)
        self._user_factors = p["user_factors"]
        self._item_factors = p["item_factors"]
        self._user_means   = p["user_means"]
        self._uid_to_row   = p["uid_to_row"]
        self._bid_to_col   = p["bid_to_col"]
        self._row_to_uid   = p["row_to_uid"]
        self._col_to_bid   = p["col_to_bid"]
        self._trained      = True
        return self

    def _check_trained(self) -> None:
        if not self._trained:
            raise RuntimeError("CollaborativeEngine must be fitted before inference.")
