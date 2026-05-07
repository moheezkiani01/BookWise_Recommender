"""
train.py
─────────
End-to-end training pipeline for the BookWise Recommender.

Run
---
    python train.py

What it does
------------
1. Generates synthetic data if CSVs are absent
2. Loads and splits the interaction dataset
3. Trains the collaborative-filtering SVD model
4. Builds the content-based TF-IDF model
5. Assembles the hybrid engine
6. Serialises all artefacts to /artifacts/
"""

from __future__ import annotations

import os
import sys
import pickle
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.utils.loader    import load_config, DataStore
from src.collaborative.svd_engine   import CollaborativeEngine
from src.content_based.tfidf_engine import ContentEngine
from src.hybrid.fusion              import HybridEngine


def banner(msg: str) -> None:
    print(f"\n{'─'*55}\n  {msg}\n{'─'*55}")


def main():
    cfg = load_config(os.path.join(ROOT, "config", "settings.yaml"))

    # ── 1. Generate data if needed ─────────────────────────────────────────
    books_path = os.path.join(ROOT, cfg["data"]["books_path"])
    inter_path = os.path.join(ROOT, cfg["data"]["interactions_path"])

    if not os.path.exists(books_path) or not os.path.exists(inter_path):
        banner("Generating synthetic dataset …")
        import subprocess
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "data", "generate_data.py")],
            check=True,
        )

    # ── 2. Load & split data ───────────────────────────────────────────────
    banner("Loading and splitting data …")
    store = DataStore(cfg).load()
    print(f"  Books       : {len(store.books)}")
    print(f"  Train rows  : {len(store.train)}")
    print(f"  Test rows   : {len(store.test)}")

    # ── 3. Collaborative filtering ─────────────────────────────────────────
    banner("Training collaborative filter (SVD) …")
    t0 = time.time()
    collab = CollaborativeEngine(cfg)
    collab.fit(store.train)
    print(f"  ✓ SVD trained in {time.time() - t0:.1f}s")

    cv_scores = collab.cross_validate(store.all_interactions, cv=3)
    print(f"  Cross-val RMSE : {cv_scores['rmse_mean']:.4f}")
    print(f"  Cross-val MAE  : {cv_scores['mae_mean']:.4f}")

    # ── 4. Content-based filtering ─────────────────────────────────────────
    banner("Building content-based model (TF-IDF) …")
    t0 = time.time()
    content = ContentEngine(cfg)
    combined_text = store.build_combined_text()
    content.fit(store.books, combined_text)
    print(f"  ✓ TF-IDF + similarity matrix built in {time.time() - t0:.1f}s")

    # ── 5. Assemble hybrid engine ──────────────────────────────────────────
    banner("Assembling hybrid engine …")
    hybrid = HybridEngine(collab, content, cfg)
    hybrid.register_known_users(store.known_users())
    print(f"  α (collab weight)  : {hybrid._alpha:.2f}")
    print(f"  β (content weight) : {hybrid._beta:.2f}")

    # ── 6. Persist artefacts ───────────────────────────────────────────────
    banner("Saving artefacts …")
    artefact_dir = os.path.join(ROOT, "artifacts")
    os.makedirs(artefact_dir, exist_ok=True)

    with open(os.path.join(artefact_dir, "hybrid_engine.pkl"), "wb") as fh:
        pickle.dump({
            "hybrid":      hybrid,
            "known_users": store.known_users(),
        }, fh)

    collab.save(os.path.join(artefact_dir, "collab_model.pkl"))
    content.save(os.path.join(artefact_dir, "content_model.pkl"))

    print("  ✓ artifacts/hybrid_engine.pkl")
    print("  ✓ artifacts/collab_model.pkl")
    print("  ✓ artifacts/content_model.pkl")
    banner("Training complete! Run evaluate.py for metrics.")


if __name__ == "__main__":
    main()
