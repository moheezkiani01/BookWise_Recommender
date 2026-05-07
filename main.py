"""
api/main.py
────────────
FastAPI application factory for the BookWise Recommender.

On startup the app loads the pre-trained model artefacts from disk
into a shared `app_state` dict so every request can reuse the same
in-memory objects without re-loading from disk each time.
"""

from __future__ import annotations

import os
import sys
import pickle

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Make project root importable regardless of where uvicorn is launched
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.router import router
from src.utils.loader import load_config

# ── Shared mutable state (loaded once at startup) ──────────────────────────
app_state: dict = {
    "hybrid":      None,
    "books":       None,
    "interactions": None,
    "known_users": set(),
}

# ── Application factory ────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title="BookWise Recommendation API",
        description=(
            "Hybrid collaborative + content-based book recommender. "
            "Supports warm and cold-start users."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="")

    @app.on_event("startup")
    def _load_models():
        cfg        = load_config(os.path.join(ROOT, "config", "settings.yaml"))
        artefact_dir = os.path.join(ROOT, "artifacts")

        hybrid_path = os.path.join(artefact_dir, "hybrid_engine.pkl")
        if not os.path.exists(hybrid_path):
            print("[WARNING] No trained model found. Run `python train.py` first.")
            return

        with open(hybrid_path, "rb") as fh:
            payload = pickle.load(fh)

        app_state["hybrid"]       = payload["hybrid"]
        app_state["known_users"]  = payload["known_users"]
        app_state["books"]        = pd.read_csv(os.path.join(ROOT, cfg["data"]["books_path"]))
        app_state["interactions"] = pd.read_csv(os.path.join(ROOT, cfg["data"]["interactions_path"]))
        print(f"[INFO] Model loaded. Known users: {len(app_state['known_users'])}")

    return app


app = create_app()


if __name__ == "__main__":
    cfg  = load_config(os.path.join(ROOT, "config", "settings.yaml"))
    uvicorn.run(
        "api.main:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        reload=cfg["api"]["reload"],
    )
