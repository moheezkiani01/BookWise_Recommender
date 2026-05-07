"""
src/utils/loader.py
───────────────────
Handles all I/O and train/test splitting so the rest of
the pipeline never touches raw files directly.
"""

from __future__ import annotations

import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


class DataStore:
    """
    Central data store: loads the catalogue and interaction history,
    then exposes clean train/test splits for downstream modules.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._books: pd.DataFrame | None = None
        self._interactions: pd.DataFrame | None = None
        self._train: pd.DataFrame | None = None
        self._test: pd.DataFrame | None = None

    # ── Loaders ────────────────────────────────────────────────────────────
    def load(self) -> "DataStore":
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        books_path  = os.path.join(base, self.cfg["data"]["books_path"])
        inter_path  = os.path.join(base, self.cfg["data"]["interactions_path"])

        self._books        = pd.read_csv(books_path)
        self._interactions = pd.read_csv(inter_path)
        self._split()
        return self

    def _split(self) -> None:
        seed = self.cfg["data"]["random_seed"]
        ratio = self.cfg["data"]["test_split"]
        self._train, self._test = train_test_split(
            self._interactions,
            test_size=ratio,
            random_state=seed,
            stratify=self._interactions["user_id"] if self._check_stratify() else None,
        )

    def _check_stratify(self) -> bool:
        """Only stratify if every user has ≥ 2 interactions (sklearn requirement)."""
        counts = self._interactions["user_id"].value_counts()
        return bool((counts >= 2).all())

    # ── Public accessors ───────────────────────────────────────────────────
    @property
    def books(self) -> pd.DataFrame:
        self._guard()
        return self._books  # type: ignore[return-value]

    @property
    def train(self) -> pd.DataFrame:
        self._guard()
        return self._train  # type: ignore[return-value]

    @property
    def test(self) -> pd.DataFrame:
        self._guard()
        return self._test  # type: ignore[return-value]

    @property
    def all_interactions(self) -> pd.DataFrame:
        self._guard()
        return self._interactions  # type: ignore[return-value]

    def known_users(self) -> set:
        return set(self._train["user_id"].unique())  # type: ignore[union-attr]

    def _guard(self) -> None:
        if self._books is None:
            raise RuntimeError("Call DataStore.load() before accessing data.")

    # ── Helpers ────────────────────────────────────────────────────────────
    def build_combined_text(self) -> pd.Series:
        """Concatenate text fields specified in config for TF-IDF ingestion."""
        fields = self.cfg["content_based"]["text_fields"]
        available = [f for f in fields if f in self._books.columns]  # type: ignore
        return (
            self._books[available]  # type: ignore
            .fillna("")
            .apply(lambda row: " ".join(str(v) for v in row), axis=1)
        )
