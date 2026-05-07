"""
Synthetic dataset generator for BookWise Recommender.
Produces two CSV files:
  - books_catalog.csv      : book metadata
  - user_interactions.csv  : user–book rating history
"""

import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Book catalogue constants ────────────────────────────────────────────────
GENRES = [
    "Literary Fiction", "Science Fiction", "Fantasy", "Mystery",
    "Historical Fiction", "Non-Fiction", "Biography", "Self-Help",
    "Horror", "Romance", "Thriller", "Philosophy",
]

AUTHORS = [
    "Ayaan Mirza", "Sophia Crane", "Luca Ferretti", "Nadia Osei",
    "James Calloway", "Elena Voss", "Omar Khalid", "Priya Nandan",
    "Samuel Bright", "Yuki Tanaka", "Clara Novak", "Reza Ahmadi",
    "Isabelle Morin", "Dev Sharma", "Amara Diallo",
]

DESCRIPTION_TEMPLATES = [
    "A gripping tale set in {setting} where {protagonist} must navigate "
    "a world of {conflict} and self-discovery.",
    "Exploring themes of {theme} and identity, this {genre} follows "
    "{protagonist} through unexpected twists.",
    "Set against the backdrop of {setting}, the story unravels {conflict} "
    "while questioning the nature of {theme}.",
    "An award-winning work that blends {genre} with philosophy, "
    "centering on {protagonist}'s journey through {conflict}.",
    "This deeply human story captures {theme} through the eyes of "
    "{protagonist}, set in a vivid {setting}.",
]

SETTINGS   = ["dystopian cities", "Victorian England", "outer space",
               "rural Japan", "a post-war Europe", "modern Tehran",
               "the deep ocean", "colonial India", "near-future Lagos"]
CONFLICTS  = ["political intrigue", "family betrayal", "existential dread",
               "a race against time", "forbidden knowledge", "systemic oppression"]
THEMES     = ["freedom", "love", "power", "memory", "identity", "grief"]
PROTAGONISTS = ["a young scholar", "a seasoned detective", "an unlikely hero",
                 "a grieving mother", "an AI researcher", "a rogue journalist"]


def make_books(n: int = 200) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        genre = random.choice(GENRES)
        tpl   = random.choice(DESCRIPTION_TEMPLATES)
        desc  = tpl.format(
            setting=random.choice(SETTINGS),
            protagonist=random.choice(PROTAGONISTS),
            conflict=random.choice(CONFLICTS),
            theme=random.choice(THEMES),
            genre=genre.lower(),
        )
        rows.append({
            "book_id":     i,
            "title":       f"Book Title {i:03d}",
            "author":      random.choice(AUTHORS),
            "genre":       genre,
            "year":        random.randint(1980, 2024),
            "avg_rating":  round(random.uniform(2.5, 5.0), 2),
            "description": desc,
        })
    return pd.DataFrame(rows)


def make_interactions(books_df: pd.DataFrame,
                      n_users: int = 300,
                      sparsity: float = 0.05) -> pd.DataFrame:
    """
    Simulate user–book ratings.
    Each user rates a random subset of books; ratings follow a
    skewed distribution biased toward each user's preferred genres.
    """
    book_ids = books_df["book_id"].tolist()
    genre_map = books_df.set_index("book_id")["genre"].to_dict()
    rows = []

    for uid in range(1, n_users + 1):
        fav_genres = random.sample(GENRES, k=random.randint(2, 4))
        n_rated    = max(5, int(len(book_ids) * sparsity * random.uniform(0.5, 2.0)))
        rated_books = random.sample(book_ids, min(n_rated, len(book_ids)))

        for bid in rated_books:
            base = 4.0 if genre_map[bid] in fav_genres else 2.5
            rating = round(np.clip(np.random.normal(base, 0.8), 1.0, 5.0), 1)
            rows.append({"user_id": uid, "book_id": bid, "rating": rating})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import os
    out_dir = os.path.dirname(__file__)

    print("Generating book catalogue …")
    books = make_books(n=200)
    books.to_csv(os.path.join(out_dir, "books_catalog.csv"), index=False)
    print(f"  ✓ {len(books)} books written to books_catalog.csv")

    print("Generating user interactions …")
    interactions = make_interactions(books, n_users=300, sparsity=0.06)
    interactions.to_csv(os.path.join(out_dir, "user_interactions.csv"), index=False)
    print(f"  ✓ {len(interactions)} interactions written to user_interactions.csv")
    print("Done.")
