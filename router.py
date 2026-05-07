"""
api/router.py
──────────────
Defines all HTTP routes for the BookWise recommender.

Endpoints
---------
GET  /health              → liveness check
GET  /recommend/{user_id} → top-N personalised book recommendations
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from api.schemas import RecommendationResponse, BookRecommendation, HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Simple liveness probe – confirms the API is up and the model is loaded."""
    return HealthResponse()


@router.get(
    "/recommend/{user_id}",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Get personalised book recommendations for a user",
)
def get_recommendations(
    user_id: int,
    top_n:   int = Query(default=10, ge=1, le=50, description="Number of books to return"),
):
    """
    Returns the top-N books most likely to interest **user_id**.

    * **Warm users** (seen during training) receive a hybrid blend of
      collaborative and content-based scores.
    * **Cold-start users** (never seen before) receive content-based
      recommendations derived from catalogue popularity and genre similarity.
    """
    from api.main import app_state   # lazy import to avoid circular deps

    hybrid  = app_state["hybrid"]
    books   = app_state["books"]
    inter   = app_state["interactions"]

    if hybrid is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded. Run train.py first.")

    is_cold = user_id not in app_state["known_users"]

    raw_recs = hybrid.recommend(
        user_id=user_id,
        interactions_df=inter,
        top_n=top_n,
    )

    book_index = books.set_index("book_id")
    enriched: list[BookRecommendation] = []

    for rec in raw_recs:
        bid = rec["book_id"]
        if bid not in book_index.index:
            continue
        row = book_index.loc[bid]
        enriched.append(BookRecommendation(
            book_id    = bid,
            title      = str(row["title"]),
            author     = str(row["author"]),
            genre      = str(row["genre"]),
            avg_rating = float(row.get("avg_rating", 0.0)),
            score      = rec["score"],
            source     = rec["source"],
        ))

    return RecommendationResponse(
        user_id         = user_id,
        is_cold_start   = is_cold,
        recommendations = enriched,
        total           = len(enriched),
    )
