"""
api/schemas.py
───────────────
Pydantic models that define the request and response shapes
for the BookWise REST API.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal


class BookRecommendation(BaseModel):
    book_id:     int   = Field(..., description="Unique identifier of the recommended book")
    title:       str   = Field(..., description="Book title")
    author:      str   = Field(..., description="Author name")
    genre:       str   = Field(..., description="Genre category")
    avg_rating:  float = Field(..., description="Average community rating (1–5)")
    score:       float = Field(..., description="Recommendation relevance score (0–1)")
    source:      Literal["hybrid", "content_based"] = Field(
        ..., description="Which filtering strategy produced this result"
    )


class RecommendationResponse(BaseModel):
    user_id:         int                     = Field(..., description="Requested user ID")
    is_cold_start:   bool                    = Field(..., description="True when user has no history")
    recommendations: list[BookRecommendation] = Field(..., description="Ranked list of top-N books")
    total:           int                     = Field(..., description="Number of recommendations returned")


class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str = "1.0.0"
    message: str = "BookWise Recommender API is running"
