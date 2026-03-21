from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List


class Source(BaseModel):
    ref: str  # es: S1, S2...
    chunk_id: str  # id interno in Chroma
    path: str
    excerpt: str  # snippet breve


class Citation(BaseModel):
    ref: str  # deve essere uno tra S1..Sk
    quote: str = Field(
        ..., description="Estratto breve (<= 200 char) che giustifica l'affermazione."
    )


class AnswerWithCitations(BaseModel):
    answer_md: str
    citations: List[Citation]
    confidence: str = Field(..., description="alta|media|bassa")
    open_questions: List[str] = []


class RiskItem(BaseModel):
    severity: str = Field(..., description="alta|media|bassa")
    description: str


class PrNotes(BaseModel):
    title: str
    summary: List[str]
    files_changed: List[str]
    risks: List[RiskItem]
    suggested_tests: List[str]
    rollout_plan: List[str]
    rollback_plan: List[str]
    open_questions: List[str]
    citations: List[Citation]  # riusa Citation (ref, quote)
    confidence: str = Field(..., description="alta|media|bassa")
