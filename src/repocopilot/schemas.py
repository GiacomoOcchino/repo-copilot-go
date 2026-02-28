from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List

class Source(BaseModel):
    ref: str                 # es: S1, S2...
    chunk_id: str            # id interno in Chroma
    path: str
    excerpt: str             # snippet breve

class Citation(BaseModel):
    ref: str                 # deve essere uno tra S1..Sk
    quote: str = Field(..., description="Estratto breve (<= 200 char) che giustifica l'affermazione.")

class AnswerWithCitations(BaseModel):
    answer_md: str
    citations: List[Citation]
    confidence: str = Field(..., description="alta|media|bassa")
    open_questions: List[str] = []