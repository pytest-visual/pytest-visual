from typing import Any, List, Optional

from pydantic import BaseModel, Field, validator


class HashVectors_(BaseModel):
    Vectors: List[List[float]]
    ErrorThreshold: float


class ReferenceStatement(BaseModel):
    Type: str
    Text: Optional[str] = None
    Assets: List[str] = []
    Hash: str
    HashVectors: Optional[HashVectors_] = None
    Metadata: dict = {}

    @validator("Type")
    def type_must_be_in_allowed_values(cls, v: str) -> str:
        allowed_values = {"text", "images", "figure"}
        if v not in allowed_values:
            raise ValueError(f"Type must be one of {allowed_values}")
        return v

    @validator("Text", always=True)
    def content_for_text_type_only(cls, v: Optional[str], values: dict) -> Optional[str]:
        if values.get("Type") == "text" and v is None:
            raise ValueError("Text field is required for 'text' type")
        elif values.get("Type") != "text" and v is not None:
            raise ValueError("Text field should only be provided for 'text' type")
        return v


class MaterialStatement(BaseModel):
    Type: str
    Text: Optional[str] = None
    Assets: List[Any] = []
    Hash: str
    HashVectors: Optional[HashVectors_] = None
    Metadata: dict = {}
