from typing import List, Union, Optional, Any
from pydantic import BaseModel, Field, validator

class HashVectors_(BaseModel):
    Vectors: List[List[float]]
    ErrorThreshold: float


class ReferenceStatement(BaseModel):
    Type: str
    Content: Optional[str] = None
    Assets: List[str] = []
    Hash: str
    HashVectors: Optional[HashVectors_] = None
    Metadata: dict = {}

    @validator('Type')
    def type_must_be_in_allowed_values(cls, v):
        allowed_values = {"text", "images", "plot"}
        if v not in allowed_values:
            raise ValueError(f"Type must be one of {allowed_values}")
        return v
    
    @validator('Content', always=True)
    def content_for_text_type_only(cls, v, values):
        if values.get("Type") == "text" and v is None:
            raise ValueError("Content is required for 'text' type")
        elif values.get("Type") != "text" and v is not None:
            raise ValueError("Content should only be provided for 'text' type")
        return v


class MaterialStatement(BaseModel):
    Type: str
    Content: Optional[str] = None
    Assets: List[Any] = []
    Hash: str
    HashVectors: Optional[HashVectors_] = None
    Metadata: dict = {}
