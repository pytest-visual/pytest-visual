from typing import List, Literal, Optional, Union

import numpy as np
from plotly.graph_objs import Figure
from pydantic import BaseModel, Field, validator


class HashVector_(BaseModel):
    Vector: List[float]
    ErrorThreshold: float


class OnDiskStatement(BaseModel):
    Type: Literal["text", "image", "figure"]
    Text: Optional[str] = None
    Asset: Optional[str] = None
    Hash: str
    HashVector: Optional[HashVector_] = None
    Metadata: dict = {}

    @validator("Type")
    def type_must_be_in_allowed_values(cls, v: str) -> str:
        allowed_values = {"text", "image", "figure"}
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


class Statement(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    Type: Literal["text", "image", "figure"]
    Text: Optional[str] = None
    Asset: Optional[Union[np.ndarray, Figure]] = None
    Hash: str
    HashVector: Optional[HashVector_] = None
    Metadata: dict = {}
