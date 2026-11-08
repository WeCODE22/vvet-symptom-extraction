"""Request and response bodies for POST /extract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Value = Literal["yes", "no", "not_sure"]

# A WhatsApp note is short. The cap also bounds the prompt we send to the model.
MAX_MESSAGE_CHARS = 4000


class ExtractRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)

    @field_validator("message")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message is empty")
        return cleaned


class Finding(BaseModel):
    sign_id: str
    value: Value
    quote: str


class ExtractResponse(BaseModel):
    findings: list[Finding]
    escalate: bool
    reason: str
