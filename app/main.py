"""HTTP entrypoint. The model proposes signs. This module does not decide urgency."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from app.extract import ModelOutputError, extract_message
from app.llm import LLM, LLMError, llm_from_env
from app.schemas import ExtractRequest, ExtractResponse

app = FastAPI(title="V-Vet symptom extraction", version="0.1.0")


def get_llm() -> LLM:
    return llm_from_env()


@app.post("/extract", response_model=ExtractResponse)
def extract(body: ExtractRequest, llm: LLM = Depends(get_llm)) -> ExtractResponse:
    try:
        return extract_message(body.message, llm)
    except ModelOutputError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
