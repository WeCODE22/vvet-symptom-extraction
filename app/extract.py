"""Turn a farmer's message into findings, then decide escalation in code."""

from __future__ import annotations

import json
import logging
import re

from app.llm import LLM
from app.prompt import build_system_prompt
from app.schemas import ExtractResponse, Finding, Value
from app.signs import (
    ESCALATE_SIGN_IDS,
    ESCALATE_VALUES,
    SIGN_IDS,
    SIGN_ORDER,
    VALUE_RANK,
    canonical_sign_id,
)

logger = logging.getLogger(__name__)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class ModelOutputError(Exception):
    """The model replied, but the reply was not a findings object we can read."""


def extract_message(message: str, llm: LLM) -> ExtractResponse:
    raw = llm.complete(build_system_prompt(), message)
    findings = validate_findings(raw, message)
    escalate, reason = decide_escalation(findings)
    return ExtractResponse(findings=findings, escalate=escalate, reason=reason)


def validate_findings(raw: str, message: str) -> list[Finding]:
    """Keep only known signs whose quote actually appears in the message.

    A bad item is dropped. A reply that is not JSON at all is an error, because
    an empty finding list would look like a healthy animal.
    """
    items = _findings_list(raw)
    kept: dict[str, Finding] = {}
    rejected = 0

    for item in items:
        finding = _one_finding(item, message)
        if finding is None:
            rejected += 1
            continue
        current = kept.get(finding.sign_id)
        if current is None or VALUE_RANK[finding.value] > VALUE_RANK[current.value]:
            kept[finding.sign_id] = finding

    if rejected:
        # Counts only. The farmer's words are not logged.
        logger.warning("dropped %s findings that failed validation; kept %s", rejected, len(kept))

    # The model tried to report signs and none of them could be checked.
    # An empty 200 would look like "no signs", so the caller gets an error
    # and can treat a failed extraction as urgent.
    if items and not kept:
        raise ModelOutputError("model findings could not be checked against the message")

    return [kept[sign_id] for sign_id in SIGN_IDS if sign_id in kept]


def decide_escalation(findings: list[Finding]) -> tuple[bool, str]:
    """Urgent when froth, inability to stand, or a cloudy eye is yes or not_sure.

    An explicit no does not escalate. Signs outside that trio never escalate.
    """
    triggered = [
        finding
        for finding in findings
        if finding.sign_id in ESCALATE_SIGN_IDS and finding.value in ESCALATE_VALUES
    ]
    if not triggered:
        return (
            False,
            "Do not escalate: froth_nose_mouth, unable_to_stand, and cloudy_eye "
            "were not reported as yes or not_sure.",
        )

    triggered.sort(key=lambda finding: SIGN_ORDER[finding.sign_id])
    detail = "; ".join(f"{finding.sign_id} is {finding.value}" for finding in triggered)
    return True, f"Escalate because {detail}."


def _findings_list(raw: str) -> list:
    data = _load_json_object(raw)
    findings = data.get("findings")
    if not isinstance(findings, list):
        raise ModelOutputError("model JSON must contain a findings list")
    return findings


def _load_json_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = _FENCE.sub("", text).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise ModelOutputError("model did not return JSON")
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ModelOutputError("model did not return JSON") from exc

    if not isinstance(data, dict):
        raise ModelOutputError("model JSON must be an object")
    return data


def _one_finding(item: object, message: str) -> Finding | None:
    if not isinstance(item, dict):
        return None

    sign_raw = item.get("sign_id")
    value_raw = item.get("value")
    quote_raw = item.get("quote")
    if not isinstance(sign_raw, str) or not isinstance(value_raw, str) or not isinstance(quote_raw, str):
        return None

    sign_id = canonical_sign_id(sign_raw)
    value = _normalise_value(value_raw)
    quote = quote_raw.strip()
    if sign_id is None or value is None:
        return None
    if not _quote_in_message(quote, message):
        return None
    return Finding(sign_id=sign_id, value=value, quote=quote)


def _normalise_value(raw: str) -> Value | None:
    value = raw.strip().casefold().replace("-", "_").replace(" ", "_")
    if value == "yes" or value == "no" or value == "not_sure":
        return value
    return None


def _quote_in_message(quote: str, message: str) -> bool:
    """The quote has to be the farmer's words, not the model's paraphrase."""
    folded_quote = " ".join(quote.split()).casefold()
    folded_message = " ".join(message.split()).casefold()
    return len(folded_quote) >= 2 and folded_quote in folded_message
