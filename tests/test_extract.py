"""Escalation and validation, with the model replaced by a fake."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.extract import decide_escalation, extract_message
from app.llm import LLMError
from app.main import app, get_llm
from app.prompt import build_system_prompt
from app.schemas import Finding
from app.signs import ESCALATE_SIGN_IDS, SIGN_IDS

EATING = "My cow has not been eating since yesterday and she is breathing very fast."
FOAM = "There's foam coming from her nose and she can't get up."
CLOUDY = "I think the eyes look a bit cloudy but I'm not sure. She has mbandabvu under the ear."
TICKS = "The calf is fine, just lots of ticks on the ears."
SHONA = "Mombe yangu haisi kudya, and her urine is dark like tea."


class FakeLLM:
    def __init__(self, content: str | Exception) -> None:
        self.content = content
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if isinstance(self.content, Exception):
            raise self.content
        return self.content


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def post(message: str, model_text: str) -> tuple[TestClient, FakeLLM]:
    fake = FakeLLM(model_text)
    app.dependency_overrides[get_llm] = lambda: fake
    return TestClient(app), fake


def findings(*items: dict) -> str:
    return json.dumps({"findings": list(items)})


def test_prompt_lists_every_sign_and_does_not_ask_the_model_to_escalate():
    prompt = build_system_prompt()
    for sign_id in SIGN_IDS:
        assert sign_id in prompt
    assert "mbandabvu" in prompt
    assert "escalate" not in prompt.casefold()
    assert ESCALATE_SIGN_IDS <= set(SIGN_IDS)


def test_appetite_and_fast_breathing_do_not_escalate():
    client, fake = post(
        EATING,
        findings(
            {"sign_id": "reduced_appetite", "value": "yes", "quote": "has not been eating"},
            {"sign_id": "rapid_breathing", "value": "yes", "quote": "breathing very fast"},
        ),
    )
    response = client.post("/extract", json={"message": EATING})
    assert response.status_code == 200
    body = response.json()
    assert [item["sign_id"] for item in body["findings"]] == [
        "reduced_appetite",
        "rapid_breathing",
    ]
    assert body["escalate"] is False
    assert fake.calls[0][1] == EATING


def test_foam_and_unable_to_stand_escalates():
    client, _ = post(
        FOAM,
        findings(
            {"sign_id": "unable_to_stand", "value": "yes", "quote": "can't get up"},
            {"sign_id": "froth_nose_mouth", "value": "yes", "quote": "foam coming from her nose"},
        ),
    )
    body = client.post("/extract", json={"message": FOAM}).json()
    assert body["escalate"] is True
    assert body["reason"] == (
        "Escalate because froth_nose_mouth is yes; unable_to_stand is yes."
    )


def test_unsure_cloudy_eye_escalates():
    client, _ = post(
        CLOUDY,
        findings(
            {"sign_id": "cloudy_eye", "value": "not_sure", "quote": "eyes look a bit cloudy"},
            {"sign_id": "swelling", "value": "yes", "quote": "mbandabvu under the ear"},
        ),
    )
    body = client.post("/extract", json={"message": CLOUDY}).json()
    assert body["escalate"] is True
    assert body["reason"] == "Escalate because cloudy_eye is not_sure."
    assert body["findings"][0]["sign_id"] == "swelling"
    assert body["findings"][1]["value"] == "not_sure"


def test_ticks_only_do_not_escalate():
    client, _ = post(
        TICKS,
        findings({"sign_id": "ticks", "value": "yes", "quote": "lots of ticks"}),
    )
    body = client.post("/extract", json={"message": TICKS}).json()
    assert body["findings"] == [
        {"sign_id": "ticks", "value": "yes", "quote": "lots of ticks"}
    ]
    assert body["escalate"] is False


def test_shona_appetite_and_dark_urine_do_not_escalate():
    client, _ = post(
        SHONA,
        findings(
            {"sign_id": "reduced_appetite", "value": "yes", "quote": "haisi kudya"},
            {"sign_id": "red_urine", "value": "yes", "quote": "urine is dark like tea"},
        ),
    )
    body = client.post("/extract", json={"message": SHONA}).json()
    assert [item["sign_id"] for item in body["findings"]] == [
        "reduced_appetite",
        "red_urine",
    ]
    assert body["escalate"] is False


def test_explicit_no_on_an_urgent_sign_does_not_escalate():
    message = "Her eyes are clear, there is no foam, and she is up and walking."
    client, _ = post(
        message,
        findings(
            {"sign_id": "cloudy_eye", "value": "no", "quote": "eyes are clear"},
            {"sign_id": "froth_nose_mouth", "value": "no", "quote": "no foam"},
            {"sign_id": "unable_to_stand", "value": "no", "quote": "up and walking"},
        ),
    )
    body = client.post("/extract", json={"message": message}).json()
    assert body["escalate"] is False
    assert len(body["findings"]) == 3


def test_weakness_does_not_escalate():
    message = "The ox is dull and lying down more than usual."
    result = extract_message(
        message,
        FakeLLM(
            findings(
                {"sign_id": "weakness", "value": "yes", "quote": "lying down more than usual"}
            )
        ),
    )
    assert result.escalate is False
    assert result.findings[0].sign_id == "weakness"


def test_model_cannot_turn_escalation_off():
    message = "Foam is coming from the nose."
    raw = json.dumps(
        {
            "escalate": False,
            "reason": "the model says this can wait",
            "findings": [
                {"sign_id": "froth_nose_mouth", "value": "yes", "quote": "Foam is coming"}
            ],
        }
    )
    result = extract_message(message, FakeLLM(raw))
    assert result.escalate is True
    assert result.reason == "Escalate because froth_nose_mouth is yes."


def test_unknown_sign_is_dropped_and_a_real_one_is_kept():
    message = "Lots of ticks on the ears."
    result = extract_message(
        message,
        FakeLLM(
            findings(
                {"sign_id": "fever", "value": "yes", "quote": "ticks"},
                {"sign_id": "ticks", "value": "YES", "quote": "ticks"},
            )
        ),
    )
    assert [finding.sign_id for finding in result.findings] == ["ticks"]
    assert result.findings[0].value == "yes"


def test_paraphrased_quote_is_rejected():
    message = "My cow is not eating."
    client, _ = post(
        message,
        findings(
            {
                "sign_id": "reduced_appetite",
                "value": "yes",
                "quote": "the animal has stopped grazing",
            }
        ),
    )
    response = client.post("/extract", json={"message": message})
    assert response.status_code == 502
    assert response.json()["detail"] == "model findings could not be checked against the message"


def test_empty_findings_do_not_escalate():
    message = "Good morning, please call me later."
    result = extract_message(message, FakeLLM('{"findings": []}'))
    assert result.findings == []
    assert result.escalate is False


def test_fenced_and_wrapped_json_is_accepted():
    message = "She has ticks on the ears."
    wrapped = '```json\n{"findings": [{"sign_id": "ticks", "value": "yes", "quote": "ticks"}]}\n```'
    result = extract_message(message, FakeLLM(wrapped))
    assert result.findings[0].sign_id == "ticks"

    prose = 'Sure. {"findings": [{"sign_id": "ticks", "value": "yes", "quote": "ticks"}]} Done.'
    result = extract_message(message, FakeLLM(prose))
    assert result.escalate is False


def test_hedging_value_with_a_space_is_not_sure():
    message = "I think the eye looks a bit cloudy."
    result = extract_message(
        message,
        FakeLLM(
            findings(
                {"sign_id": "Cloudy_Eye", "value": "Not Sure", "quote": "a bit cloudy"}
            )
        ),
    )
    assert result.findings[0].sign_id == "cloudy_eye"
    assert result.findings[0].value == "not_sure"
    assert result.escalate is True


def test_duplicate_sign_keeps_the_more_serious_value():
    message = "She is not eating. I think she is not eating."
    result = extract_message(
        message,
        FakeLLM(
            findings(
                {"sign_id": "reduced_appetite", "value": "not_sure", "quote": "I think she is not eating"},
                {"sign_id": "reduced_appetite", "value": "yes", "quote": "She is not eating"},
            )
        ),
    )
    assert len(result.findings) == 1
    assert result.findings[0].value == "yes"
    assert result.findings[0].quote == "She is not eating"


def test_invalid_value_is_dropped():
    message = "There are ticks on the calf."
    result = extract_message(
        message,
        FakeLLM(
            findings(
                {"sign_id": "ticks", "value": "maybe", "quote": "ticks"},
                {"sign_id": "ticks", "value": "yes", "quote": "ticks"},
            )
        ),
    )
    assert result.findings[0].value == "yes"


def test_unreadable_model_output_is_502():
    client, _ = post("The cow is sick.", "I could not tell.")
    response = client.post("/extract", json={"message": "The cow is sick."})
    assert response.status_code == 502


def test_provider_failure_is_502():
    fake = FakeLLM(LLMError("LLM_API_KEY is not set"))
    app.dependency_overrides[get_llm] = lambda: fake
    response = TestClient(app).post("/extract", json={"message": "The cow is sick."})
    assert response.status_code == 502
    assert "LLM_API_KEY" in response.json()["detail"]


def test_blank_message_is_422():
    response = TestClient(app).post("/extract", json={"message": "   "})
    assert response.status_code == 422


def test_decide_escalation_ignores_non_urgent_yes():
    findings_list = [
        Finding(sign_id="red_urine", value="yes", quote="dark like tea"),
        Finding(sign_id="cloudy_eye", value="no", quote="eyes are clear"),
    ]
    escalate, reason = decide_escalation(findings_list)
    assert escalate is False
    assert "not reported as yes or not_sure" in reason
