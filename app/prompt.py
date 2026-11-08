"""Instructions for the model. Built from the sign catalog so the two cannot drift."""

from __future__ import annotations

from app.signs import SIGNS


def build_system_prompt() -> str:
    sign_lines = "\n".join(f"- {sign_id}: {text}" for sign_id, text in SIGNS)
    return (
        "You extract animal-health signs from one farmer's WhatsApp message.\n"
        "The message may be English, Shona, or a mix of both.\n"
        "Return a single JSON object and no other text:\n"
        '{"findings": [{"sign_id": "<id>", "value": "yes" | "no" | "not_sure", '
        '"quote": "<exact words from the message>"}]}\n'
        "\n"
        "Signs:\n"
        f"{sign_lines}\n"
        "\n"
        "Rules:\n"
        "- Include a sign only if the message mentions it. Leave out signs the message does not talk about.\n"
        "- Do not add a no for every sign that is missing.\n"
        "- value yes: the farmer says the sign is present.\n"
        "- value no: the farmer explicitly says that sign is absent.\n"
        "- value not_sure: the farmer is hedging about that sign "
        "(I think, maybe, not sure, seems, a bit).\n"
        "- quote must be copied from the message. Do not translate it and do not paraphrase it.\n"
        "- Lying down more than usual is weakness. unable_to_stand only if the animal "
        "cannot get up or stay on its feet.\n"
        "- Do not infer one sign from another. Fast breathing is not froth. Ticks are not a lump.\n"
        '- If the message mentions none of the signs, return {"findings": []}.\n'
    )
