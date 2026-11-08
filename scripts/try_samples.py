"""Send the brief's sample messages to a live model.

Pytest does not run this file. It needs LLM_API_KEY.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.extract import extract_message
from app.llm import llm_from_env

SAMPLES = (
    "My cow has not been eating since yesterday and she is breathing very fast.",
    "There's foam coming from her nose and she can't get up.",
    "I think the eyes look a bit cloudy but I'm not sure. She has mbandabvu under the ear.",
    "The calf is fine, just lots of ticks on the ears.",
    "Mombe yangu haisi kudya, and her urine is dark like tea.",
)


def main() -> None:
    llm = llm_from_env()
    if not llm.api_key:
        raise SystemExit("Set LLM_API_KEY in the environment or in a .env file.")

    for message in SAMPLES:
        result = extract_message(message, llm)
        print(message)
        print(result.model_dump_json(indent=2))
        print()


if __name__ == "__main__":
    main()
