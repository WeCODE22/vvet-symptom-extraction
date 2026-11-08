"""The fixed sign list. The model is not allowed to invent ids outside this file."""

from __future__ import annotations

SIGNS: tuple[tuple[str, str], ...] = (
    ("reduced_appetite", "eating less, or has stopped eating or grazing"),
    ("weakness", "dull, weak, lying down more than usual"),
    (
        "swelling",
        "a lump below the ear, under the jaw, or in front of the shoulder "
        "(farmers may call it mbandabvu)",
    ),
    ("rapid_breathing", "breathing fast while resting"),
    ("cloudy_eye", "the eye looks white, grey or milky"),
    ("froth_nose_mouth", "foam or froth from the nose or mouth"),
    ("red_urine", "urine that is red, brown, or the colour of strong tea"),
    ("unable_to_stand", "cannot get up or stay standing"),
    ("ticks", "ticks can be seen on the animal"),
)

SIGN_IDS: tuple[str, ...] = tuple(sign_id for sign_id, _ in SIGNS)
SIGN_ORDER: dict[str, int] = {sign_id: index for index, sign_id in enumerate(SIGN_IDS)}

# Membership is checked in code. The model never decides this.
ESCALATE_SIGN_IDS: frozenset[str] = frozenset(
    {"froth_nose_mouth", "unable_to_stand", "cloudy_eye"}
)
ESCALATE_VALUES: frozenset[str] = frozenset({"yes", "not_sure"})

# When the model reports the same sign twice, keep the more serious value.
VALUE_RANK: dict[str, int] = {"no": 0, "not_sure": 1, "yes": 2}

_SIGN_BY_FOLD: dict[str, str] = {sign_id.casefold(): sign_id for sign_id in SIGN_IDS}


def canonical_sign_id(raw: str) -> str | None:
    """Return the official id, or None if this is not one of our signs."""
    return _SIGN_BY_FOLD.get(raw.strip().casefold())
