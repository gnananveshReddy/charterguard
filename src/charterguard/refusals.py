"""Deterministic refusal copy.

Refusals are harness-authored, not model-authored: a blocked turn must never be
handed back to the model to phrase, because that reintroduces the injected text
into the context. Copy is boring on purpose -- no lecturing, no restating the
attack, no hinting at the bypass.
"""
from __future__ import annotations

from .types import Category, Verdict

_SELF_HARM = (
    "I can't help with that, and I don't want to leave you without support. If you're "
    "in distress, please reach out to a local crisis line or someone you trust right now. "
    "In India, Tele-MANAS is available 24x7 on 14416."
)

_REFUSALS: dict[Category, str] = {
    Category.INJECTION: (
        "I found instructions embedded in the content I was given that try to change how I "
        "operate. I've ignored them and stopped here. If this content is expected, the "
        "source should be reviewed."
    ),
    Category.JAILBREAK: (
        "I can't take that framing. Ask me the underlying question directly and I'll help "
        "with whatever part of it is legitimate."
    ),
    Category.HARMFUL: "I can't help with that request.",
    Category.SELF_HARM: _SELF_HARM,
    Category.MALWARE: (
        "I can't help build that. If this is defensive or research work, I can discuss "
        "detection, hardening, and mitigations instead."
    ),
    Category.COMMAND: (
        "That action is destructive or privileged, so I won't run or draft it. If it's "
        "genuinely needed, it has to go through a reviewed, approved change."
    ),
    Category.RESTRICTED_TOPIC: (
        "That topic is restricted for this assistant. Please raise it with the owning team."
    ),
    Category.SECRET: (
        "My response contained what looked like a credential, so I've withheld it. "
        "Secrets should be fetched from the vault at runtime, never passed through me."
    ),
    Category.PII: (
        "My response contained personal data that I'm not permitted to return in this "
        "context. I've withheld it."
    ),
    Category.EXFIL: (
        "The content I was given tried to get data sent to an external destination. "
        "I've ignored it and stopped."
    ),
    Category.VOLUME: (
        "This session has reached its data-volume limit, so I've stopped retrieving. "
        "Start a new session or narrow the request."
    ),
    Category.UNGROUNDED: (
        "I couldn't ground that answer in the sources available to me, so I'd rather not "
        "give it. Let me try a different search, or point me at the right document."
    ),
    Category.OUT_OF_SCOPE: (
        "That's outside what I'm set up to handle, so I'd be guessing. Ask me something "
        "in my area, or route this to the team that owns it."
    ),
}

_DEFAULT = "I can't proceed with that request."


def refusal_for(verdict: Verdict, *, charter_refusal: str | None = None) -> str:
    top = verdict.top
    if top is None:
        return _DEFAULT
    if top.category is Category.OUT_OF_SCOPE and charter_refusal:
        return charter_refusal
    return _REFUSALS.get(top.category, _DEFAULT)
