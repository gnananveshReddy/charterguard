"""Policy as a file: build a `GuardrailConfig` from TOML (or JSON).

Enforcement is a product decision that changes with rollout phase, so it
belongs in a reviewed file rather than in code::

    # guardrails.toml
    preset = "default"            # default | strict | report_only
    policy_version = "acme-2026.11"

    [modes]                       # report | flag | block, per category
    all = "flag"                  # optional: applied first, then overrides
    injection = "block"
    secret = "block"

    [flag_at]
    pii = 0.4

    [block_at]
    injection = 0.75

    [budget]
    max_chars = 100_000

    [model]                       # the default tier; per layer below
    enabled = true
    fail_closed = true

    [model_by_layer.input]
    escalate_clean = true

    [redaction]
    categories = ["secret", "pii"]

    [volume]
    max_rows = 2_000

Loading is strict on purpose. An unknown key, a wrong type, or a threshold
outside [0, 1] raises `PolicyError` naming the exact key: a misspelt
``injectoin = "block"`` that silently did nothing would be a security bug.
Only the stdlib is used (`tomllib`, Python 3.11+).
"""
from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from .config import GuardrailConfig, ModelTier, RedactionConfig
from .types import Category, Layer, Mode

__all__ = ["PolicyError", "load_policy", "policy_from_mapping", "policy_to_mapping"]

_PRESETS = {
    "default": GuardrailConfig,
    "strict": GuardrailConfig.strict,
    "report_only": GuardrailConfig.report_only,
}
_FLAGS = (
    "enable_encoded_scan",
    "enable_obfuscation_scan",
    "dampen_quoted_mentions",
    "short_circuit_on_critical",
)
_TOP_LEVEL = frozenset(
    {"preset", "policy_version", "modes", "flag_at", "block_at", "budget", "model", "model_by_layer",
     "redaction", "volume", *_FLAGS}
)


class PolicyError(ValueError):
    """A policy file that cannot be applied exactly as written."""


# ----------------------------------------------------------------- parsing
def _enum(kind: type[Any], value: object, where: str) -> Any:
    try:
        return kind(str(value).strip().lower())
    except ValueError:
        allowed = ", ".join(m.value for m in kind)
        raise PolicyError(f"{where}: {value!r} is not one of {allowed}") from None


def _table(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyError(f"{where}: expected a table, got {type(value).__name__}")
    return value


def _coerce(value: object, expected: object, where: str) -> object:
    """Check `value` against the type of the dataclass default `expected`."""
    if isinstance(expected, bool):
        if not isinstance(value, bool):
            raise PolicyError(f"{where}: expected true/false, got {value!r}")
        return value
    if isinstance(expected, int):
        # bool is an int subclass; `max_chars = true` must not mean 1.
        if isinstance(value, bool) or not isinstance(value, int):
            raise PolicyError(f"{where}: expected an integer, got {value!r}")
        if value < 0:
            raise PolicyError(f"{where}: must not be negative")
        return value
    if isinstance(expected, float):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise PolicyError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if isinstance(expected, str):
        if not isinstance(value, str):
            raise PolicyError(f"{where}: expected a string, got {value!r}")
        return value
    raise PolicyError(f"{where}: not settable from a policy file")


def _dataclass_from(base: Any, table: Mapping[str, Any], where: str, *, skip: frozenset[str] = frozenset()) -> Any:
    known = {f.name for f in fields(base)} - skip
    changes: dict[str, object] = {}
    for key, value in table.items():
        if key not in known:
            raise PolicyError(f"{where}.{key}: unknown key (known: {', '.join(sorted(known))})")
        changes[key] = _coerce(value, getattr(base, key), f"{where}.{key}")
    return replace(base, **changes)


def _unit_interval(value: object, where: str) -> float:
    number = _coerce(value, 0.0, where)
    assert isinstance(number, float)  # noqa: S101 -- narrowing for mypy; _coerce guarantees it
    if not 0.0 <= number <= 1.0:
        raise PolicyError(f"{where}: {number} is outside [0, 1]")
    return number


def _per_category(table: Mapping[str, Any], where: str) -> dict[Category, Any]:
    return {_enum(Category, key, f"{where} key"): value for key, value in table.items()}


def _tier(base: ModelTier, table: Mapping[str, Any], where: str) -> ModelTier:
    tier: ModelTier = _dataclass_from(base, table, where)
    for name in ("escalate_low", "escalate_high", "clean_sample_rate", "weight"):
        _unit_interval(getattr(tier, name), f"{where}.{name}")
    if tier.escalate_low > tier.escalate_high:
        raise PolicyError(f"{where}: escalate_low must not exceed escalate_high")
    return tier


def policy_from_mapping(data: Mapping[str, Any], *, source: str = "policy") -> GuardrailConfig:
    """Build a config from an already-parsed mapping (TOML, JSON, or a dict)."""
    unknown = set(data) - _TOP_LEVEL
    if unknown:
        raise PolicyError(f"{source}: unknown key(s) {', '.join(sorted(unknown))}")

    preset = str(data.get("preset", "default")).strip().lower()
    if preset not in _PRESETS:
        raise PolicyError(f"{source}.preset: {preset!r} is not one of {', '.join(_PRESETS)}")
    cfg = _PRESETS[preset]()

    changes: dict[str, Any] = {}
    if "policy_version" in data:
        changes["policy_version"] = _coerce(data["policy_version"], "", f"{source}.policy_version")
    for flag in _FLAGS:
        if flag in data:
            changes[flag] = _coerce(data[flag], True, f"{source}.{flag}")

    if "modes" in data:
        table = dict(_table(data["modes"], f"{source}.modes"))
        modes = dict(cfg.modes)
        if "all" in table:
            everything = _enum(Mode, table.pop("all"), f"{source}.modes.all")
            modes = {c: everything for c in Category}
        for category, value in _per_category(table, f"{source}.modes").items():
            modes[category] = _enum(Mode, value, f"{source}.modes.{category.value}")
        changes["modes"] = modes

    for name in ("flag_at", "block_at"):
        if name in data:
            thresholds = dict(getattr(cfg, name))
            for category, value in _per_category(_table(data[name], f"{source}.{name}"), f"{source}.{name}").items():
                thresholds[category] = _unit_interval(value, f"{source}.{name}.{category.value}")
            changes[name] = thresholds

    if "budget" in data:
        changes["budget"] = _dataclass_from(cfg.budget, _table(data["budget"], f"{source}.budget"),
                                            f"{source}.budget")
    if "volume" in data:
        changes["volume"] = _dataclass_from(cfg.volume, _table(data["volume"], f"{source}.volume"),
                                            f"{source}.volume")
    if "redaction" in data:
        table = dict(_table(data["redaction"], f"{source}.redaction"))
        redaction: RedactionConfig = cfg.redaction
        if "categories" in table:
            raw = table.pop("categories")
            if not isinstance(raw, list):
                raise PolicyError(f"{source}.redaction.categories: expected a list")
            redaction = replace(
                redaction,
                redact_categories=tuple(_enum(Category, c, f"{source}.redaction.categories") for c in raw),
            )
        redaction = _dataclass_from(redaction, table, f"{source}.redaction", skip=frozenset({"redact_categories"}))
        try:
            redaction.placeholder.format(kind="X")
        except (KeyError, IndexError, ValueError) as exc:
            raise PolicyError(f"{source}.redaction.placeholder: only {{kind}} may be used ({exc})") from None
        changes["redaction"] = redaction

    model = cfg.model
    if "model" in data:
        model = _tier(cfg.model, _table(data["model"], f"{source}.model"), f"{source}.model")
        changes["model"] = model
    if "model_by_layer" in data:
        by_layer = dict(cfg.model_by_layer)
        for key, table in _table(data["model_by_layer"], f"{source}.model_by_layer").items():
            layer = _enum(Layer, key, f"{source}.model_by_layer key")
            where = f"{source}.model_by_layer.{layer.value}"
            by_layer[layer] = _tier(by_layer.get(layer, model), _table(table, where), where)
        changes["model_by_layer"] = by_layer

    cfg = replace(cfg, **changes)
    for category in Category:
        if cfg.flag_at[category] > cfg.block_at[category]:
            raise PolicyError(
                f"{source}: flag_at.{category.value} ({cfg.flag_at[category]}) exceeds "
                f"block_at.{category.value} ({cfg.block_at[category]})"
            )
    return cfg


def load_policy(path: str | Path) -> GuardrailConfig:
    """Load a ``.toml`` (or ``.json``) policy file into a `GuardrailConfig`."""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise PolicyError(f"{p}: cannot read ({exc.strerror or exc})") from exc
    try:
        data = json.loads(raw) if p.suffix.lower() == ".json" else tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PolicyError(f"{p}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise PolicyError(f"{p}: top level must be a table")
    return policy_from_mapping(data, source=p.name)


def policy_to_mapping(cfg: GuardrailConfig) -> dict[str, Any]:
    """The effective policy, as plain data (for review and for `charterguard policy`)."""

    def plain(obj: Any) -> Any:
        return {f.name: getattr(obj, f.name) for f in fields(obj)}

    redaction = plain(cfg.redaction)
    redaction["categories"] = [c.value for c in redaction.pop("redact_categories")]
    return {
        "policy_version": cfg.policy_version,
        **{flag: getattr(cfg, flag) for flag in _FLAGS},
        "modes": {c.value: cfg.modes[c].value for c in Category},
        "flag_at": {c.value: cfg.flag_at[c] for c in Category},
        "block_at": {c.value: cfg.block_at[c] for c in Category},
        "budget": plain(cfg.budget),
        "model": plain(cfg.model),
        "model_by_layer": {layer.value: plain(t) for layer, t in cfg.model_by_layer.items()},
        "redaction": redaction,
        "volume": plain(cfg.volume),
    }

