"""Rule packs: domain rules shipped as reviewed TOML files.

    pack = load_rule_pack("my-pack.toml")          # or bundled_rule_pack("agent-tools")
    pipeline = GuardrailPipeline(config, rule_packs=[pack])

A pack is data, not code, and it is held to a stricter bar than the built-in
rules because whoever wrote it is not whoever reviewed this library:

* **Every rule proves itself.** Each rule lists ``examples`` it must match
  and may list ``counterexamples`` it must not. They are checked through the
  real scanner (normalization, prefilter, validators) every time the pack
  loads, so a pack that stops doing what it claims fails loudly instead of
  silently matching nothing.
* **Patterns are bounded.** Python's ``re`` cannot be interrupted, so a
  pattern that backtracks catastrophically would hang the request it scans.
  Pack patterns may not nest quantifiers, may repeat a group at most
  ``MAX_GROUP_REPEAT`` times, and must have a bounded maximum width that fits
  the prefilter window (``\\s+`` counts as one character: the normalizer
  collapses whitespace runs). No capturing groups, backreferences, inline
  flags or conditionals.
* **Namespaced ids.** Rule ids start with ``<pack name>.``, the name cannot
  shadow a built-in prefix (``inj``, ``pii``, ...), and duplicate ids across
  packs are an error, so a pack can never replace a built-in rule.

Patterns run on normalized text: lowercase, confusables folded, whitespace
collapsed. Write them in lowercase.

File format::

    [pack]
    name = "agent-tools"
    version = "1.0.0"
    description = "Dangerous instructions aimed at coding agents"

    [[rule]]
    id = "agent-tools.git.force-push"
    category = "command"          # a charterguard Category value
    severity = "high"             # low | medium | high | critical
    description = "Force-push to a protected branch"
    pattern = 'git\\s+push\\s+(?:--force|-f)\\s+\\S{1,40}\\s+(?:main|master)\\b'
    layers = ["untrusted", "tool_output"]   # default: all four content layers
    validate = "luhn"             # optional: a built-in validator name
    examples = ["run git push --force origin main now"]
    counterexamples = ["never git push --force to main"]
"""
from __future__ import annotations

import importlib
import re
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .normalize import normalize
from .patterns import RULES, PatternError, Rule
from .prefilter import WINDOW
from .types import Category, Layer, Severity

__all__ = [
    "RulePack",
    "RulePackError",
    "load_rule_pack",
    "rule_pack_from_mapping",
    "combine_rules",
    "bundled_rule_pack",
    "bundled_rule_packs",
    "MAX_GROUP_REPEAT",
]

_sre: Any = importlib.import_module("re._parser")

MAX_PATTERN_CHARS = 1000
MAX_RULES = 200
MAX_EXAMPLES = 20
MAX_EXAMPLE_CHARS = 2000
MAX_GROUP_REPEAT = 4
MIN_WIDTH = 3

_NAME = re.compile(r"[a-z][a-z0-9-]{1,39}")
_VERSION = re.compile(r"\d{1,4}\.\d{1,4}\.\d{1,6}")
_ID_TAIL = re.compile(r"[a-z0-9_][a-z0-9_-]{0,40}(?:\.[a-z0-9_][a-z0-9_-]{0,40}){0,5}")
_RESERVED = frozenset(
    {r.id.split(".", 1)[0] for r in RULES}
    | {"envelope", "volume", "scope", "model", "charterguard", "structural", "encoded", "obfuscation"}
)
_SEVERITIES = {s.name.lower(): s for s in Severity}
_LAYERS = {
    "input": Layer.INPUT,
    "untrusted": Layer.UNTRUSTED,
    "tool_output": Layer.TOOL_OUTPUT,
    "output": Layer.OUTPUT,
}
_CONTENT_LAYERS = (Layer.INPUT, Layer.UNTRUSTED, Layer.TOOL_OUTPUT, Layer.OUTPUT)
_PACK_KEYS = frozenset({"name", "version", "description", "homepage", "license", "authors"})
_RULE_KEYS = frozenset(
    {"id", "category", "severity", "description", "pattern", "layers", "validate", "examples", "counterexamples"}
)
_SIMPLE_OPS = frozenset({_sre.LITERAL, _sre.NOT_LITERAL, _sre.IN, _sre.ANY, _sre.CATEGORY})
_REPEATS = frozenset(
    op for op in (_sre.MAX_REPEAT, _sre.MIN_REPEAT, getattr(_sre, "POSSESSIVE_REPEAT", None)) if op is not None
)
_FORBIDDEN_GLOBAL_FLAGS = re.IGNORECASE | re.MULTILINE | re.DOTALL | re.VERBOSE | re.ASCII | re.LOCALE


class RulePackError(ValueError):
    """A rule pack that cannot be loaded exactly as written."""


@dataclass(frozen=True, slots=True)
class RulePack:
    name: str
    version: str
    description: str
    rules: tuple[Rule, ...]
    source: str = "<mapping>"


# ------------------------------------------------------------ pattern safety
_UNBOUNDED = 1 << 30


def _is_whitespace_item(sub: Any) -> bool:
    """``\\s`` or ``[\\s]`` or a literal space, alone.

    The normalizer collapses every whitespace run to one character, so on
    scanned text a repeat of exactly this item matches at most one character.
    Decided on the PARSE TREE, never by rewriting the pattern source: a text
    rewrite of ``\\s+`` also rewrites ``\\\\s+`` (a literal backslash, then an
    unbounded ``s+``) and hides it from the checks.
    """
    if len(sub) != 1:
        return False
    op, av = sub[0]
    if op == _sre.LITERAL:
        return bool(av == 0x20)
    return bool(op == _sre.IN and list(av) == [(_sre.CATEGORY, _sre.CATEGORY_SPACE)])


def _max_width(items: Any) -> int:
    """Longest text ``items`` can match on normalized input."""
    total = 0
    for op, av in items:
        if op in _SIMPLE_OPS:
            total += 1
        elif op == _sre.SUBPATTERN:
            total += _max_width(av[3])
        elif op == _sre.BRANCH:
            total += max(_max_width(b) for b in av[1])
        elif op in _REPEATS:
            low, high, sub = av
            if _is_whitespace_item(sub):
                total += 1 if high >= 1 else 0
            else:
                inner = _max_width(sub)
                total += _UNBOUNDED if (high >= _sre.MAXREPEAT and inner) else high * inner
        elif op == getattr(_sre, "ATOMIC_GROUP", object()):
            total += _max_width(av)
        # AT (anchors) and lookarounds consume nothing.
        total = min(total, _UNBOUNDED)
    return total


def _walk(items: Any, where: str, *, in_repeat: bool) -> None:
    for op, av in items:
        if op == _sre.SUBPATTERN:
            group, add_flags, del_flags, sub = av
            if group is not None:
                raise RulePackError(f"{where}: capturing group; use (?: ... )")
            if add_flags or del_flags:
                raise RulePackError(f"{where}: inline flags are not allowed")
            _walk(sub, where, in_repeat=in_repeat)
        elif op in _REPEATS:
            low, high, sub = av
            if _is_whitespace_item(sub):
                continue  # at most one character on normalized text: no backtracking
            if high >= _sre.MAXREPEAT:
                # Anywhere -- including inside a lookaround, which the overall
                # width does not count: "x(?=a*a*a*b)" backtracks polynomially.
                raise RulePackError(
                    f"{where}: unbounded repeat; use a bounded one such as \\S{{1,40}} instead of * or +"
                )
            if in_repeat and high > 1:
                raise RulePackError(
                    f"{where}: nested quantifier; unroll the inner repeat "
                    "(nested repeats can backtrack catastrophically)"
                )
            simple = len(sub) == 1 and sub[0][0] in _SIMPLE_OPS
            if not simple and high > MAX_GROUP_REPEAT:
                raise RulePackError(f"{where}: a group may repeat at most {MAX_GROUP_REPEAT} times")
            _walk(sub, where, in_repeat=in_repeat or high > 1)
        elif op == _sre.BRANCH:
            for sub in av[1]:
                _walk(sub, where, in_repeat=in_repeat)
        elif op in (_sre.ASSERT, _sre.ASSERT_NOT):
            _walk(av[1], where, in_repeat=in_repeat)
            if _max_width(av[1]) > WINDOW:
                raise RulePackError(f"{where}: a lookaround may span at most {WINDOW} characters")
        elif op == getattr(_sre, "ATOMIC_GROUP", object()):
            _walk(av, where, in_repeat=in_repeat)
        elif op in (_sre.GROUPREF, _sre.GROUPREF_EXISTS):
            raise RulePackError(f"{where}: backreferences and conditionals are not allowed")
        elif op == _sre.LITERAL and chr(av).isupper():
            raise RulePackError(f"{where}: uppercase literal {chr(av)!r}; patterns run on lowercased text")


def check_pattern(pattern: str, where: str = "pattern") -> None:
    """Raise `RulePackError` unless ``pattern`` meets the pack safety rules."""
    if not isinstance(pattern, str) or not pattern:
        raise RulePackError(f"{where}: must be a non-empty string")
    if len(pattern) > MAX_PATTERN_CHARS:
        raise RulePackError(f"{where}: longer than {MAX_PATTERN_CHARS} characters")
    try:
        re.compile(pattern)
        parsed = _sre.parse(pattern)
    except re.error as exc:
        raise RulePackError(f"{where}: invalid regex ({exc})") from exc
    if parsed.state.flags & _FORBIDDEN_GLOBAL_FLAGS:
        raise RulePackError(f"{where}: inline flags are not allowed")
    _walk(parsed, where, in_repeat=False)
    high = _max_width(parsed)
    if high > WINDOW:
        raise RulePackError(
            f"{where}: can match up to {high} characters; the limit is {WINDOW}. "
            "Use bounded repeats such as \\S{1,40}"
        )
    low = parsed.getwidth()[0]
    if low < MIN_WIDTH:
        raise RulePackError(f"{where}: can match fewer than {MIN_WIDTH} characters")


# ------------------------------------------------------------------ parsing
def _str(table: Mapping[str, Any], key: str, where: str, *, required: bool = True) -> str:
    value = table.get(key)
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()):
        raise RulePackError(f"{where}.{key}: expected a non-empty string")
    return value


def _str_list(value: object, where: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise RulePackError(f"{where}: expected a list of strings")
    if len(value) > MAX_EXAMPLES:
        raise RulePackError(f"{where}: at most {MAX_EXAMPLES} entries")
    for v in value:
        if not v.strip() or len(v) > MAX_EXAMPLE_CHARS:
            raise RulePackError(f"{where}: entries must be 1-{MAX_EXAMPLE_CHARS} characters")
    return list(value)


def _matches(rule: Rule, text: str) -> bool:
    from .detectors.base import RuleDetector
    from .scanner import compile_rules

    hits = compile_rules([rule]).scan(normalize(text).text, limit=1, keep=RuleDetector._keep)
    return bool(hits)


def _rule(table: object, pack: str, index: int) -> Rule:
    where = f"rule[{index}]"
    if not isinstance(table, Mapping):
        raise RulePackError(f"{where}: expected a table")
    unknown = set(table) - _RULE_KEYS
    if unknown:
        raise RulePackError(f"{where}: unknown key(s) {', '.join(sorted(unknown))}")
    rule_id = _str(table, "id", where)
    where = f"rule {rule_id!r}"
    prefix = f"{pack}."
    if not rule_id.startswith(prefix) or not _ID_TAIL.fullmatch(rule_id[len(prefix):]):
        raise RulePackError(f"{where}: ids must look like '{pack}.<name>[.<name>...]' (lowercase)")
    try:
        category = Category(_str(table, "category", where).strip().lower())
    except ValueError as exc:
        raise RulePackError(f"{where}: unknown category; one of {', '.join(c.value for c in Category)}") from exc
    severity = _SEVERITIES.get(_str(table, "severity", where).strip().lower())
    if severity is None:
        raise RulePackError(f"{where}: severity must be one of {', '.join(_SEVERITIES)}")
    pattern = _str(table, "pattern", where)
    check_pattern(pattern, f"{where}: pattern")
    if "layers" in table:
        names = _str_list(table["layers"], f"{where}: layers")
        bad = [n for n in names if n not in _LAYERS]
        if bad or not names:
            raise RulePackError(f"{where}: layers must be a non-empty subset of {', '.join(_LAYERS)}")
        layers = tuple(dict.fromkeys(_LAYERS[n] for n in names))
    else:
        layers = _CONTENT_LAYERS
    validate = _str(table, "validate", where, required=False) or None
    if validate is not None:
        from .detectors.base import _VALIDATORS

        if validate not in _VALIDATORS:
            raise RulePackError(f"{where}: unknown validator; one of {', '.join(sorted(_VALIDATORS))}")
    try:
        rule = Rule(rule_id, category, severity, pattern, _str(table, "description", where), layers, validate)
    except PatternError as exc:
        raise RulePackError(f"{where}: {exc}") from exc

    examples = _str_list(table.get("examples"), f"{where}: examples")
    if not examples:
        raise RulePackError(f"{where}: needs at least one example it matches")
    for ex in examples:
        if not _matches(rule, ex):
            raise RulePackError(f"{where}: does not match its example {ex[:80]!r}")
    for ex in _str_list(table.get("counterexamples", []), f"{where}: counterexamples"):
        if _matches(rule, ex):
            raise RulePackError(f"{where}: matches its counterexample {ex[:80]!r}")
    return rule


def rule_pack_from_mapping(data: Mapping[str, Any], *, source: str = "<mapping>") -> RulePack:
    unknown = set(data) - {"pack", "rule"}
    if unknown:
        raise RulePackError(f"{source}: unknown top-level key(s) {', '.join(sorted(unknown))}")
    meta = data.get("pack")
    if not isinstance(meta, Mapping):
        raise RulePackError(f"{source}: missing [pack] table")
    unknown = set(meta) - _PACK_KEYS
    if unknown:
        raise RulePackError(f"{source}: [pack] unknown key(s) {', '.join(sorted(unknown))}")
    name = _str(meta, "name", "pack")
    if not _NAME.fullmatch(name):
        raise RulePackError(f"{source}: pack.name must be 2-40 chars of a-z, 0-9 and '-', starting with a letter")
    if name in _RESERVED:
        raise RulePackError(f"{source}: pack.name {name!r} is reserved for built-in rules")
    version = _str(meta, "version", "pack")
    if not _VERSION.fullmatch(version):
        raise RulePackError(f"{source}: pack.version must look like 1.2.3")
    tables = data.get("rule")
    if not isinstance(tables, list) or not tables:
        raise RulePackError(f"{source}: a pack needs at least one [[rule]]")
    if len(tables) > MAX_RULES:
        raise RulePackError(f"{source}: at most {MAX_RULES} rules per pack")
    try:
        rules = tuple(_rule(t, name, i) for i, t in enumerate(tables))
    except RulePackError as exc:
        raise RulePackError(f"{source}: {exc}") from exc
    seen: set[str] = set()
    for r in rules:
        if r.id in seen:
            raise RulePackError(f"{source}: duplicate rule id {r.id!r}")
        seen.add(r.id)
    return RulePack(name, version, _str(meta, "description", "pack", required=False), rules, source)


def load_rule_pack(path: str | Path) -> RulePack:
    p = Path(path)
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RulePackError(f"{p}: cannot read ({exc.strerror or exc})") from exc
    except tomllib.TOMLDecodeError as exc:
        raise RulePackError(f"{p}: invalid TOML ({exc})") from exc
    return rule_pack_from_mapping(data, source=str(p))


def bundled_rule_packs() -> list[str]:
    """Names of the packs that ship inside the charterguard wheel."""
    from importlib.resources import files

    folder = files("charterguard") / "rulepacks"
    return sorted(e.name[:-5] for e in folder.iterdir() if e.name.endswith(".toml"))


def bundled_rule_pack(name: str) -> RulePack:
    """Load a pack that ships with charterguard, by name (see `bundled_rule_packs`)."""
    from importlib.resources import files

    if name not in bundled_rule_packs():
        raise RulePackError(f"no bundled rule pack {name!r}; available: {', '.join(bundled_rule_packs())}")
    resource = files("charterguard") / "rulepacks" / f"{name}.toml"
    try:
        data = tomllib.loads(resource.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:  # pragma: no cover - shipped packs are tested
        raise RulePackError(f"{name}: invalid TOML ({exc})") from exc
    pack = rule_pack_from_mapping(data, source=f"charterguard:{name}")
    if pack.name != name:  # pragma: no cover - shipped packs are tested
        raise RulePackError(f"bundled pack file {name}.toml declares name {pack.name!r}")
    return pack


def combine_rules(packs: Iterable[RulePack], *, builtin: Sequence[Rule] = RULES) -> tuple[Rule, ...]:
    """Built-in rules plus every pack's rules. Duplicate ids are an error."""
    combined = list(builtin)
    seen = {r.id for r in combined}
    names: set[str] = set()
    for pack in packs:
        if pack.name in names:
            raise RulePackError(f"rule pack {pack.name!r} loaded twice")
        names.add(pack.name)
        for r in pack.rules:
            if r.id in seen:
                raise RulePackError(f"rule id {r.id!r} from pack {pack.name!r} is already defined")
            seen.add(r.id)
            combined.append(r)
    return tuple(combined)
