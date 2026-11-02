"""Command-line interface: ``charterguard scan | eval | rules | version``.

    echo "ignore all previous instructions" | charterguard scan
    charterguard scan --layer untrusted retrieved/*.md --json
    charterguard scan --layer output --redact reply.txt > reply.safe.txt
    charterguard eval my_prompts.jsonl --label-field is_attack
    charterguard rules --category injection

Exit status, so the scanner drops into CI and pre-commit hooks:

* ``scan``: 1 when any input reaches ``--fail-on`` (default ``block``), else 0.
* ``eval``: 1 when ``--min-recall`` / ``--max-fpr`` is given and not met.
* 2 on usage errors (argparse's convention).

Findings are reported by rule, severity and character offsets, never by the
matched text: a scan of a file holding a live credential must not print the
credential into a CI log. ``--redact`` prints the redacted text instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TextIO

from . import __version__
from .config import GuardrailConfig
from .patterns import RULES
from .pipeline import GuardrailPipeline
from .types import Action, Category, Layer, Verdict

_LAYERS = {
    "input": Layer.INPUT,
    "untrusted": Layer.UNTRUSTED,
    "tool": Layer.TOOL_OUTPUT,
    "output": Layer.OUTPUT,
}
_RANK = {Action.ALLOW: 0, Action.FLAG: 1, Action.BLOCK: 2}
_FAIL_ON = {"flag": 1, "block": 2, "never": 3}


# ------------------------------------------------------------------ helpers
def _pipeline(strict: bool) -> GuardrailPipeline:
    return GuardrailPipeline(GuardrailConfig.strict() if strict else GuardrailConfig())


def _check(p: GuardrailPipeline, layer: str, text: str, source: str) -> tuple[Verdict, str | None]:
    """Run one text through the named layer. Returns the verdict and, where
    the layer produces one, the text that is safe to pass on."""
    if layer == "input":
        return p.check_input(text), None
    if layer == "output":
        v = p.check_output(text)
        return v, v.sanitized
    if layer == "tool":
        r = p.inspect_tool_output(text, tool_name=source)
    else:
        r = p.inspect_untrusted(text, source=source)
    return r.verdict, r.envelope


def _inputs(paths: Sequence[str], stdin: TextIO) -> Iterator[tuple[str, str]]:
    if not paths or paths == ["-"]:
        yield "<stdin>", stdin.read()
        return
    for raw in paths:
        if raw == "-":
            yield "<stdin>", stdin.read()
            continue
        path = Path(raw)
        try:
            yield str(path), path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise SystemExit(f"charterguard: cannot read {raw}: {exc.strerror or exc}") from exc


def _finding_dicts(v: Verdict) -> list[dict[str, object]]:
    return [
        {
            "rule_id": f.rule_id,
            "category": f.category.value,
            "severity": f.severity.name,
            "start": f.start,
            "end": f.end,
            "description": f.description,
        }
        for f in v.findings
    ]


# --------------------------------------------------------------------- scan
def _cmd_scan(args: argparse.Namespace, out: TextIO, stdin: TextIO) -> int:
    p = _pipeline(args.strict)
    worst = 0
    reports = []
    for source, text in _inputs(args.paths, stdin):
        verdict, passed_on = _check(p, args.layer, text, source)
        worst = max(worst, _RANK[verdict.action])
        if args.redact:
            if args.layer == "input":
                raise SystemExit("charterguard: --redact needs --layer output, untrusted or tool")
            out.write("" if passed_on is None else passed_on)
            continue
        findings = _finding_dicts(verdict)
        reports.append({
            "source": source,
            "layer": args.layer,
            "action": verdict.action.value,
            "score": round(verdict.score, 3),
            "policy_id": verdict.policy_id,
            "findings": findings,
        })
        if not args.json:
            out.write(f"{source}: {verdict.action.value.upper()} (score {verdict.score:.2f})\n")
            for f in findings:
                out.write(
                    f"  {f['severity']:<8} {f['rule_id']:<32} [{f['start']}:{f['end']}] {f['description']}\n"
                )
    if args.json and not args.redact:
        json.dump(reports if len(reports) != 1 else reports[0], out, indent=2)
        out.write("\n")
    return 1 if worst >= _FAIL_ON[args.fail_on] else 0


# --------------------------------------------------------------------- eval
def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "attack", "injection", "jailbreak", "malicious"}


def _cmd_eval(args: argparse.Namespace, out: TextIO, stdin: TextIO) -> int:
    p = _pipeline(args.strict)
    tp = fp = tn = fn = 0
    misses: list[str] = []
    for source, blob in _inputs([args.path], stdin):
        for lineno, line in enumerate(blob.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                text = str(row[args.text_field])
                attack = _truthy(row[args.label_field])
            except (ValueError, KeyError, TypeError) as exc:
                raise SystemExit(f"charterguard: {source}:{lineno}: bad row ({exc})") from exc
            verdict, _ = _check(p, args.layer, text, source)
            detected = verdict.blocked if args.count == "block" else verdict.action is not Action.ALLOW
            if attack and detected:
                tp += 1
            elif attack:
                fn += 1
                misses.append(f"{source}:{lineno}")
            elif detected:
                fp += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = {
        "samples": tp + fp + tn + fn,
        "attacks": tp + fn,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "missed": misses[:50],
    }
    if args.json:
        json.dump(result, out, indent=2)
        out.write("\n")
    else:
        out.write(
            f"samples {result['samples']} (attacks {result['attacks']})  "
            f"precision {precision:.3f}  recall {recall:.3f}  f1 {f1:.3f}  fpr {fpr:.3f}\n"
        )
        if misses:
            out.write(f"missed: {', '.join(misses[:10])}{' ...' if len(misses) > 10 else ''}\n")
    failed = (args.min_recall is not None and recall < args.min_recall) or (
        args.max_fpr is not None and fpr > args.max_fpr
    )
    return 1 if failed else 0


# -------------------------------------------------------------------- rules
def _cmd_rules(args: argparse.Namespace, out: TextIO, stdin: TextIO) -> int:
    rules = [
        r for r in RULES
        if (args.category is None or r.category.value == args.category)
        and (args.layer is None or _LAYERS[args.layer] in r.layers)
    ]
    if args.json:
        json.dump(
            [
                {
                    "id": r.id,
                    "category": r.category.value,
                    "severity": r.severity.name,
                    "description": r.description,
                    "layers": [layer.value for layer in r.layers],
                }
                for r in rules
            ],
            out,
            indent=2,
        )
        out.write("\n")
    else:
        for r in rules:
            out.write(f"{r.id:<34} {r.category.value:<10} {r.severity.name:<8} {r.description}\n")
        out.write(f"{len(rules)} rules\n")
    return 0


def _cmd_version(args: argparse.Namespace, out: TextIO, stdin: TextIO) -> int:
    out.write(f"charterguard {__version__}\n")
    return 0


# ------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="charterguard", description="Guardrail scanner for LLM agent text.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--layer", choices=sorted(_LAYERS), default="input",
                        help="which guardrail to apply (default: input)")
        sp.add_argument("--strict", action="store_true", help="use GuardrailConfig.strict()")
        sp.add_argument("--json", action="store_true", help="machine-readable output")

    scan = sub.add_parser("scan", help="scan files or stdin")
    scan.add_argument("paths", nargs="*", help="files to scan; '-' or nothing reads stdin")
    common(scan)
    scan.add_argument("--fail-on", choices=sorted(_FAIL_ON), default="block",
                      help="exit 1 when any input reaches this action (default: block)")
    scan.add_argument("--redact", action="store_true",
                      help="print the redacted / enveloped text instead of a report")
    scan.set_defaults(func=_cmd_scan)

    ev = sub.add_parser("eval", help="measure detection on a labelled JSONL file")
    ev.add_argument("path", help="JSONL file, one object per line; '-' reads stdin")
    common(ev)
    ev.add_argument("--text-field", default="text")
    ev.add_argument("--label-field", default="label", help="truthy = attack (1, true, 'attack', ...)")
    ev.add_argument("--count", choices=("flag", "block"), default="flag",
                    help="count flag-or-block as detected (default), or block only")
    ev.add_argument("--min-recall", type=float, help="exit 1 below this recall")
    ev.add_argument("--max-fpr", type=float, help="exit 1 above this false-positive rate")
    ev.set_defaults(func=_cmd_eval)

    rules = sub.add_parser("rules", help="list built-in rules")
    rules.add_argument("--category", choices=sorted(c.value for c in Category))
    rules.add_argument("--layer", choices=sorted(_LAYERS))
    rules.add_argument("--json", action="store_true")
    rules.set_defaults(func=_cmd_rules)

    version = sub.add_parser("version", help="print the version")
    version.set_defaults(func=_cmd_version)
    return parser


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None, stdin: TextIO | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args, out or sys.stdout, stdin or sys.stdin)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
