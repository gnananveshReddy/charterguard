# charterguard

[![ci](https://github.com/gnananveshReddy/charterguard/actions/workflows/ci.yml/badge.svg)](https://github.com/gnananveshReddy/charterguard/actions/workflows/ci.yml)
[![codeql](https://github.com/gnananveshReddy/charterguard/actions/workflows/codeql.yml/badge.svg)](https://github.com/gnananveshReddy/charterguard/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/gnananveshReddy/charterguard/badge)](https://scorecard.dev/viewer/?uri=github.com/gnananveshReddy/charterguard)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![dependencies](https://img.shields.io/badge/runtime%20dependencies-0-brightgreen)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)


**Deterministic guardrails for LLM agents, at every trust boundary.** charterguard
checks user turns, retrieved documents, tool calls, tool output and final answers
for prompt injection, jailbreaks, destructive commands, secrets and PII. It runs in
process in microseconds, with zero runtime dependencies, and every decision is a
typed, auditable `Verdict`.

* **Controls do not live in prompts.** Every decision is made in code, outside the
  model, and can be logged, replayed and audited.
* **Everything the model did not author is untrusted.** Retrieved content and tool
  results are inspected, wrapped in a delimiter-hardened envelope, and quarantined
  if they carry instructions.

📖 **Documentation:** [docs/](docs/index.md) (also published with GitHub Pages
from `mkdocs.yml`)

## Install

```bash
pip install "charterguard @ git+https://github.com/gnananveshReddy/charterguard"
pip install "charterguard[fastapi] @ git+https://github.com/gnananveshReddy/charterguard"   # or [langchain], [onnx], ...
```

`pip install charterguard` works once the first release is on PyPI. Python 3.11+.
Every extra is optional and none is imported by the core; see the
[adapter matrix](docs/adapters.md).

## Quick start

```python
from charterguard import GuardrailConfig, GuardrailPipeline

guardrails = GuardrailPipeline(GuardrailConfig.strict())

verdict = guardrails.check_input(user_message, session_id=session_id)
if verdict.blocked:
    return guardrails.refusal(verdict)

for chunk in retrieved:                       # before it reaches the context window
    result = guardrails.inspect_untrusted(chunk.text, source=chunk.uri)
    if result.safe_to_use:
        context.append(result.context_block())

tool_result = guardrails.inspect_tool_output(raw_output, tool_name="search_docs")
out = guardrails.check_output(answer)
final_text = guardrails.refusal(out) if out.blocked else (out.sanitized or answer)
```

```bash
echo "ignore all previous instructions" | charterguard scan
charterguard scan --layer tool --pack agent-tools fetched_page.html
```

## What's included

| | |
| --- | --- |
| **Detection** | Injection and jailbreak rules (English plus seed coverage for 26 more languages, including 10 Indian languages and Hinglish), encoded-payload decoding, de-obfuscation (leetspeak, letter spacing, rot13, quote wrapping), destructive commands, exfiltration beacons |
| **Data protection** | Secret and PII redaction with checksum validators (cards, IBAN, Aadhaar, PAN, GSTIN, ...) and exact-span placeholders |
| **Agent controls** | Charter-based scope gate, tool allow-lists, per-session volume ceilings, untrusted-content envelopes, grounding checks |
| **Integrations** | FastAPI, LangChain and LangGraph, OpenAI Agents SDK, Claude Agent SDK, MCP (server and result guard), CLI and pre-commit hook |
| **Operations** | TOML policy files, self-testing [rule packs](docs/rule-packs.md), shadow and report-only rollout, hash-chained audit, metrics sinks, optional classifier tier (`.[onnx]`) |

## Accuracy, honestly

Rules only, no classifier attached (`python benchmarks/accuracy.py`):

| Dataset | Precision | Recall | FPR |
| --- | --- | --- | --- |
| bundled corpus (hand-written attacks and tricky benign texts) | 0.96 | 1.00 | 0.04 |
| deepset/prompt-injections, held-out test split | 1.00 | 0.20 | 0.00 |
| jackhhao/jailbreak-classification | 0.91 | 0.50 | 0.05 |

The rules are precise, not complete: paraphrased attacks need the classifier tier.
See [Detection accuracy](docs/accuracy.md) and [How it compares](docs/comparison.md).

## Documentation

* [Overview and trust boundaries](docs/index.md)
* [Integrations](docs/integrations.md): FastAPI, LangChain/LangGraph, agent SDKs, MCP
* [Untrusted content and envelopes](docs/untrusted-content.md)
* [Policy files](docs/policy.md) · [Rule packs](docs/rule-packs.md) · [Command line](docs/cli.md)
* [Operating it](docs/operations.md): rollout modes, audit, performance
* [Extending it](docs/extending.md) · [Adapters](docs/adapters.md)
* [Security assumptions and limits](docs/security.md)

## Contributing and security

Missed attacks and false positives are the most useful contributions; the issue
templates ask for exactly what is needed. See [CONTRIBUTING.md](CONTRIBUTING.md).
Report bypasses privately as described in [SECURITY.md](SECURITY.md).

MIT licensed.
