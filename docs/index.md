# charterguard

A framework-agnostic guardrail pipeline for LLM agent harnesses: prompt-injection
and jailbreak detection, harmful-content and destructive-command rules, scope
enforcement against an agent charter, untrusted-content envelopes, tool-output
inspection, secret and PII redaction, grounding checks, session volume ceilings,
and audit-ready verdicts.

Zero runtime dependencies. Python 3.11+. FastAPI is an optional integration, not
a requirement.

## Why this exists

Two ideas drive the whole design.

**Controls do not live in prompts.** A system prompt is a request, not a control.
Every decision here is made in code, outside the model, and returns a typed
`Verdict` that can be logged, replayed, and audited.

**Everything the model did not author is untrusted.** Retrieved documents, tool
results, web pages, and file contents are inspected before they reach the context
window, wrapped in a delimiter-hardened envelope that marks them as data, and
quarantined if they carry instructions.

## Install

Until the first PyPI release, install from GitHub (afterwards `pip install charterguard`):

```bash
pip install "charterguard @ git+https://github.com/gnananveshReddy/charterguard"
# from a checkout, with extras:
pip install -e .                 # core + every stdlib adapter, no dependencies
pip install -e ".[fastapi]"      # ASGI middleware
pip install -e ".[langchain]"    # LangChain runnables, guarded tools + retriever
pip install -e ".[langgraph]"    # LangGraph nodes (the adapter needs no import)
pip install -e ".[redis]"        # session ceilings that hold across hosts
pip install -e ".[prometheus]"   # or .[otel] -- metrics egress
pip install -e ".[spacy]"        # or .[presidio] -- free-text PII (names)
pip install -e ".[onnx]"         # local model tier: onnxruntime, no torch
```

Every extra is genuinely optional and none is imported by the core. All six
extension points have a working standard-library option -- metrics to
logging/console/StatsD, session ceilings and audit to sqlite, model/intent/NER
over plain HTTP. See [the adapter matrix](adapters.md) for the full matrix
with measured per-call costs.

## Quick start

```python
from charterguard import GuardrailPipeline, GuardrailConfig, Charter

charter = Charter(
    agent_id="policy-assistant",
    version="3",
    purpose="Answer questions about internal policy documents.",
    in_scope=("policy", "standard", "control", "benefit"),
    out_of_scope=("legal advice", "compensation decisions"),
    allowed_tools=("search_docs", "get_policy"),
)

guardrails = GuardrailPipeline(GuardrailConfig.report_only(), charter=charter)

# 1. Inbound turn
verdict = guardrails.check_input(user_message, session_id=session_id)
if verdict.blocked:
    return guardrails.refusal(verdict)

# 2. Retrieved content, before it touches the context window
for chunk in retrieved:
    result = guardrails.inspect_untrusted(chunk.text, source=chunk.uri, kind="retrieval")
    if result.safe_to_use:
        context.append(result.context_block())

# 3. Tool calls
call_verdict = guardrails.check_tool_call("search_docs", arguments, session_id=session_id)

# 4. Tool results
tool_result = guardrails.inspect_tool_output(raw_output, tool_name="search_docs")

# 5. Outbound answer
out = guardrails.check_output(answer, allowed_citations=[c.uri for c in retrieved])
final_text = guardrails.refusal(out) if out.blocked else (out.sanitized or answer)
```

## Trust boundaries

| Boundary | Call | What it is for |
| --- | --- | --- |
| Inbound turn | `check_input` | Injection, jailbreak, harmful intent, destructive commands |
| Scope | `check_scope` | Charter adherence, out-of-scope requests |
| Tool invocation | `check_tool_call` | Tool allow-list, argument inspection, call ceilings |
| Untrusted content | `inspect_untrusted` | Retrieved or fetched content, envelope wrapping, quarantine |
| Tool output | `inspect_tool_output` | Same as above, plus secret and PII redaction before context insertion |
| Outbound answer | `check_output` | Secret and PII redaction, grounding, cumulative volume |

`acheck_input` and `acheck_scope` are async variants for the model-assisted tier.
`check_input_or_raise` raises `GuardrailBlocked` instead of returning a verdict.
