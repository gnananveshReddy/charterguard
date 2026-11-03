# Integrations

## FastAPI integration

```python
from fastapi import Depends, FastAPI
from charterguard import GuardrailPipeline, Verdict
from charterguard.integrations.fastapi import (
    guard_response,
    input_guard,
    install_exception_handler,
)

app = FastAPI()
pipeline = GuardrailPipeline()
install_exception_handler(app)          # renders GuardrailBlocked as HTTP 422 + audit
guard = input_guard(pipeline, enforce_scope=True)

@app.post("/turn")
async def turn(body: dict, verdict: Verdict = Depends(guard)):
    answer = await run_agent(body["message"])
    text, out_verdict = guard_response(pipeline, answer)
    return {"answer": text, "policy_id": out_verdict.policy_id}
```

The verdict is injected into the handler even when nothing is blocked, so
handlers can log FLAG outcomes during a phased rollout.

Construction is cheap (~150 us) and compiled rule sets are shared process-wide,
so building a pipeline per request in a dependency is fine. Scanning is pure and
thread-safe; only the session ledger holds mutable state, behind a lock.

## LangChain and LangGraph

Both frameworks are supported, and the guard decisions are shared between them,
so a graph and a chain quarantine on the same rules and emit the same audit rows.

```python
# LangChain: guards compose with `|`
from charterguard.integrations.langchain import guard_input, guard_output
chain = guard_input(pipeline) | prompt | model | guard_output(pipeline)

# LangGraph: guards are nodes, and the router enforces
from charterguard.integrations.langgraph import (
    guard_input_node, guardrail_router, refusal_node,
)
g.add_node("guard_in", guard_input_node(pipeline))
g.add_conditional_edges("guard_in", guardrail_router(ok="model", blocked="refuse"))
```

Streaming needs the streaming guard, not `guard_output`: a `RunnableLambda`
buffers the whole response before it can return, so a streamed chain silently
loses streaming and gets no incremental enforcement. Use
`guard_output_stream` (LangChain) or `guard_message_stream` (LangGraph); both
enforce as tokens arrive via a hold-back window, so a secret split across two
chunks is caught before any of it is emitted.

Blocked actions do not have to dead-end. `approval_node` converts a block into a
durable LangGraph `interrupt()`, so an irreversible action can be escalated to a
human reviewer and resumed with `Command(resume=...)`. It fails closed: an
unrecognised resume value is a rejection.

The LangGraph adapter imports **no third-party package**: nodes are plain
callables over state, so they work across langgraph versions and are tested with
the framework absent. Guards write a JSON-serialisable record to state (a
`Verdict` in state would break checkpointers) and the router decides, which lets
you ship in observe-only mode first.

Two contract details the adapters get right, and hand-rolled versions usually do
not: a blocked tool call still answers *every* `tool_call_id` (a stranded id is a
provider 400 on the next turn), and a quarantined tool result is neutralised
in place rather than deleted, without refusing the turn. Poisoned *retrieved
documents* are dropped instead, since they carry no id awaiting an answer.

See [the adapter matrix](adapters.md#langgraph) for the full wiring, the
`safe_output`-versus-history split, and the guarded retriever and tool wrappers.


## Agent SDKs

**OpenAI Agents SDK** (`pip install "charterguard[openai-agents]"`):

```python
from charterguard.integrations.openai_agents import (
    input_guardrail, output_guardrail, tool_input_guardrail, tool_output_guardrail,
)

agent = Agent(name="support",
              input_guardrails=[input_guardrail(pipeline)],
              output_guardrails=[output_guardrail(pipeline)])
```

The input guardrail runs blocking (`run_in_parallel=False`), so the agent does not
start acting on a turn that is about to be blocked. The tool output guardrail replaces
quarantined or redacted tool output with the enveloped text.

**Claude Agent SDK** (`pip install "charterguard[claude-agent-sdk]"`):

```python
from charterguard.integrations import claude_agent_hooks

options = ClaudeAgentOptions(hooks=claude_agent_hooks(pipeline))
```

`UserPromptSubmit` blocks attacks before the model sees them, `PreToolUse` denies
forbidden or command-injecting tool calls, and `PostToolUse` replaces quarantined
output, or output carrying secrets or PII, with a same-shape redacted or withheld copy
(`updatedToolOutput`). The SDK keeps the original output if a replacement doesn't
match the tool's schema, so for tools you control, guarding inside the tool (or
serving it through `charterguard mcp`) is still the strongest option.

**Pydantic AI** (`pip install "charterguard[pydantic-ai]"`):

```python
from charterguard.integrations.pydantic_ai import guard_capability

agent = Agent("openai:gpt-5", capabilities=[guard_capability(pipeline)])
```

One capability covers the whole run. A blocked prompt raises `GuardrailBlocked` before
any model request. A blocked tool call is skipped, and the model gets a refusal as the
tool result. Quarantined or redacted tool results are replaced with the enveloped text.
The final output is checked too: a block raises, and plain-text output comes back
redacted. A structured output is rebuilt from its redacted JSON, and the run raises if
that is not possible, so a redaction is never silently dropped. The session id defaults
to the run's `conversation_id`, which stays stable across runs that continue a
`message_history`. Pass `session_id=` to derive it from your deps.

## MCP

`charterguard mcp` runs an [MCP](https://modelcontextprotocol.io) server over stdio
(stdlib only), exposing `scan_text`, `redact_text` and `list_rules` to any MCP client:

```json
{"mcpServers": {"charterguard": {"command": "charterguard", "args": ["mcp", "--policy", "guardrails.toml"]}}}
```

For the other direction, results from *other* MCP servers are untrusted input. Run
them through the tool-output guard before the model sees them:

```python
from charterguard.integrations import guard_mcp_result

safe = guard_mcp_result(pipeline, result, tool_name="search")
```

It fails closed. Every readable part (text, embedded resources, link descriptions,
`structuredContent`) is scanned as one payload, so an attack split across blocks is
caught. The model gets a single redacted, enveloped text block. Raw
`structuredContent`, unknown or binary blocks and server `_meta` are never forwarded,
media only with `allow_media=True`, and a quarantined result is marked `isError`.
