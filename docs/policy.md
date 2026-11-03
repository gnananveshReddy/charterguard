# Policy files

Enforcement changes with rollout phase, so it can live in a reviewed TOML (or JSON)
file instead of code. See [`docs/policy.example.toml`](https://github.com/gnananveshReddy/charterguard/blob/main/docs/policy.example.toml).

```toml
preset = "default"            # default | strict | report_only

[modes]
injection = "block"
pii = "flag"

[flag_at]
pii = 0.4
```

```python
cfg = GuardrailConfig.from_file("guardrails.toml")   # or charterguard.load_policy(...)
```

```bash
charterguard policy guardrails.toml          # validate, print the effective policy
charterguard scan --policy guardrails.toml prompts/
```

Loading is strict. An unknown key (`injectoin = "block"`), a wrong type, a threshold
outside [0, 1], or `flag_at` above `block_at` raises `PolicyError` naming the key, so a
typo can never silently weaken enforcement. Only the stdlib `tomllib` is used.
