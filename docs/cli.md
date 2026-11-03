# Command line

Installing the package adds a `charterguard` command (also `python -m charterguard`):

```bash
echo "ignore all previous instructions" | charterguard scan --strict   # exit 1: blocked
charterguard scan --layer untrusted docs/*.md --json                 # RAG documents
charterguard scan --layer output --redact reply.txt > reply.safe.txt # redacted text
charterguard eval labelled.jsonl --label-field is_attack --min-recall 0.9
charterguard rules --category pii
```

* `scan` exits 1 when any input reaches `--fail-on` (`block` by default, or `flag`,
  `never`), so it works as a CI step. Reports show rule, severity and offsets, never
  the matched text, so scanning a file with a live key does not put the key in a log.
* `eval` reports precision, recall, F1 and false-positive rate on any JSONL file, and
  can gate CI with `--min-recall` / `--max-fpr`.
* `.pre-commit-hooks.yaml` exposes `charterguard-prompts` and `charterguard-secrets`
  for use from other repositories.
