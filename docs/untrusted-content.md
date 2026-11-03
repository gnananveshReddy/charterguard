# Untrusted content and envelopes

`inspect_untrusted` returns an `UntrustedResult`:

* `verdict` — the typed decision, with findings and audit payload.
* `envelope` — the text to actually append to the context window, wrapped in
  delimiters that declare it as data and carry a nonce so injected content cannot
  forge a boundary.
* `context_block()` — the envelope, or a neutral placeholder note when the
  content was quarantined. It never returns the original text for quarantined
  content.
* `safe_to_use` / `quarantined` — whether the content was admitted.

Pair this with `pipeline.system_rules()`, which emits the envelope-handling rules
the model needs, and `verify_prefix()`, which asserts your assembled system
prefix still contains them.
