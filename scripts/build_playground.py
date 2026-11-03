"""Build the wheel the docs playground installs, plus its manifest.

    python scripts/build_playground.py && mkdocs serve

Writes docs/assets/playground/<wheel> and wheel.json. Both are build output
(gitignored); the docs workflow runs this before `mkdocs build`, so the
playground always runs the code from the same commit as the docs.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets" / "playground"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.whl"):
        old.unlink()
    subprocess.run(  # noqa: S603 - fixed argv, no user input
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--quiet", "-w", str(OUT), str(ROOT)],
        check=True,
    )
    wheels = sorted(OUT.glob("charterguard-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one charterguard wheel in {OUT}, found {[w.name for w in wheels]}")
    wheel = wheels[0]
    version = wheel.name.split("-")[1]
    (OUT / "wheel.json").write_text(json.dumps({"wheel": wheel.name, "version": version}) + "\n", encoding="utf-8")
    print(f"playground wheel: {wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
