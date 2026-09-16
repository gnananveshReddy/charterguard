"""Dependency-free test runner.

The suite is written for pytest, but this component must be verifiable in
locked-down environments where pytest is not installed (air-gapped build
agents, minimal containers). This runner discovers the same tests, supports
async tests, and shims the two pytest helpers the suite uses.

    python tests/run_tests.py          # no pytest needed
    python -m pytest tests -q          # identical results where pytest exists
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import sys
import traceback
import types
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))


def _install_pytest_shim() -> None:
    if "pytest" in sys.modules:
        return
    try:
        import pytest  # noqa: F401
        return
    except ImportError:
        pass

    module = types.ModuleType("pytest")

    @contextmanager
    def raises(expected, **_kwargs):
        try:
            yield
        except expected:
            return
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"expected {expected.__name__}, got {type(exc).__name__}: {exc}"
            ) from exc
        raise AssertionError(f"expected {expected.__name__}, nothing raised")

    class _Mark:
        def __getattr__(self, _name):
            def decorator(fn=None, **_kw):
                return fn if fn is not None else (lambda f: f)

            return decorator

    module.raises = raises
    module.mark = _Mark()
    module.fixture = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    sys.modules["pytest"] = module


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    _install_pytest_shim()
    files = sorted(p for p in HERE.glob("test_*.py"))
    passed: list[str] = []
    failed: list[tuple[str, str]] = []

    for path in files:
        module = _load(path)
        for name, fn in sorted(vars(module).items()):
            if not name.startswith("test_") or not callable(fn):
                continue
            label = f"{path.stem}::{name}"
            try:
                if inspect.iscoroutinefunction(fn):
                    asyncio.run(fn())
                else:
                    fn()
            except Exception:  # noqa: BLE001
                failed.append((label, traceback.format_exc(limit=3)))
                print(f"FAIL {label}")
            else:
                passed.append(label)
                print(f"ok   {label}")

    print()
    print(f"{len(passed)} passed, {len(failed)} failed, {len(files)} files")
    for label, tb in failed:
        print(f"\n===== {label} =====\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
