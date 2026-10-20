"""Run the test suite where pytest is not installed (e.g. an offline VM).

Usage, from the repo root:  python3 tools/run_tests_without_pytest.py
CI and normal development use pytest; this is only a fallback.
"""

import contextlib
import importlib
import inspect
import sys
import tempfile
import traceback
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "pytest" not in sys.modules:
    try:
        import pytest  # noqa: F401
    except ImportError:
        shim = types.ModuleType("pytest")

        import re
        import warnings

        @contextlib.contextmanager
        def raises(error, match=None):
            try:
                yield
            except error as caught:
                if match is not None and not re.search(match, str(caught)):
                    raise AssertionError(f"{caught!r} does not match {match!r}")
                return
            raise AssertionError(f"{error.__name__} not raised")

        @contextlib.contextmanager
        def warns(category, match=None):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                yield caught
            hits = [w for w in caught if issubclass(w.category, category)]
            if match is not None:
                hits = [w for w in hits if re.search(match, str(w.message))]
            if not hits:
                raise AssertionError(f"no {category.__name__} warning")

        class approx:
            def __init__(self, expected, rel=1e-6, abs=1e-12):
                self.expected, self.rel, self.abs = expected, rel, abs

            def __eq__(self, other):
                import numpy as np

                return bool(
                    np.allclose(other, self.expected, rtol=self.rel, atol=self.abs)
                )

        shim.raises, shim.warns, shim.approx = raises, warns, approx
        sys.modules["pytest"] = shim

MODULES = sys.argv[1:] or [
    "tests.test_preprocessing",
    "tests.test_preprocessing_registry",
]
failed = 0
for name in MODULES:
    module = importlib.import_module(name)
    for test_name, test in inspect.getmembers(module, inspect.isfunction):
        if not test_name.startswith("test_"):
            continue
        kwargs = {}
        if "tmp_path" in inspect.signature(test).parameters:
            kwargs["tmp_path"] = Path(tempfile.mkdtemp())
        try:
            test(**kwargs)
            print(f"PASS {name}.{test_name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}.{test_name}")
            traceback.print_exc()
print(f"{failed} failed")
sys.exit(1 if failed else 0)
