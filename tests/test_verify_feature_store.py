"""Regression coverage for the root-level ``verify_feature_store.py`` script (issue #998).

The script is a hand-run smoke check for the Feature Store. Nothing executed it,
so it could rot silently: a renamed module, a moved file, or a signature change
in :mod:`astroml.features.feature_store` would leave CI green while the script
reported failure to whoever happened to run it by hand.

The script catches its own exceptions and returns ``False`` instead of raising,
so asserting on its return values is the only way a failure surfaces here. Each
check is run in-process against a temporary feature store, and the captured
output is attached to any assertion so the failure is diagnosable from the
pytest report alone.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "verify_feature_store.py"

_CHECKS = (
    "test_imports",
    "test_basic_functionality",
    "test_data_structures",
    "test_file_structure",
    "test_integration",
)


@pytest.fixture(scope="module")
def verify_feature_store() -> ModuleType:
    """Import ``verify_feature_store.py`` by path, the way a hand-run script loads.

    Loading by path rather than as ``import verify_feature_store`` keeps this
    independent of pytest's import mode and of the repo root happening to be on
    ``sys.path``. The module's ``if __name__ == "__main__"`` guard keeps
    ``exec_module`` from running ``main()`` as a side effect of the import.
    """
    if not _SCRIPT_PATH.is_file():
        pytest.fail(f"verification script is missing: {_SCRIPT_PATH}")

    spec = importlib.util.spec_from_file_location("verify_feature_store", _SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        pytest.fail(f"cannot build an import spec for {_SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("check_name", _CHECKS)
def test_check_reports_success(
    verify_feature_store: ModuleType,
    capsys: pytest.CaptureFixture[str],
    check_name: str,
) -> None:
    """A verification check must report success, not swallow a failure.

    Args:
        verify_feature_store: The loaded script module.
        capsys: Pytest capture fixture, used to report the check's output.
        check_name: Name of the check function under test.
    """
    check = getattr(verify_feature_store, check_name)

    result = check()
    output = capsys.readouterr().out

    assert result is True, f"{check_name}() reported failure:\n{output}"


def test_main_runs_every_module_level_check(verify_feature_store: ModuleType) -> None:
    """Every ``test_*`` helper must be reachable from ``main()``'s table.

    Guards the other direction of rot: a check added to the script but not wired
    into ``main()`` would never run for someone following the script's own
    "5/5" output.
    """
    source = inspect.getsource(verify_feature_store.main)
    declared = {
        name
        for name, value in vars(verify_feature_store).items()
        if name.startswith("test_") and callable(value)
    }

    # ``main``'s table references the checks as ``("Label", test_imports)``, so
    # match the bare name rather than expecting call syntax.
    unwired = sorted(
        name for name in declared if re.search(rf"\b{re.escape(name)}\b", source) is None
    )

    assert not unwired, f"checks defined but never run by main(): {unwired}"


def test_main_reports_full_success(
    verify_feature_store: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main()`` is the script's own pass/fail gate; it must still report all-pass.

    ``main()``'s boolean is what becomes the process exit code, so this is the
    contract a CI gate or a pre-release check would rely on.
    """
    result = verify_feature_store.main()
    output = capsys.readouterr().out

    assert result is True, f"main() reported failure:\n{output}"
    assert f"{len(_CHECKS)}/{len(_CHECKS)} tests passed" in output
