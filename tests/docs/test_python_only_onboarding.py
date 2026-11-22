"""Keep the documented Docker-free onboarding path tied to its smoke tests."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_python_only_onboarding_documents_the_local_sample_path() -> None:
    """The no-Docker instructions must point at the runnable sample workflow."""
    onboarding = (ROOT / "docs/ONBOARDING.md").read_text(encoding="utf-8")

    assert "requirements-cpu.txt -r requirements-dev.txt" in onboarding
    assert "pytest tests/integration/test_e2e_sample_data.py -m e2e -q" in onboarding
    assert "does not start or connect to either service" in onboarding
    assert (ROOT / "tests/integration/test_e2e_sample_data.py").is_file()


def test_python_only_onboarding_marks_the_database_boundary() -> None:
    """The PostgreSQL quick start must not be presented as Docker-free."""
    onboarding = (ROOT / "docs/ONBOARDING.md").read_text(encoding="utf-8")

    assert "`make quickstart` / `python -m astroml.quick_start`" in onboarding
    assert "needs a reachable" in onboarding
    assert "PostgreSQL server" in onboarding
    assert (
        "tests/test_quick_start_smoke.py::"
        "test_run_quickstart_returns_failure_when_session_creation_fails"
    ) in onboarding
    assert (ROOT / "tests/test_quick_start_smoke.py").is_file()
