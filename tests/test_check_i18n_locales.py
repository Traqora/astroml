"""Tests for scripts/check_i18n_locales.py — issue #851.

docs/i18n/languages.yml is the single source of truth for which locales
exist, but nothing previously enforced that its entries match the actual
docs/i18n/<dir>/ folders. These pin the drift cases the checker is meant to
catch: a registered locale with a missing folder or README, a folder that
exists but was never registered, and an invalid progress value.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_i18n_locales.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_i18n_locales", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _write_locale_dir(tmp_path: Path, dir_name: str, code: str, readme: bool = True) -> None:
    locale_dir = tmp_path / dir_name
    locale_dir.mkdir(parents=True)
    if readme:
        (locale_dir / f"README.{code}.md").write_text("# stub\n", encoding="utf-8")


class TestLocaleDrift:
    def test_consistent_registry_reports_no_problems(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("# source\n", encoding="utf-8")
        _write_locale_dir(tmp_path, "es", "es")

        locales = [
            {"code": "en", "dir": ".", "progress": "complete", "source": True},
            {"code": "es", "dir": "es", "progress": "partial"},
        ]

        assert checker.find_locale_drift(tmp_path, locales) == []

    def test_registered_locale_with_missing_folder_is_reported(self, tmp_path: Path) -> None:
        locales = [{"code": "fr", "dir": "fr", "progress": "stub"}]

        problems = checker.find_locale_drift(tmp_path, locales)

        assert len(problems) == 1
        assert "fr" in problems[0]
        assert "does not exist" in problems[0]

    def test_registered_locale_with_missing_readme_is_reported(self, tmp_path: Path) -> None:
        _write_locale_dir(tmp_path, "es", "es", readme=False)
        locales = [{"code": "es", "dir": "es", "progress": "partial"}]

        problems = checker.find_locale_drift(tmp_path, locales)

        assert len(problems) == 1
        assert "README.es.md" in problems[0]

    def test_unregistered_folder_is_reported(self, tmp_path: Path) -> None:
        _write_locale_dir(tmp_path, "pt", "pt")

        problems = checker.find_locale_drift(tmp_path, [])

        assert len(problems) == 1
        assert "pt/" in problems[0]
        assert "not registered" in problems[0]

    def test_invalid_progress_value_is_reported(self, tmp_path: Path) -> None:
        _write_locale_dir(tmp_path, "es", "es")
        locales = [{"code": "es", "dir": "es", "progress": "done"}]

        problems = checker.find_locale_drift(tmp_path, locales)

        assert len(problems) == 1
        assert "progress 'done'" in problems[0]

    def test_source_locale_does_not_require_a_suffixed_readme(self, tmp_path: Path) -> None:
        # The English source lives at the repo root as README.md, not inside
        # docs/i18n/./README.en.md -- the source flag opts a locale out of
        # the per-locale README check.
        (tmp_path / ".").mkdir(exist_ok=True)
        locales = [{"code": "en", "dir": ".", "progress": "complete", "source": True}]

        assert checker.find_locale_drift(tmp_path, locales) == []

    def test_entry_missing_code_or_dir_is_reported_without_crashing(self, tmp_path: Path) -> None:
        locales = [{"progress": "stub"}]

        problems = checker.find_locale_drift(tmp_path, locales)

        assert len(problems) == 1
        assert "missing" in problems[0]


class TestLoadLocales:
    def test_loads_locales_list_from_real_languages_yml(self) -> None:
        locales = checker._load_locales(checker.LANGUAGES_FILE)

        codes = {locale["code"] for locale in locales}
        assert "en" in codes
        assert "es" in codes


class TestMainCli:
    def test_main_exits_zero_against_the_real_registry(self) -> None:
        assert checker.main([]) == 0

    def test_main_exits_one_when_languages_file_is_missing(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(checker, "LANGUAGES_FILE", tmp_path / "missing.yml")

        assert checker.main([]) == 1
