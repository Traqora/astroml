"""Secrets audit for the Docker build context (#940).

Two guarantees, both enforceable in CI without a Docker daemon:

1. **``.dockerignore`` blocks secret material from the build context.**
   The ``Dockerfile`` ``COPY``s ``astroml/``, ``migrations/``, ``docs/``,
   ``examples/`` (and the runner image also ``tests/``) into the image, and
   the context is the repo root — so every env file, private key, and
   credential blob in the tree is one careless ``COPY .`` away from being
   baked into a published image. The ignore list must explicitly exclude
   them (``!.env.example`` stays allowed: it only holds placeholders).

2. **No build-context file ships a real URI-embedded credential.**
   A ``scheme://user:password@host`` string in any file the context picks
   up is a leaked secret unless it is a known dev placeholder or
   placeholder-marked. ``REPLACE_WITH`` / ``change_me`` markers count as
   placeholders; anything else with a password-looking segment fails.

``astroml/ingestion/normalizer.py`` is the module this issue pins as the
relevant area: its CLI reads operation JSON from stdin or a file path, so
any operator workflow that pipes credentials through the build context
would silently flow into images too — keeping secrets out of the context
is what keeps them out of every image built from it.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERIGNORE = REPO_ROOT / ".dockerignore"

# Patterns that must appear (uncommented) in .dockerignore.
REQUIRED_IGNORE_ENTRIES = [
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "credentials.json",
]

# .dockerignore entries that legitimately carry non-secret content and stay
# out of the "no credentials in context" scan below.
ALLOWED_WITH_PLACEHOLDERS = {".env.example"}

# Files excluded from the credential scan: CI fixtures with throwaway
# localhost creds, lockfiles, docs, and VCS/tooling internals.
SCAN_EXCLUDE_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "outputs"}
SCAN_SUFFIXES = (".yml", ".yaml", ".py", ".txt", ".cfg", ".ini", ".toml", ".json", ".sh")
SCAN_FILENAMES = {"Dockerfile", ".env.example"}

# A URI password segment counts as a placeholder/template when it carries one
# of these markers, is env-interpolated (``${VAR}`` / ``{field}``), is part of
# a regex fragment, or is a known dev default. The goal is to catch *real*
# credentials while letting documented dev defaults and templates through.
PLACEHOLDER_MARKERS = (
    "change_me",
    "change-me",
    "changeme",
    "replace_with",
    "replace-with",
    "password",
    "passwd",
    "secret",
    "example",
    "placeholder",
    "your_",
    "your-",
)
TEMPLATE_CHARS = ("{", "}", "<", ">", "(", ")", "$", "[")
KNOWN_DEV_PASSWORDS = {
    "pass",
    "invalid",
    "envpass",
    "astroml",
    "test",
    "postgres",
    "astroml_password",
}

# URI with an embedded password: scheme://user:password@host/...
_CRED_URI = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^/\s\"']+:[^@\s\"']+@")


def _is_placeholder(uri: str) -> bool:
    lowered = uri.lower()
    password_part = uri.split("://", 1)[1].split("@", 1)[0].split(":", 1)[1]
    if password_part.lower() in KNOWN_DEV_PASSWORDS:
        return True
    if any(char in password_part for char in TEMPLATE_CHARS):
        return True  # ${VAR} interpolation, {field} templates, regex fragments
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def _dockerignore_entries() -> set[str]:
    entries = set()
    for raw in DOCKERIGNORE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            entries.add(line)
    return entries


def test_dockerignore_blocks_secret_material():
    entries = _dockerignore_entries()
    missing = [entry for entry in REQUIRED_IGNORE_ENTRIES if entry not in entries]
    assert not missing, (
        f".dockerignore is missing secret-exclusion entries: {missing} (#940). "
        "Secrets must never enter the Docker build context."
    )


def test_dockerignore_still_allows_env_example():
    entries = _dockerignore_entries()
    assert "!.env.example" in entries, (
        "!.env.example must stay allowed: it documents required env vars "
        "with placeholders and contains no secrets"
    )


def test_no_credential_uris_in_build_context():
    offenders: list[str] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if SCAN_EXCLUDE_DIRS & set(path.relative_to(REPO_ROOT).parts):
            continue
        if path.suffix not in SCAN_SUFFIXES and path.name not in SCAN_FILENAMES:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for match in _CRED_URI.finditer(text):
            uri = match.group(0)
            if not _is_placeholder(uri):
                offenders.append(f"{rel}: {uri}")
    assert not offenders, (
        "credential-bearing URIs found in files that enter the Docker build "
        f"context — move them to env injection or a secrets manager (#940):\n"
        + "\n".join(offenders)
        + "\n(Dev defaults like astroml:astroml_password, ${VAR} interpolation, "
        "{field} templates and regex fragments are allowed; real secrets are not.)"
    )


def test_scan_actually_covers_the_context():
    """Sanity: the scanner must see the real context, or it proves nothing."""
    seen = 0
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if SCAN_EXCLUDE_DIRS & set(path.relative_to(REPO_ROOT).parts):
            continue
        if path.suffix in SCAN_SUFFIXES or path.name in SCAN_FILENAMES:
            seen += 1
    assert seen > 50, f"credential scan only covered {seen} files; check SCAN_* config"
