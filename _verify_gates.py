"""Local CI-parity verification helper (temporary; deleted before finishing)."""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(r"c:\Users\MENDOS\Desktop\astroml")
OUT = ROOT / "_gates_result.txt"
log = OUT.open("w", encoding="utf-8")


def w(msg: str) -> None:
    log.write(msg + "\n")
    log.flush()


target = ROOT / "astroml" / "cache" / "graph_cache.py"
head_bytes = subprocess.run(
    ["git", "show", "HEAD:astroml/cache/graph_cache.py"],
    cwd=ROOT,
    capture_output=True,
).stdout
local_bytes = target.read_bytes()

w(f"graph_cache.py unchanged vs HEAD: {local_bytes == head_bytes}")
for label, blob in (("HEAD blob", head_bytes), ("local file", local_bytes)):
    for enc in ("utf-8", "cp1252"):
        try:
            ast.parse(blob.decode(enc))
            w(f"{label} parses OK as {enc}")
        except SyntaxError as exc:
            w(f"{label} as {enc}: SyntaxError: {exc}")

lines = local_bytes.split(b"\n")
for idx in (10, 11, 12):
    w(f"  raw line {idx + 1}: {lines[idx]!r}")

w("")
w("--- mypy (strict, repo config) on my module ---")
proc = subprocess.run(
    [sys.executable, "-m", "mypy", "astroml/tracking/model_registry.py"],
    cwd=ROOT,
    capture_output=True,
    text=True,
)
w(proc.stdout.strip() or "(no stdout)")
w(proc.stderr.strip() or "(no stderr)")
w(f"mypy exit={proc.returncode}")

log.close()
print("done")
