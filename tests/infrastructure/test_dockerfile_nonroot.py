"""Non-root container verification for the root Dockerfile (#867).

Every build stage in the root ``Dockerfile`` that is actually run as a
container (i.e. declares a ``CMD``/``ENTRYPOINT``) must execute as a
non-root user by the time that instruction is reached — whether because
the stage sets ``USER <non-root>`` itself, or because it inherits one from
its ``FROM <stage>`` base stage.

This is a *regression* test, not a parser library: a small line-by-line
state machine walks the Dockerfile once, recording for every named stage
the user in effect at each ``CMD``/``ENTRYPOINT`` plus the user in effect
at the *end* of the stage (for stages a later one inherits ``FROM``). A
stage that never issues its own ``USER`` resolves to whatever its base
stage ended on; a stage whose base is an *external* image (e.g.
``python:3.11-slim``, ``nvidia/cuda:...``) has no visible Dockerfile text
to resolve, so it is conservatively treated as root until the stage sets
its own ``USER``.

Per the issue's own investigation notes, every runnable stage in the
current Dockerfile (``ingestion``, ``training-base``, ``training-cpu``,
``development``, ``feature-store``, ``production``, and ``training`` via
inheritance from ``training-base``) already does this correctly. This
test exists to catch a *future* regression — e.g. a new stage that adds a
``CMD`` but forgets ``USER`` — not to fix anything in the Dockerfile today.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"

_FROM_RE = re.compile(r"^FROM\s+(?P<base>\S+)(?:\s+[Aa][Ss]\s+(?P<alias>\S+))?\s*$", re.IGNORECASE)
_USER_RE = re.compile(r"^USER\s+(?P<user>\S+)\s*$", re.IGNORECASE)
_RUNS_RE = re.compile(r"^(CMD|ENTRYPOINT)\b", re.IGNORECASE)

# A USER value that means "running as root" — the bare name, or UID 0,
# with or without a trailing ":<group>".
_ROOT_USERS = {"root", "0"}


def _is_root_user(user: str | None) -> bool:
    """True if `user` names root. None means "not yet set in this stage"."""
    if user is None:
        return True
    return user.split(":", 1)[0] in _ROOT_USERS


@dataclass
class _Stage:
    name: str
    base: str  # raw FROM token: a local stage name, or an external image ref
    # One entry per CMD/ENTRYPOINT: the local USER value in effect at that
    # point (None if this stage hadn't set USER yet when it ran).
    user_at_run: list[str | None] = field(default_factory=list)
    # The local USER value in effect at the end of the stage (what a child
    # stage inherits if it issues no USER of its own). None if this stage
    # never set USER at all.
    final_user: str | None = None


def _parse_stages(text: str) -> dict[str, _Stage]:
    stages: dict[str, _Stage] = {}
    current: _Stage | None = None
    # True while the next physical line is a continuation of the previous
    # instruction (e.g. HEALTHCHECK's own `CMD ...` sub-clause, or any
    # instruction broken across lines with a trailing `\`) rather than a
    # new Dockerfile instruction in its own right.
    is_continuation = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        this_line_is_continuation = is_continuation
        # Determine whether the *next* line will be a continuation of this
        # one before we strip/consume it.
        is_continuation = line.endswith("\\")

        if not line or line.startswith("#") or this_line_is_continuation:
            continue

        from_match = _FROM_RE.match(line)
        if from_match:
            base = from_match.group("base")
            alias = from_match.group("alias") or f"__anon_{len(stages)}"
            current = _Stage(name=alias, base=base)
            stages[alias] = current
            continue

        if current is None:
            continue  # lines before the first FROM (comments/blank only)

        user_match = _USER_RE.match(line)
        if user_match:
            current.final_user = user_match.group("user")
            continue

        if _RUNS_RE.match(line):
            current.user_at_run.append(current.final_user)
            continue

    return stages


def _resolve_final_user_is_root(
    stage_name: str, stages: dict[str, _Stage], _visiting: frozenset[str] = frozenset()
) -> bool:
    """Resolve whether `stage_name` ends up root, following FROM-based
    inheritance through local stages whose own `final_user` is unset."""
    if stage_name in _visiting:
        return True  # cycle guard; not reachable in a valid Dockerfile
    stage = stages.get(stage_name)
    if stage is None:
        # `stage_name` is an external image (python:3.11-slim, nvidia/cuda:
        # ...), not a local stage we have text for. We can't know its
        # default user, so treat it conservatively as root.
        return True
    if stage.final_user is not None:
        return _is_root_user(stage.final_user)
    return _resolve_final_user_is_root(stage.base, stages, _visiting | {stage_name})


def _effective_root_at_run(index: int, stage: _Stage, stages: dict[str, _Stage]) -> bool:
    """Whether the stage was running as root at its `index`-th CMD/ENTRYPOINT,
    resolving inheritance from the base stage only when this stage had not
    yet issued its own USER at that point."""
    local_user = stage.user_at_run[index]
    if local_user is not None:
        return _is_root_user(local_user)
    return _resolve_final_user_is_root(stage.base, stages)


# Stages the issue's investigation confirmed are both runnable (declare a
# CMD/ENTRYPOINT) and already correctly non-root as of this writing.
KNOWN_RUNNABLE_NONROOT_STAGES = {
    "ingestion",
    "training-base",
    "training-cpu",
    "development",
    "feature-store",
    "production",
    "training",
}


def test_dockerfile_has_expected_stages() -> None:
    """Sanity check: the parser must see the stages this test is built
    around, or it is silently testing nothing."""
    stages = _parse_stages(DOCKERFILE.read_text(encoding="utf-8"))
    expected = KNOWN_RUNNABLE_NONROOT_STAGES | {"base"}
    missing = expected - stages.keys()
    assert not missing, f"Dockerfile parser did not find expected stages: {missing}"


def test_every_runnable_stage_ends_up_non_root() -> None:
    """Every stage with a CMD/ENTRYPOINT must be non-root at that point,
    whether set directly or inherited from its base stage. This is the
    regression guard: a future stage that adds a CMD but forgets USER (and
    whose base doesn't already provide one) fails here."""
    stages = _parse_stages(DOCKERFILE.read_text(encoding="utf-8"))

    failures = [
        f"stage '{name}' runs CMD/ENTRYPOINT #{idx + 1} as root "
        f"(base='{stage.base}') — add `USER <non-root>` before it"
        for name, stage in stages.items()
        for idx in range(len(stage.user_at_run))
        if _effective_root_at_run(idx, stage, stages)
    ]

    assert (
        not failures
    ), "Dockerfile stage(s) run as root at a CMD/ENTRYPOINT (#867):\n" + "\n".join(failures)


def test_known_nonroot_stages_are_present_and_passing() -> None:
    """Pin the specific stages the issue's investigation verified, so a
    future edit that silently removes or renames one of these stages
    (rather than leaving it root) is still caught by name."""
    stages = _parse_stages(DOCKERFILE.read_text(encoding="utf-8"))

    for name in KNOWN_RUNNABLE_NONROOT_STAGES:
        stage = stages.get(name)
        assert stage is not None, f"expected stage '{name}' to still exist in the Dockerfile"
        assert stage.user_at_run, f"expected stage '{name}' to declare a CMD/ENTRYPOINT"
        for idx in range(len(stage.user_at_run)):
            assert not _effective_root_at_run(
                idx, stage, stages
            ), f"stage '{name}' should be non-root at CMD/ENTRYPOINT #{idx + 1}"


def test_base_builder_stage_has_no_cmd_despite_staying_root() -> None:
    """`base` never switches USER (it needs root for apt/pip installs) and
    never declares a CMD — it is a builder-only intermediate stage, not a
    runnable image, so staying root is intentional. If it ever gains a
    CMD/ENTRYPOINT without also gaining a USER switch, this is now a real
    violation: `test_every_runnable_stage_ends_up_non_root` would catch it,
    but this test pins the current, intentional shape explicitly so a
    change here is a deliberate decision, not a silent one."""
    stages = _parse_stages(DOCKERFILE.read_text(encoding="utf-8"))
    base = stages["base"]
    assert _resolve_final_user_is_root("base", stages) is True, (
        "'base' is expected to stay root internally; if this changes, "
        "update this test's assumptions"
    )
    assert not base.user_at_run, (
        "'base' gained a CMD/ENTRYPOINT while still root — add `USER "
        "<non-root>` before it, this is a real non-root violation (#867)"
    )


def test_training_stage_inherits_nonroot_from_training_base() -> None:
    """The one stage in this Dockerfile that relies purely on inheritance
    (no local USER of its own): `training` FROM training-base. Pinned
    separately so a regression in training-base's non-root switch has an
    unambiguous failure point."""
    stages = _parse_stages(DOCKERFILE.read_text(encoding="utf-8"))
    training = stages["training"]
    assert training.base == "training-base"
    assert training.final_user is None, (
        "expected 'training' to set no USER of its own — test assumption "
        "changed, update this test"
    )
    assert _resolve_final_user_is_root("training-base", stages) is False, (
        "'training-base' no longer resolves to non-root — 'training' "
        "would silently inherit a root user"
    )
