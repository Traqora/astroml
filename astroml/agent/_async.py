"""Small async helpers shared across the agent framework.

The agent core is async-first (LLM calls are IO bound, and the ingestion
stack in this repository is async as well), but most callers — CLI entry
points, notebooks, tests — prefer a blocking API.  :func:`run_sync` bridges
the two without exploding when a loop is already running.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run *coro* to completion and return its result.

    ``asyncio.run`` is used when the current thread has no running event
    loop.  If a loop is already running (for example inside a Jupyter
    notebook or an async test), the coroutine is executed on a dedicated
    worker thread that owns its own loop, which keeps the call blocking
    and side-effect free for the caller.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    results: list = []
    errors: list = []

    def _target() -> None:
        try:
            results.append(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            errors.append(exc)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()

    if errors:
        raise errors[0]
    return results[0]
