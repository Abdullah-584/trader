"""Sync->async bridge for the (async-only) QuotexAPI library.

The rest of this bot is synchronous; QuotexAPI is async-first. This bridge
runs coroutines on a private event loop from sync code, and stays correct
when called from inside a running loop (Streamlit, tests) by delegating to
a worker thread instead of exploding.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading

_loop: asyncio.AbstractEventLoop | None = None
_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, daemon=True,
                             name="quotex-async-loop").start()
        return _loop


def run_async(coro, timeout: float = 60.0):
    """Run ``coro`` to completion from sync code and return its result.

    * From a normal (non-async) context: schedules on the private loop.
    * From inside a running event loop (Streamlit rerun, async tests):
      delegates to a worker thread running its own loop, because the
      caller's loop cannot be blocked.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        fut = asyncio.run_coroutine_threadsafe(coro, _get_loop())
        return fut.result(timeout=timeout)

    # caller is inside a running loop: use a side thread
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result(timeout=timeout)
