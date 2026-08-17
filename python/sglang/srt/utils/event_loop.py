"""Select the accelerated asyncio backend for the current platform."""

import asyncio
import importlib
import sys
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar

_T = TypeVar("_T")


@dataclass(frozen=True)
class EventLoopConfig:
    backend: str
    uvicorn_loop: str
    granian_loop: str


def resolve_event_loop_config(platform: str) -> EventLoopConfig:
    if platform == "win32":
        # Uvicorn has no named Winloop backend. "none" preserves the policy
        # installed by install_event_loop(), while Granian supports it directly.
        return EventLoopConfig(
            backend="winloop",
            uvicorn_loop="none",
            granian_loop="winloop",
        )
    return EventLoopConfig(
        backend="uvloop",
        uvicorn_loop="uvloop",
        granian_loop="uvloop",
    )


EVENT_LOOP_CONFIG = resolve_event_loop_config(sys.platform)


def _event_loop_module():
    return importlib.import_module(EVENT_LOOP_CONFIG.backend)


def install_event_loop() -> None:
    backend = _event_loop_module()
    asyncio.set_event_loop_policy(backend.EventLoopPolicy())


def run_event_loop(main: Coroutine[Any, Any, _T]) -> _T:
    return _event_loop_module().run(main)
