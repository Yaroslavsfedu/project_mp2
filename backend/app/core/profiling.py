"""Замер времени по стадиям пайплайна.

Выключен по умолчанию: без активной записи `stage()` ничего не делает (несколько наносекунд).
Запись включается `start_recording()`; контекст копируется в `asyncio.to_thread`, поэтому стадии,
выполняемые в рабочих потоках, попадают в тот же словарь.
Вложенные стадии считаются отдельно: суммируйте только непересекающиеся имена.
"""
from __future__ import annotations

import contextvars
import time
from collections.abc import Iterator
from contextlib import contextmanager

_recorder: contextvars.ContextVar[dict[str, float] | None] = contextvars.ContextVar(
    "pipeline_stage_recorder", default=None
)


@contextmanager
def stage(name: str) -> Iterator[None]:
    recorder = _recorder.get()
    if recorder is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        recorder[name] = recorder.get(name, 0.0) + (time.perf_counter() - started)


def start_recording() -> dict[str, float]:
    """Включает запись в текущем контексте и возвращает словарь {стадия: секунды}."""
    recorder: dict[str, float] = {}
    _recorder.set(recorder)
    return recorder


def stop_recording() -> None:
    _recorder.set(None)
