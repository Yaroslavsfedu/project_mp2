"""Определение аппаратного обеспечения и рекомендации по моделям.

Используется для честного отображения возможностей контура (CPU/GPU) и
подсказок, какую модель запускать на конкретной видеокарте.
"""
from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def gpu_info() -> dict:
    """Сведения о первой доступной NVIDIA GPU (nvidia-smi), без падений при её отсутствии."""
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "reason": "nvidia-smi не найден"}

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return {"available": False, "reason": str(exc)}

    first_line = (result.stdout or "").strip().splitlines()
    if result.returncode != 0 or not first_line:
        return {"available": False, "reason": "GPU не обнаружена"}

    parts = [part.strip() for part in first_line[0].split(",")]
    name = parts[0] if parts else "NVIDIA GPU"
    try:
        memory_mb = int(float(parts[1])) if len(parts) > 1 else 0
    except (TypeError, ValueError):
        memory_mb = 0
    driver = parts[2] if len(parts) > 2 else None

    return {
        "available": True,
        "name": name,
        "memory_mb": memory_mb,
        "driver": driver,
        "recommended_llm": recommend_llm(memory_mb),
        "recommended_vision": recommend_vision(memory_mb),
    }


def recommend_llm(vram_mb: int) -> str:
    """Подбор LLM под объём видеопамяти (Q4_K_M)."""
    if vram_mb >= 20_000:
        return "qwen2.5:14b-instruct"
    if vram_mb >= 7_000:
        return "qwen2.5:7b-instruct"
    if vram_mb >= 4_500:
        return "qwen2.5:3b-instruct"
    return "qwen2.5:1.5b-instruct"


def recommend_vision(vram_mb: int) -> str:
    if vram_mb >= 11_000:
        return "qwen2-vl:7b"
    if vram_mb >= 7_000:
        return "llava:7b"
    return "qwen2.5vl:3b"


def cpu_info() -> dict:
    try:
        import os

        return {
            "cores": os.cpu_count() or 1,
            "note": "CPU-режим: рекомендуется qwen2.5:3b-instruct или 1.5b",
        }
    except Exception:  # noqa: BLE001 - pragma: no cover
        return {"cores": 1}
