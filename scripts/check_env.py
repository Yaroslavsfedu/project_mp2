#!/usr/bin/env python3
"""Проверка готовности окружения к запуску IDP-платформы (без внешних зависимостей).

Запуск:  python scripts/check_env.py            # человекочитаемый отчёт
         python scripts/check_env.py --json     # машинный вывод
         python scripts/check_env.py --strict   # exit code 1 при критичных проблемах

Скрипт ничего не устанавливает — только проверяет и подсказывает команды.
Работает на чистом Python 3.9+ (stdlib), поэтому его можно запускать до pip install.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-m3")
VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "qwen2-vl:7b")

OK, WARN, FAIL = "ok", "warn", "fail"
ICONS = {OK: "✓", WARN: "!", FAIL: "✗"}


def check_python() -> dict:
    version = sys.version_info
    status = OK if version >= (3, 11) else FAIL
    return {
        "name": "Python",
        "status": status,
        "value": f"{version.major}.{version.minor}.{version.micro}",
        "hint": "" if status == OK else "Нужен Python 3.11+ (python.org или pyenv)",
        "critical": True,
    }


def check_node() -> dict:
    binary = shutil.which("node")
    if not binary:
        return {
            "name": "Node.js",
            "status": FAIL,
            "value": "не найден",
            "hint": "Установите Node.js 20+ (nodejs.org) для запуска фронтенда",
            "critical": True,
        }
    try:
        version = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        version = "?"
    major = 0
    try:
        major = int(version.lstrip("v").split(".")[0])
    except ValueError:
        pass
    status = OK if major >= 20 else WARN
    return {
        "name": "Node.js",
        "status": status,
        "value": version,
        "hint": "" if status == OK else "Рекомендуется Node.js 20+ (у вас более старая версия)",
        "critical": True,
    }


def check_tesseract() -> dict:
    binary = shutil.which("tesseract")
    if not binary:
        return {
            "name": "Tesseract OCR",
            "status": WARN,
            "value": "не найден",
            "hint": (
                "Для сканов/фото: apt install tesseract-ocr tesseract-ocr-rus "
                "(Windows: UB-Mannheim installer + rus.traineddata). "
                "PDF с текстовым слоем, DOCX и TXT работают и без него."
            ),
            "critical": False,
        }
    try:
        info = subprocess.run(
            [binary, "--list-langs"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        info = ""
    languages = [line.strip() for line in info.splitlines()[1:] if line.strip()]
    has_russian = "rus" in languages
    return {
        "name": "Tesseract OCR",
        "status": OK if has_russian else WARN,
        "value": f"языки: {', '.join(languages[:6]) or '—'}",
        "hint": "" if has_russian else "Не найден языковой пакет rus (tesseract-ocr-rus)",
        "critical": False,
    }


def _http_json(url: str, timeout: float = 2.5):
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def check_ollama() -> dict:
    try:
        tags = _http_json(f"{OLLAMA_URL}/api/tags")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {
            "name": "Ollama (локальная LLM)",
            "status": WARN,
            "value": f"недоступна ({OLLAMA_URL})",
            "hint": (
                "Установите Ollama (ollama.com) и запустите `ollama serve`. "
                "Без неё работает встроенный эвристический fallback-движок."
            ),
            "details": str(exc),
            "critical": False,
        }
    installed = [item.get("name", "") for item in tags.get("models", [])]

    def present(model: str) -> bool:
        base = model.split(":")[0]
        return any(name == model or name.split(":")[0] == base for name in installed)

    missing = [
        model
        for model in (OLLAMA_MODEL, EMBEDDING_MODEL, VISION_MODEL)
        if not present(model)
    ]
    return {
        "name": "Ollama (локальная LLM)",
        "status": OK if not missing else WARN,
        "value": f"моделей установлено: {len(installed)}",
        "installed": installed,
        "missing": missing,
        "hint": (
            ""
            if not missing
            else "Загрузите модели: " + " && ".join(f"ollama pull {m}" for m in missing)
        ),
        "critical": False,
    }


def check_gpu() -> dict:
    if shutil.which("nvidia-smi") is None:
        return {
            "name": "GPU (NVIDIA)",
            "status": WARN,
            "value": "nvidia-smi не найден",
            "hint": "Без GPU LLM будет работать на CPU: используйте qwen2.5:3b-instruct или 1.5b",
            "critical": False,
        }
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        output = ""
    if not output:
        return {
            "name": "GPU (NVIDIA)",
            "status": WARN,
            "value": "не обнаружена",
            "hint": "Проверьте драйвер NVIDIA и nvidia-container-toolkit для Docker",
            "critical": False,
        }
    parts = [part.strip() for part in output.splitlines()[0].split(",")]
    try:
        vram = int(float(parts[1])) if len(parts) > 1 else 0
    except (TypeError, ValueError):
        vram = 0
    if vram >= 20000:
        recommended = "qwen2.5:14b-instruct"
    elif vram >= 7000:
        recommended = "qwen2.5:7b-instruct"
    elif vram >= 4500:
        recommended = "qwen2.5:3b-instruct"
    else:
        recommended = "qwen2.5:1.5b-instruct"
    return {
        "name": "GPU (NVIDIA)",
        "status": OK,
        "value": f"{parts[0]}, {vram} МБ VRAM, драйвер {parts[2] if len(parts) > 2 else '—'}",
        "recommended_llm": recommended,
        "hint": f"Рекомендуемая LLM для этой карты: {recommended}",
        "critical": False,
    }


def check_samples() -> dict:
    samples = ROOT / "samples"
    truth = samples / "ground_truth.json"
    files = [p.name for p in samples.glob("*") if p.suffix.lower() in {".pdf", ".docx", ".txt", ".jpg"}]
    if not files:
        return {
            "name": "Демо-документы",
            "status": WARN,
            "value": "нет файлов в samples/",
            "hint": "Сгенерируйте: cd backend && PYTHONPATH=. python -m scripts.generate_samples --out ../samples",
            "critical": False,
        }
    return {
        "name": "Демо-документы",
        "status": OK if truth.exists() else WARN,
        "value": f"{len(files)} файлов, эталон: {'есть' if truth.exists() else 'нет'}",
        "hint": "" if truth.exists() else "Для метрик качества нужен samples/ground_truth.json",
        "critical": False,
    }


def check_disk() -> dict:
    try:
        usage = shutil.disk_usage(ROOT)
        free_gb = usage.free / 1024**3
    except OSError:
        free_gb = 0
    return {
        "name": "Свободное место",
        "status": OK if free_gb >= 5 else WARN,
        "value": f"{free_gb:.1f} ГБ",
        "hint": "" if free_gb >= 5 else "Для весов LLM (5–15 ГБ) может не хватить места",
        "critical": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка окружения IDP-платформы")
    parser.add_argument("--json", action="store_true", help="вывод в JSON")
    parser.add_argument("--strict", action="store_true", help="exit 1 при критичных проблемах")
    args = parser.parse_args()

    checks = [
        check_python(),
        check_node(),
        check_gpu(),
        check_tesseract(),
        check_ollama(),
        check_samples(),
        check_disk(),
    ]

    if args.json:
        print(json.dumps(checks, ensure_ascii=False, indent=2))
    else:
        print("\nПроверка окружения IDP-платформы\n" + "=" * 44)
        for check in checks:
            print(f"[{ICONS[check['status']]}] {check['name']:<24} {check['value']}")
            if check.get("hint") and check["status"] != OK:
                print(f"    → {check['hint']}")
        print("=" * 44)
        print(
            "Дальше:  ./scripts/bootstrap.sh   (установка)   ·   make dev   (запуск)"
        )

    if args.strict and any(check["status"] == FAIL for check in checks):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
