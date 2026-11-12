"""Замер производительности пайплайна на конкретной машине (CPU или GPU).

Скрипт даёт честные локальные цифры: сколько секунд занимает обработка одного
документа, сколько — вызовы LLM и эмбеддингов, какая модель и GPU реально
используются. Запускайте после `ollama pull`, чтобы первый холодный старт
(загрузка весов в VRAM) был учтён отдельно.

CLI:
    cd backend
    PYTHONPATH=. python -m scripts.benchmark                  # демо-документы
    PYTHONPATH=. python -m scripts.benchmark --limit 3 --repeat 2
    PYTHONPATH=. python -m scripts.benchmark --out ../reports/benchmark.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"

_TMP_DIR = Path(tempfile.mkdtemp(prefix="idp-bench-"))
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{_TMP_DIR}/bench.db")
os.environ.setdefault("DATA_DIR", str(_TMP_DIR / "data"))
os.environ.setdefault("SEED_DEMO", "false")
os.environ.setdefault("QUEUE_BACKEND", "inline")

from sqlalchemy import select  # noqa: E402

from app.core.database import SessionLocal, init_db  # noqa: E402
from app.core.hardware import cpu_info, gpu_info  # noqa: E402
from app.ml_pipeline.llm_service import analyze_document, embed_texts, llm_status  # noqa: E402
from app.ml_pipeline.ocr_engine import resolve_engine  # noqa: E402
from app.models.document import Document, DocumentStatus  # noqa: E402
from app.models.user import User, UserRole  # noqa: E402
from app.services.pipeline_service import process_document  # noqa: E402
from app.services.storage_service import original_key, storage  # noqa: E402
from app.core.profiling import start_recording, stop_recording  # noqa: E402

BENCH_TEXT = (
    "ДОГОВОР ПОСТАВКИ № 47/2024. г. Ростов-на-Дону, 12 марта 2024 г. "
    "ООО «ТехноСервис», ИНН 6165048312, в лице генерального директора Кузнецовой Анны Сергеевны, "
    "обязуется поставить оборудование на сумму 1 250 000 руб. 00 коп. в срок до 26 апреля 2024 г. "
    "За просрочку поставки поставщик уплачивает неустойку в размере 0,1 % за каждый день просрочки."
)


async def ensure_user() -> User:
    async with SessionLocal() as session:
        user = await session.scalar(select(User).where(User.email == "bench@idp.local"))
        if user is None:
            user = User(
                email="bench@idp.local",
                password_hash="-",
                full_name="Бенчмарк",
                role=UserRole.ADMIN,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
        return user


async def run_document(path: Path, user: User) -> dict:
    data = path.read_bytes()
    document_id = str(uuid.uuid4())
    await storage.put(original_key(document_id, path.name), data)

    async with SessionLocal() as session:
        session.add(
            Document(
                id=document_id,
                user_id=user.id,
                filename=path.name,
                s3_key_original=original_key(document_id, path.name),
                mime_type="application/octet-stream",
                file_size=len(data),
                status=DocumentStatus.PENDING,
            )
        )
        await session.commit()

    recorder = start_recording()
    started = time.perf_counter()
    try:
        await process_document(document_id)
    finally:
        duration = time.perf_counter() - started
        stop_recording()

    async with SessionLocal() as session:
        document = await session.scalar(select(Document).where(Document.id == document_id))
        return {
            "filename": path.name,
            "size_kb": round(len(data) / 1024, 1),
            "pages": document.page_count if document else 0,
            "status": document.status if document else "unknown",
            "duration_sec": round(duration, 3),
            "stages_sec": {k: round(v, 4) for k, v in sorted(recorder.items())},
        }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Замер производительности IDP-пайплайна")
    parser.add_argument("--samples", default=str(PROJECT_ROOT / "samples"))
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=1, help="повторов на документ (для стабильности)")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "benchmark.json"))
    args = parser.parse_args()

    samples = Path(args.samples)
    files = sorted(
        path
        for path in samples.iterdir()
        if path.suffix.lower() in {".pdf", ".docx", ".txt", ".jpg", ".jpeg", ".png"}
        and not path.name.startswith("scan_")
    )[: args.limit]
    if not files:
        print(f"Нет документов в {samples}")
        return 1

    await init_db()
    user = await ensure_user()
    status = await llm_status()
    ocr_engine = await resolve_engine()

    print("=== Окружение ===")
    print(f"LLM:          {status['mode']} ({status['model']})")
    print(f"Эмбеддинги:   {status['embedding_model']}")
    print(f"OCR-движок:   {ocr_engine}")
    gpu = gpu_info()
    if gpu.get("available"):
        print(f"GPU:          {gpu['name']} · {gpu['memory_mb']} МБ VRAM")
    else:
        print(f"GPU:          не обнаружена ({gpu.get('reason', '—')}) · CPU ядер: {cpu_info()['cores']}")
    print()

    # --- изолированные замеры компонентов ИИ ---
    llm_times: list[float] = []
    embed_times: list[float] = []
    for _ in range(max(args.repeat, 1)):
        started = time.perf_counter()
        await analyze_document(BENCH_TEXT)
        llm_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        await embed_texts(["неустойка за просрочку поставки"])
        embed_times.append(time.perf_counter() - started)

    print("=== Компоненты ИИ ===")
    print(f"Анализ документа:   {[round(t, 2) for t in llm_times]} с")
    print(f"Эмбеддинги:         {[round(t, 3) for t in embed_times]} с")
    print()

    # --- сквозная обработка ---
    rows: list[dict] = []
    print("=== Документы ===")
    for path in files:
        best = None
        for _ in range(max(args.repeat, 1)):
            result = await run_document(path, user)
            if best is None or result["duration_sec"] < best["duration_sec"]:
                best = result
        rows.append(best)
        print(
            f"{best['filename']:<32} {best['size_kb']:>7} КБ  "
            f"{best['pages']} стр.  {best['duration_sec']:>6.2f} с  {best['status']}"
        )

    total = sum(row["duration_sec"] for row in rows)
    stage_totals: dict[str, float] = {}
    for row in rows:
        for name, value in row.get("stages_sec", {}).items():
            stage_totals[name] = stage_totals.get(name, 0.0) + value
    top_level = sum(v for k, v in stage_totals.items() if "/" not in k)
    stage_report = {
        "per_document_avg_sec": {
            k: round(v / len(rows), 4) for k, v in sorted(stage_totals.items(), key=lambda kv: -kv[1])
        },
        "top_level_tracked_sec": round(top_level, 3),
        "untracked_sec": round(total - top_level, 3),
        "untracked_share": round((total - top_level) / total, 3) if total else None,
        "note": "имена с '/' — вложенные стадии внутри верхнего уровня; в сумму не входят",
    }
    pages = sum(row["pages"] for row in rows)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": {
            "llm_mode": status["mode"],
            "llm_model": status["model"],
            "embedding_model": status["embedding_model"],
            "ocr_engine": ocr_engine,
            "gpu": gpu,
            "cpu": cpu_info(),
        },
        "components_sec": {
            "llm_analyze": [round(t, 3) for t in llm_times],
            "embedding": [round(t, 3) for t in embed_times],
        },
        "documents": rows,
        "stages": stage_report,
        "summary": {
            "documents": len(rows),
            "pages": pages,
            "total_sec": round(total, 2),
            "avg_sec_per_document": round(total / len(rows), 2),
            "avg_sec_per_page": round(total / pages, 2) if pages else None,
            "docs_per_minute": round(60 / (total / len(rows)), 1),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=== Итог ===")
    print(f"Документов: {len(rows)} · страниц: {pages}")
    print(f"Среднее на документ: {report['summary']['avg_sec_per_document']} с")
    print(f"Пропускная способность: {report['summary']['docs_per_minute']} док/мин")
    print(f"Отчёт: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
