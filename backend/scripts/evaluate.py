"""Оценка качества пайплайна на эталонном датасете (CER / WER / NER F1).

Скрипт прогоняет документы из `samples/ground_truth.json` через настоящий пайплайн
(препроцессинг → OCR/текстовый слой → локальный ИИ) в изолированной временной БД,
считает метрики и сохраняет отчёт `quality.json` + `quality.md`.

CLI:
    cd backend
    PYTHONPATH=. python -m scripts.evaluate                       # демо-датасет
    PYTHONPATH=. python -m scripts.evaluate --limit 3 --quiet
    PYTHONPATH=. python -m scripts.evaluate --dataset ../samples/ground_truth.json \
        --out ../reports/quality

Пути к БД и хранилищу можно переопределить через DATABASE_URL / DATA_DIR —
иначе используются временные каталоги, чтобы не засорять рабочую базу.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"

_TMP_DIR = Path(tempfile.mkdtemp(prefix="idp-eval-"))
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{_TMP_DIR}/eval.db")
os.environ.setdefault("DATA_DIR", str(_TMP_DIR / "data"))
os.environ.setdefault("SEED_DEMO", "false")
os.environ.setdefault("QUEUE_BACKEND", "inline")

from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import selectinload  # noqa: E402

from app.core.database import SessionLocal, init_db  # noqa: E402
from app.ml_pipeline.llm_service import llm_status  # noqa: E402
from app.ml_pipeline.ocr_engine import available_engine, resolve_engine  # noqa: E402
from app.models.document import Document, DocumentStatus  # noqa: E402
from app.models.user import User, UserRole  # noqa: E402
from app.services import quality_service  # noqa: E402
from app.services.pipeline_service import process_document  # noqa: E402
from app.services.storage_service import storage  # noqa: E402


async def ensure_eval_user() -> User:
    async with SessionLocal() as session:
        user = await session.scalar(select(User).where(User.email == "eval@idp.local"))
        if user is None:
            user = User(
                email="eval@idp.local",
                password_hash="-",
                full_name="Оценка качества",
                role=UserRole.ADMIN,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
        return user


async def evaluate_document(item: dict, samples_dir: Path, user: User) -> dict:
    filename = item["filename"]
    source = None
    for candidate in (samples_dir / filename, PROJECT_ROOT / "samples" / filename):
        if candidate.exists():
            source = candidate
            break
    if source is None:
        return {
            "filename": filename,
            "error": "файл не найден",
            "cer": 1.0,
            "wer": 1.0,
            "reference_entities": item.get("entities", {}),
            "predicted_entities": {},
        }

    data = source.read_bytes()
    document_id = str(uuid.uuid4())
    key = f"originals/{document_id}/{source.name}"
    await storage.put(key, data)

    async with SessionLocal() as session:
        session.add(
            Document(
                id=document_id,
                user_id=user.id,
                filename=source.name,
                s3_key_original=key,
                mime_type="application/pdf" if source.suffix == ".pdf" else "application/octet-stream",
                file_size=len(data),
                status=DocumentStatus.PENDING,
            )
        )
        await session.commit()

    started = time.perf_counter()
    await process_document(document_id)
    duration = time.perf_counter() - started

    async with SessionLocal() as session:
        document = await session.scalar(
            select(Document)
            .options(selectinload(Document.pages), selectinload(Document.extracted))
            .where(Document.id == document_id)
        )
        if document is None:
            return {"filename": filename, "error": "документ не создан", "cer": 1.0, "wer": 1.0}

        text = "\n".join(
            (page.corrected_text or page.raw_text or "")
            for page in sorted(document.pages, key=lambda p: p.page_number)
        )
        predicted_entities = (document.extracted.entities if document.extracted else {}) or {}
        reference_text = item.get("text", "")
        reference_entities = item.get("entities", {})

        result = {
            "filename": filename,
            "status": document.status,
            "error": document.error_message,
            "pages": len(document.pages),
            "duration_sec": round(duration, 3),
            "cer": round(quality_service.cer(reference_text, text), 4),
            "wer": round(quality_service.wer(reference_text, text), 4),
            "doc_type_expected": item.get("doc_type"),
            "doc_type_predicted": document.doc_type,
            "doc_type_correct": (document.doc_type or "") == (item.get("doc_type") or ""),
            "reference_entities": reference_entities,
            "predicted_entities": predicted_entities,
            "llm_engine": document.extracted.engine if document.extracted else "—",
            "ocr_source": document.pages[0].source if document.pages else "—",
        }
        return result


async def run(dataset_path: Path, out_dir: Path, limit: int | None, quiet: bool) -> dict:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = dataset.get("documents", dataset if isinstance(dataset, list) else [])
    if limit:
        items = items[:limit]
    if not items:
        raise SystemExit(f"Датасет пуст: {dataset_path}")

    await init_db()
    user = await ensure_eval_user()
    samples_dir = dataset_path.parent
    ocr_engine = await resolve_engine()

    per_document: list[dict] = []
    skipped: list[dict] = []
    for index, item in enumerate(items, start=1):
        if item.get("requires_ocr") and ocr_engine == "none":
            skipped.append(
                {
                    "filename": item["filename"],
                    "reason": "требуется OCR-движок (Tesseract или локальная VLM)",
                }
            )
            if not quiet:
                print(f"[{index}/{len(items)}] {item['filename']} — пропущен (нет OCR)")
            continue
        if not quiet:
            print(f"[{index}/{len(items)}] {item['filename']} …", flush=True)
        result = await evaluate_document(item, samples_dir, user)
        per_document.append(result)
        if not quiet:
            print(
                f"    CER {result['cer'] * 100:.2f} % · WER {result['wer'] * 100:.2f} % · "
                f"{result.get('doc_type_predicted') or '—'} · {result.get('duration_sec', 0):.2f} с"
            )

    status = await llm_status()
    environment = {
        "llm_engine": status["mode"],
        "llm_model": status["model"],
        "llm_available": status["available"],
        "embedding_engine": "ollama" if status["available"] else "fallback",
        "ocr_engine": ocr_engine,
        "tesseract_available": available_engine() == "tesseract",
        "dataset": (
            str(dataset_path.relative_to(PROJECT_ROOT))
            if dataset_path.is_relative_to(PROJECT_ROOT)
            else str(dataset_path)
        ),
        "python": sys.version.split()[0],
    }
    report = quality_service.build_report(per_document, environment, skipped=skipped)
    json_path, md_path = quality_service.save_report(report, out_dir)

    if not quiet:
        metrics = report["metrics"]
        print("\n=== Итог ===")
        print(f"Документов: {report['documents']}")
        print(f"CER: {metrics['cer'] * 100:.2f} %   WER: {metrics['wer'] * 100:.2f} %")
        print(
            f"NER F1: {metrics['ner_f1'] * 100:.2f} % "
            f"(P {metrics['ner_precision'] * 100:.1f} % / R {metrics['ner_recall'] * 100:.1f} %)"
        )
        print(f"Точность классификации: {metrics['doc_type_accuracy'] * 100:.2f} %")
        print(f"Среднее время на документ: {metrics['avg_duration_sec']:.2f} с")
        if report.get("skipped"):
            print(f"Пропущено (нет OCR): {len(report['skipped'])}")
        print(f"\nОтчёт: {json_path}\n        {md_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка качества IDP-пайплайна")
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "samples" / "ground_truth.json"),
        help="путь к JSON с эталонной разметкой",
    )
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "reports" / "quality"), help="каталог для отчётов"
    )
    parser.add_argument("--limit", type=int, default=None, help="ограничить число документов")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    report = asyncio.run(
        run(Path(args.dataset).resolve(), Path(args.out).resolve(), args.limit, args.quiet)
    )
    if args.quiet:
        print(json.dumps(report.get("metrics", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
