"""Бенчмарк поиска на большой базе (требование ТЗ §13: поиск < 200 мс на 1000+ документах).

Что делает:
  1. создаёт N документов (по умолчанию 1000) на основе реальных текстов демо-датасета:
     страницы OCR, чанки и эмбеддинги — теми же функциями, что и пайплайн обработки;
  2. замеряет время поиска в режимах fulltext / semantic / hybrid:
       * сервисный уровень — вызов search_documents() с отдельной сессией на запрос;
       * HTTP-уровень (если задан --api-url) — полный путь: запрос → API → ответ;
  3. пишет отчёт в reports/search_benchmark.json (или --out) и удаляет тестовые данные (если не --keep).

Запуск (из каталога backend, база задаётся DATABASE_URL):
  DATABASE_URL=postgresql+asyncpg://... PYTHONPATH=. python -m scripts.search_benchmark --docs 1000
  ... --api-url http://127.0.0.1:8003 --email bench@example.ru --password bench-pass-123

Важно: эмбеддинги берутся fallback-движком (hashing), поэтому замер скорости семантического
поиска соответствует пути без GPU; косинус по нему считается тем же кодом, что и с bge-m3.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import statistics
import sys
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND = PROJECT_ROOT / "backend"
GROUND_TRUTH = PROJECT_ROOT / "samples" / "ground_truth.json"
BENCH_EMAIL = "bench@example.ru"
MODES = ("fulltext", "semantic", "hybrid")
TARGET_MS = 200.0


def _base_documents() -> list[dict]:
    data = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    docs = [d for d in data["documents"] if d.get("text")]
    if not docs:
        raise SystemExit("В samples/ground_truth.json нет текстов для бенчмарка")
    return docs


def _queries(base: list[dict], count: int, rng: random.Random) -> list[str]:
    """Запросы из реальных слов и пар слов датасета: разная длина и частота."""
    words: list[str] = []
    for doc in base:
        words += [w.strip('.,:;()«»"').lower() for w in doc["text"].split()]
    words = [w for w in words if len(w) >= 6 and w.isalpha()]
    queries: set[str] = set()
    while len(queries) < count:
        if rng.random() < 0.5:
            queries.add(rng.choice(words))
        else:
            i = rng.randrange(len(words) - 1)
            queries.add(f"{words[i]} {words[i + 1]}")
    return sorted(queries)


async def _seed(session_factory, docs: int, base: list[dict], rng: random.Random) -> dict:
    from app.core.security import hash_password
    from app.ml_pipeline.text_utils import chunk_text, hashing_embedding
    from app.models.document import Document, DocumentStatus, OcrPage, DocumentChunk
    from app.models.user import User
    from app.core.config import settings
    from app.services.vector_store import is_postgres, save_pg_vectors
    from sqlalchemy import select

    async with session_factory() as session:
        user = (await session.execute(select(User).where(User.email == BENCH_EMAIL))).scalar_one_or_none()
        if user is None:
            user = User(email=BENCH_EMAIL, password_hash=hash_password("bench-pass-123"), full_name="Benchmark", role="admin")
            session.add(user)
            await session.commit()
        user_id = user.id

    now = datetime.now(UTC)
    chunks_total = 0
    batch_size = 100
    postgres = None
    for start in range(0, docs, batch_size):
        async with session_factory() as session:
            if postgres is None:
                postgres = is_postgres(session)
            new_chunks: list[DocumentChunk] = []
            for i in range(start, min(start + batch_size, docs)):
                source = base[i % len(base)]
                # уникальный вариант текста: номер, сумма и дата отличаются у каждого документа
                text = (
                    f"{source['text']}\n\nСправка № {i + 1}-{rng.randint(100, 999)}. "
                    f"Сумма {rng.randint(10_000, 9_999_999)} руб. "
                    f"Дата {rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.2025."
                )
                document = Document(
                    user_id=user_id,
                    filename=f"bench_{i + 1:05d}_{source['filename']}",
                    s3_key_original=f"bench/{uuid.uuid4()}",
                    mime_type="application/pdf",
                    file_size=len(text),
                    page_count=1,
                    status=DocumentStatus.COMPLETED,
                    progress=100,
                    doc_type=source.get("doc_type"),
                    created_at=now - timedelta(days=rng.randint(0, 365)),
                )
                session.add(document)
                await session.flush()  # id нужен для страниц и чанков
                session.add(OcrPage(document_id=document.id, page_number=1, raw_text=text,
                                    corrected_text=text, source="native", avg_confidence=1.0))
                for index, piece in enumerate(chunk_text(text)):
                    new_chunks.append(DocumentChunk(
                        document_id=document.id, page_number=1, chunk_index=index,
                        chunk_text=piece, embedding=hashing_embedding(piece, settings.EMBEDDING_DIM),
                        embedding_model="fallback",
                    ))
            session.add_all(new_chunks)
            await session.flush()
            if postgres:
                await save_pg_vectors(session, [(c.id, c.embedding) for c in new_chunks])
            await session.commit()
            chunks_total += len(new_chunks)
        print(f"    загружено документов: {min(start + batch_size, docs)}/{docs}", flush=True)
    return {"user_id": user_id, "chunks": chunks_total}


async def _cleanup(session_factory) -> int:
    from sqlalchemy import delete, select
    from app.models.document import Document, DocumentChunk, OcrPage
    from app.models.user import User

    async with session_factory() as session:
        user = (await session.execute(select(User).where(User.email == BENCH_EMAIL))).scalar_one_or_none()
        if user is None:
            return 0
        doc_ids = select(Document.id).where(Document.user_id == user.id)
        await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id.in_(doc_ids)))
        await session.execute(delete(OcrPage).where(OcrPage.document_id.in_(doc_ids)))
        result = await session.execute(delete(Document).where(Document.user_id == user.id))
        await session.commit()
        return result.rowcount or 0


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * len(ordered)) - 1))
    return ordered[index]


def _summary(samples_ms: list[float]) -> dict:
    return {
        "queries": len(samples_ms),
        "p50_ms": round(statistics.median(samples_ms), 2),
        "p95_ms": round(_percentile(samples_ms, 95), 2),
        "max_ms": round(max(samples_ms), 2),
        "mean_ms": round(statistics.fmean(samples_ms), 2),
        "passed": _percentile(samples_ms, 95) < TARGET_MS,
    }


async def _measure_service(session_factory, queries: list[str]) -> dict:
    from app.schemas.document import SearchFilters
    from app.services.search_service import search_documents

    results: dict[str, dict] = {}
    for mode in MODES:
        # прогрев: первый запрос включает подключение и компиляцию планов
        async with session_factory() as session:
            await search_documents(session, queries[0], SearchFilters(), mode, 20)
        samples: list[float] = []
        for query in queries:
            async with session_factory() as session:
                started = time.perf_counter()
                await search_documents(session, query, SearchFilters(), mode, 20)
                samples.append((time.perf_counter() - started) * 1000)
        results[mode] = _summary(samples)
        print(f"    сервис/{mode:9s} p50={results[mode]['p50_ms']:>8} мс  p95={results[mode]['p95_ms']:>8} мс", flush=True)
    return results


def _measure_http(api_url: str, email: str, password: str, queries: list[str]) -> dict:
    import httpx

    client = httpx.Client(base_url=api_url.rstrip("/") + "/api/v1", timeout=60)
    token = client.post("/auth/login", json={"email": email, "password": password})
    token.raise_for_status()
    headers = {"Authorization": f"Bearer {token.json()['access_token']}"}
    results: dict[str, dict] = {}
    for mode in MODES:
        client.post("/search", headers=headers, json={"query": queries[0], "mode": mode, "limit": 20})
        samples: list[float] = []
        for query in queries:
            started = time.perf_counter()
            response = client.post("/search", headers=headers, json={"query": query, "mode": mode, "limit": 20})
            samples.append((time.perf_counter() - started) * 1000)
            response.raise_for_status()
        results[mode] = _summary(samples)
        print(f"    HTTP/{mode:9s} p50={results[mode]['p50_ms']:>8} мс  p95={results[mode]['p95_ms']:>8} мс", flush=True)
    return results


async def main_async(args: argparse.Namespace) -> int:
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from app.core.config import settings
    from app.core.database import engine, init_db

    rng = random.Random(args.seed)
    base = _base_documents()
    await init_db()
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    dialect = engine.dialect.name

    print(f"==> База: {dialect} · документов: {args.docs} · запросов на режим: {args.queries}")
    removed = await _cleanup(session_factory)  # чистый старт: прошлые тестовые данные не мешают
    if removed:
        print(f"    удалено прошлых тестовых документов: {removed}")

    started = time.perf_counter()
    seeded = await _seed(session_factory, args.docs, base, rng)
    seed_seconds = round(time.perf_counter() - started, 1)
    print(f"    готово за {seed_seconds} с, чанков: {seeded['chunks']}")

    queries = _queries(base, args.queries, rng)
    print("==> Замер на сервисном уровне")
    service = await _measure_service(session_factory, queries)

    http = None
    if args.api_url:
        print(f"==> Замер через HTTP API {args.api_url}")
        http = _measure_http(args.api_url, args.email, args.password, queries)

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "target_p95_ms": TARGET_MS,
        "requirement": "ТЗ §13: время поиска < 200 мс на базе 1000+ документов",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "database_dialect": _engine_dialect_label(engine),
            "embedding_dim": settings.EMBEDDING_DIM,
            "embedding_engine": "fallback (hashing)",
        },
        "dataset": {
            "documents": args.docs,
            "chunks": seeded["chunks"],
            "seed_seconds": seed_seconds,
            "source": "реальные тексты samples/ground_truth.json, уникальные номера, суммы и даты",
        },
        "queries": queries,
        "service_level": service,
        "http_level": http,
        "passed": all(v["passed"] for v in service.values()) and (
            http is None or all(v["passed"] for v in http.values())
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"==> Отчёт: {out}")
    print(f"==> Итог: {'ПРОЙДЕН' if report['passed'] else 'НЕ ПРОЙДЕН'} (порог p95 < {TARGET_MS:.0f} мс)")

    if not args.keep:
        await _cleanup(session_factory)
        print("    тестовые документы удалены (--keep, чтобы оставить)")
    await engine.dispose()
    return 0 if report["passed"] else 1


def _engine_dialect_label(engine) -> str:
    if engine.dialect.name == "postgresql":
        return "PostgreSQL (tsvector + pgvector HNSW)"
    return "SQLite (LIKE + косинус в Python)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=int, default=1000, help="сколько документов создать (по умолчанию 1000)")
    parser.add_argument("--queries", type=int, default=40, help="запросов на каждый режим")
    parser.add_argument("--seed", type=int, default=7, help="зерно генератора (повторяемость)")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "search_benchmark.json"))
    parser.add_argument("--api-url", default="", help="если задан — дополнительно замерить через HTTP API")
    parser.add_argument("--email", default=BENCH_EMAIL)
    parser.add_argument("--password", default="bench-pass-123")
    parser.add_argument("--keep", action="store_true", help="не удалять тестовые документы после замера")
    args = parser.parse_args()
    sys.path.insert(0, str(BACKEND))
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
