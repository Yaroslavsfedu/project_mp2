"""Масштабный замер: как растут время списка, FTS и векторного поиска при росте базы в 50–100 раз.

Что делает (только PostgreSQL + pgvector — на нём заявлен путь к 2 млн документов):
  1. Генерирует документы, страницы OCR и чанки с векторами ПРЯМО НА СЕРВЕРЕ (generate_series):
     без Python-цикла по строкам, поэтому 100 тыс. документов грузятся за минуты.
     Текст — случайные слова со «скошенным» распределением: слово «договор» входит почти в каждый
     документ (частое слово — худший случай для FTS), «редкийтермин» — примерно в 1 из 5000.
  2. Для каждого размера базы (по умолчанию 2 000 → 20 000 → 100 000 документов) замеряет p50/p95:
       * список: первая страница (offset), глубокая страница (offset=50 000), курсор (keyset) в середине,
         фильтры + курсор, счётчик без фильтров (оценка pg_class.reltuples) и с фильтрами;
       * FTS: частое слово и редкое слово (сервис search_documents и q в списке);
       * семантика (pgvector HNSW) и гибрид.
  3. Снимает размеры таблиц и индексов → байт на документ → пересчёт на 2 млн документов.
  4. Снимает EXPLAIN для основных запросов (видно, что используются индексы, а не Seq Scan).
  5. Пишет reports/scale_benchmark.json.

Запуск (из backend, база ДОЛЖНА быть пустой тестовой, созданной заранее):
  DATABASE_URL=postgresql+asyncpg://postgres@/idp_scale?host=/tmp/pgdata \
  PYTHONPATH=. python -m scripts.scale_benchmark --sizes 2000,20000,100000

Ограничения песочницы (2 vCPU, 3 ГБ RAM) учтены в отчёте: замер до 100 тыс., 2 млн — экстраполяция.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DEFAULT = PROJECT_ROOT / "reports" / "scale_benchmark.json"
SCALE_USER_EMAIL = "scale@example.ru"
TARGET_MS = 100.0  # порог p95 для списка и FTS на каждом размере
DOC_TYPES = ("Договор", "Акт", "Счёт", "Приказ", "Письмо")
VOCAB = (
    "договор поставки оплата услуги акт работ счёт приказ письмо организация сумма рублей НДС "
    "реквизиты исполнитель заказчик срок неустойка поставщик покупатель товар накладная соглашение "
    "расторжение приложение спецификация банк платёж отгрузка склад количество стоимость скидка "
    "гарантия ответственность претензия арбитраж суд решение согласование подпись печать директор "
    "бухгалтерия налог отчёт квартал год январь февраль март апрель май июнь июль август сентябрь "
    "октябрь ноябрь декабрь регион город адрес телефон почта документ архив копия оригинал"
).split()
RARE_TERM = "редкийтермин"
MEDIUM_TERM = "оригинал"  # последнее слово словаря: в ~20% страниц
DIM_DEFAULT = 1024


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, int(round(pct / 100 * (len(ordered) - 1)))))
    return ordered[index]


def _stats(samples_ms: list[float]) -> dict:
    return {
        "n": len(samples_ms),
        "p50_ms": round(_percentile(samples_ms, 50), 2),
        "p95_ms": round(_percentile(samples_ms, 95), 2),
        "mean_ms": round(statistics.fmean(samples_ms), 2),
    }


# ----------------------------- генерация данных -----------------------------
def _vocab_sql() -> str:
    words = ", ".join("'" + w.replace("'", "''") + "'" for w in VOCAB)
    return f"ARRAY[{words}]::text[]"


async def _ensure_user(db) -> str:
    row = await db.execute(text("SELECT id FROM users WHERE email = :e"), {"e": SCALE_USER_EMAIL})
    found = row.scalar()
    if found:
        return found
    user_id = str(uuid.uuid4())
    await db.execute(
        text("INSERT INTO users (id, email, password_hash, full_name, role, is_active, created_at) "
             "VALUES (:id, :e, 'x', 'Scale bench', 'operator', true, now())"),
        {"id": user_id, "e": SCALE_USER_EMAIL},
    )
    await db.commit()
    return user_id


async def _seed_range(db, user_id: str, lo: int, hi: int, dim: int) -> None:
    """Добавляет документы с номерами g в [lo, hi). id детерминированы: md5('doc'||g)."""
    await db.execute(
        text("""
            INSERT INTO documents (id, user_id, filename, s3_key_original, mime_type, file_size,
                                   page_count, status, progress, doc_type, created_at, updated_at)
            SELECT md5('doc' || g)::uuid::text,
                   :uid,
                   'doc-' || g || '.txt',
                   'uploads/' || substr(md5('doc' || g), 1, 2) || '/' || substr(md5('doc' || g), 3, 2) || '/'
                       || md5('doc' || g)::uuid::text || '_doc-' || g || '.txt',
                   'text/plain', 1000 + (g % 5000), 1,
                   CASE WHEN g % 20 = 0 THEN 'failed' WHEN g % 20 = 1 THEN 'verified' ELSE 'completed' END,
                   100,
                   (ARRAY['Договор','Акт','Счёт','Приказ','Письмо'])[1 + g % 5],
                   now() - (random() * interval '700 days'),
                   now()
            FROM generate_series(:lo, :hi - 1) AS g
        """),
        {"uid": user_id, "lo": lo, "hi": hi},
    )
    await db.execute(
        text(f"""
            WITH v AS (SELECT {_vocab_sql()} AS w)
            INSERT INTO ocr_pages (id, document_id, page_number, raw_text, corrected_text, bounding_boxes,
                                   image_key, image_width, image_height, avg_confidence, source)
            SELECT md5('page' || g)::uuid::text,
                   md5('doc' || g)::uuid::text,
                   1,
                   body.text || CASE WHEN g % 5000 = 0 THEN ' {RARE_TERM}' ELSE '' END,
                   NULL, '[]', NULL, 0, 0, 0.92, 'ocr'
            FROM generate_series(:lo, :hi - 1) AS g
            CROSS JOIN v
            CROSS JOIN LATERAL (
                SELECT string_agg(word, ' ') AS text
                FROM (
                    SELECT v.w[1 + floor(cardinality(v.w) * power(random(), 2.5))::int] AS word
                    FROM generate_series(1, 40) AS n
                ) AS picked
            ) AS body
        """),
        {"lo": lo, "hi": hi},
    )
    await db.execute(
        text("""
            INSERT INTO document_chunks (id, document_id, page_number, chunk_index, chunk_text, embedding,
                                         embedding_model, embedding_vec)
            SELECT md5('chunk' || g)::uuid::text,
                   md5('doc' || g)::uuid::text,
                   1, 0,
                   left(p.raw_text, 500),
                   '[]',
                   'fallback',
                   (SELECT ('[' || string_agg(round((random() - 0.5)::numeric, 4)::text, ',') || ']')::vector
                      FROM generate_series(1, :dim))
            FROM generate_series(:lo, :hi - 1) AS g
            JOIN ocr_pages p ON p.id = md5('page' || g)::uuid::text
        """),
        {"lo": lo, "hi": hi, "dim": dim},
    )
    await db.commit()


async def _drop_hnsw(db) -> None:
    await db.execute(text("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw"))
    await db.commit()


async def _build_hnsw(db) -> float:
    started = time.perf_counter()
    await db.execute(text("SET maintenance_work_mem = '512MB'"))
    await db.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw ON document_chunks "
        "USING hnsw (embedding_vec vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    ))
    await db.commit()
    return round(time.perf_counter() - started, 2)


async def _analyze(db) -> None:
    for table in ("documents", "ocr_pages", "document_chunks"):
        await db.execute(text(f"ANALYZE {table}"))
    await db.commit()


async def _sizes(db) -> dict:
    row = await db.execute(text("""
        SELECT
          pg_total_relation_size('documents')        AS documents_total,
          pg_relation_size('documents')              AS documents_heap,
          pg_total_relation_size('ocr_pages')        AS ocr_pages_total,
          pg_relation_size('ocr_pages')              AS ocr_pages_heap,
          pg_total_relation_size('document_chunks')  AS chunks_total,
          pg_relation_size('document_chunks')        AS chunks_heap,
          coalesce((SELECT pg_relation_size(c.oid) FROM pg_class c
                    WHERE c.relname = 'ix_chunks_embedding_hnsw'), 0) AS hnsw_index,
          coalesce((SELECT sum(pg_relation_size(c.oid)) FROM pg_class c
                    JOIN pg_index i ON i.indexrelid = c.oid
                    WHERE i.indrelid = 'documents'::regclass), 0) AS documents_indexes,
          coalesce((SELECT sum(pg_relation_size(c.oid)) FROM pg_class c
                    JOIN pg_index i ON i.indexrelid = c.oid
                    WHERE i.indrelid = 'ocr_pages'::regclass), 0) AS ocr_pages_indexes
    """))
    return {key: int(value) for key, value in row.mappings().one().items()}


async def _counts(db) -> dict:
    docs = int((await db.execute(text("SELECT count(*) FROM documents"))).scalar())
    chunks = int((await db.execute(text("SELECT count(*) FROM document_chunks"))).scalar())
    return {"documents": docs, "chunks": chunks}


# ----------------------------- замеры -----------------------------
async def _timed(fn, reps: int, warmup: int = 2) -> dict:
    from app.core.database import SessionLocal

    for _ in range(warmup):
        async with SessionLocal() as db:
            await fn(db)
    samples: list[float] = []
    for _ in range(reps):
        async with SessionLocal() as db:
            started = time.perf_counter()
            await fn(db)
            samples.append((time.perf_counter() - started) * 1000)
    return _stats(samples)


async def _measure_step(size: int, reps: int, dim: int) -> dict:
    from app.core.database import SessionLocal
    from app.services import document_list_service as dls
    from app.services.document_list_service import ListQuery, list_documents_page
    from app.schemas.document import SearchFilters
    from app.services.search_service import search_documents

    ops: dict[str, dict] = {}

    async with SessionLocal() as db:
        middle = (await db.execute(text(
            "SELECT created_at, id FROM documents ORDER BY created_at DESC, id DESC OFFSET :off LIMIT 1"
        ), {"off": size // 2})).one()
    cursor = (middle[0], middle[1])

    async def first_page(db):
        await list_documents_page(db, ListQuery(), page=1, limit=20)

    async def deep_offset(db):
        # OFFSET 50 000: сколько стоит «глубокая» страница классической пагинацией
        await list_documents_page(db, ListQuery(), page=50_001 // 20 + 1, limit=20)

    async def keyset_middle(db):
        await list_documents_page(db, ListQuery(), page=1, limit=20, cursor=cursor)

    async def keyset_filtered(db):
        await list_documents_page(
            db, ListQuery(status="completed", doc_type="Договор"), page=1, limit=20, cursor=cursor
        )

    async def count_no_filters(db):
        dls.clear_count_cache()
        await dls.count_documents(db, ListQuery(), [])

    async def count_filtered_exact(db):
        dls.clear_count_cache()
        await dls.count_documents(db, ListQuery(status="completed"), [dls.Document.status == "completed"])

    async def list_fts_common(db):
        dls.clear_count_cache()
        await list_documents_page(db, ListQuery(query="договор"), page=1, limit=20)

    async def fts_common(db):
        await search_documents(db, "договор", SearchFilters(), "fulltext", 20)

    async def fts_rare(db):
        await search_documents(db, RARE_TERM, SearchFilters(), "fulltext", 20)

    async def fts_medium(db):
        await search_documents(db, MEDIUM_TERM, SearchFilters(), "fulltext", 20)

    async def semantic(db):
        await search_documents(db, "договор поставки оплата", SearchFilters(), "semantic", 20)

    async def hybrid(db):
        await search_documents(db, "договор поставки оплата", SearchFilters(), "hybrid", 20)

    ops["list_first_page_offset"] = await _timed(first_page, reps)
    ops["list_deep_offset_50k"] = await _timed(deep_offset, reps)
    ops["list_keyset_middle"] = await _timed(keyset_middle, reps)
    ops["list_keyset_filtered"] = await _timed(keyset_filtered, reps)
    ops["count_no_filters_estimate"] = await _timed(count_no_filters, max(5, reps // 3))
    ops["count_filtered_exact"] = await _timed(count_filtered_exact, max(5, reps // 3))
    ops["list_q_common_word"] = await _timed(list_fts_common, max(5, reps // 3))
    ops["fts_common_word"] = await _timed(fts_common, reps)
    ops["fts_rare_word"] = await _timed(fts_rare, reps)
    ops["fts_medium_word"] = await _timed(fts_medium, reps)
    ops["semantic_hnsw"] = await _timed(semantic, reps)
    ops["hybrid"] = await _timed(hybrid, reps)
    return ops


async def _explain(db, size: int) -> dict:
    """План основных запросов на текущем размере (EXPLAIN ANALYZE). Возвращаем строки с узлами плана."""
    plans: dict[str, list[str]] = {}
    middle = (await db.execute(text(
        "SELECT created_at, id FROM documents ORDER BY created_at DESC, id DESC OFFSET :off LIMIT 1"
    ), {"off": size // 2})).one()
    queries = {
        "fts_common_word_default_planner": (
            "SELECT p.id FROM ocr_pages p JOIN documents d ON d.id = p.document_id "
            "WHERE p.search_tsv @@ plainto_tsquery('russian', 'договор') LIMIT 2000",
            {},
        ),
        "fts_common_word_gin_only": (
            "SET LOCAL enable_seqscan = off; SELECT p.id FROM ocr_pages p JOIN documents d ON d.id = p.document_id "
            "WHERE p.search_tsv @@ plainto_tsquery('russian', 'договор') LIMIT 2000",
            {},
        ),
        "fts_rare_word_gin_only": (
            "SET LOCAL enable_seqscan = off; SELECT p.id FROM ocr_pages p JOIN documents d ON d.id = p.document_id "
            "WHERE p.search_tsv @@ plainto_tsquery('russian', 'редкийтермин') LIMIT 2000",
            {},
        ),
        "keyset_no_filter": (
            "SELECT id FROM documents WHERE (created_at, id) < (:c, :i) "
            "ORDER BY created_at DESC, id DESC LIMIT 20",
            {"c": middle[0], "i": middle[1]},
        ),
        "keyset_status_doctype": (
            "SELECT id FROM documents WHERE status = 'completed' AND doc_type = 'Договор' "
            "AND (created_at, id) < (:c, :i) ORDER BY created_at DESC, id DESC LIMIT 20",
            {"c": middle[0], "i": middle[1]},
        ),
        "fts_candidates_common_word": (
            "SELECT p.id FROM ocr_pages p JOIN documents d ON d.id = p.document_id "
            "WHERE p.search_tsv @@ plainto_tsquery('russian', 'договор') LIMIT 2000",
            {},
        ),
        "hnsw_knn": (
            "SELECT c.id FROM document_chunks c WHERE c.embedding_model = 'fallback' "
            "ORDER BY c.embedding_vec <=> (SELECT embedding_vec FROM document_chunks LIMIT 1) LIMIT 20",
            {},
        ),
    }
    for name, (sql, params) in queries.items():
        await db.execute(text("SET LOCAL hnsw.ef_search = 100"))
        # «SET LOCAL ...; SELECT ...» — SET выполняем отдельно, EXPLAIN уже на SELECT
        if sql.startswith("SET LOCAL"):
            setting, sql = sql.split(";", 1)
            await db.execute(text(setting))
            sql = sql.strip()
        rows = (await db.execute(text(f"EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF) {sql}"), params)).all()
        lines = [row[0] for row in rows]
        plans[name] = [line.strip() for line in lines if any(
            key in line for key in ("Scan", "Index", "Bitmap", "Limit", "Sort", "Seq", "Execution Time", "Nested")
        )][:8]
    await db.rollback()
    return plans


def _storage_fanout(total: int = 2_000_000, leaves: int = 65_536) -> dict:
    """Распределение файлов по листам шарда для 2 млн UUID v4 (без записи на диск)."""
    from collections import Counter

    import uuid as uuid_mod

    counter: Counter[str] = Counter()
    for _ in range(total):
        hex_id = uuid_mod.uuid4().hex
        counter[hex_id[:4]] += 1
    counts = list(counter.values())
    return {
        "documents": total,
        "leaf_directories": len(counter),
        "max_per_leaf": max(counts),
        "mean_per_leaf": round(total / leaves, 1),
        "flat_layout_entries_in_one_dir": total,
    }


def _scaling(steps: list[dict]) -> dict:
    first, last = steps[0], steps[-1]
    growth = last["documents"] / max(first["documents"], 1)
    out: dict[str, dict] = {}
    for op in first["ops"]:
        p_first = first["ops"][op]["p95_ms"]
        p_last = last["ops"][op]["p95_ms"]
        ratio = p_last / p_first if p_first else float("nan")
        out[op] = {
            "p95_first_ms": p_first,
            "p95_last_ms": p_last,
            "ratio_last_over_first": round(ratio, 2),
            "max_p95_across_sizes_ms": max(step["ops"][op]["p95_ms"] for step in steps),
        }
    return {"growth_factor_docs": round(growth, 1), "per_operation": out}


async def run(sizes: list[int], reps: int, dim: int, out_path: Path, cleanup: bool) -> dict:
    from app.core.database import SessionLocal, engine, init_db

    if engine.dialect.name != "postgresql":
        raise SystemExit("scale_benchmark работает только на PostgreSQL (DATABASE_URL=postgresql+asyncpg://...)")
    await init_db()

    async with SessionLocal() as db:
        version = (await db.execute(text("SHOW server_version"))).scalar()
        pgv = (await db.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'"))).scalar()
        user_id = await _ensure_user(db)
        existing = await _counts(db)
        if existing["documents"] and existing["documents"] >= sizes[0]:
            print(f"В базе уже {existing['documents']} документов — замер идёт с текущего размера", flush=True)

    steps: list[dict] = []
    current = (await _counts_safe())["documents"]
    for target in sizes:
        load_seconds = 0.0
        if current < target:
            print(f"==> Загрузка до {target} документов (сейчас {current})", flush=True)
            started = time.perf_counter()
            async with SessionLocal() as db:
                # HNSW удаляем на время загрузки и строим заново: вставка в граф по одной строке медленнее
                await _drop_hnsw(db)
                step_lo = current
                chunk = 20_000
                while step_lo < target:
                    step_hi = min(target, step_lo + chunk)
                    await _seed_range(db, user_id, step_lo, step_hi, dim)
                    step_lo = step_hi
                    print(f"    загружено {step_hi}", flush=True)
                current = target
            load_seconds = round(time.perf_counter() - started, 1)
        async with SessionLocal() as db:
            index_seconds = await _build_hnsw(db)
            await _analyze(db)
            counts = await _counts(db)
            sizes_bytes = await _sizes(db)
        print(f"==> Замер при {counts['documents']} документах", flush=True)
        ops = await _measure_step(counts["documents"], reps, dim)
        async with SessionLocal() as db:
            plans = await _explain(db, counts["documents"]) if target == sizes[-1] else None
        docs_total = counts["documents"]
        steps.append({
            "documents": docs_total,
            "chunks": counts["chunks"],
            "load_seconds": load_seconds,
            "index_build_seconds_hnsw": index_seconds,
            "sizes_bytes": sizes_bytes,
            "bytes_per_document": {
                "documents_total": round(sizes_bytes["documents_total"] / docs_total),
                "ocr_pages_total": round(sizes_bytes["ocr_pages_total"] / docs_total),
                "chunks_total_incl_vectors": round(sizes_bytes["chunks_total"] / docs_total),
                "hnsw_index": round(sizes_bytes["hnsw_index"] / docs_total),
            },
            "ops": ops,
            "explain": plans,
        })
        for name, stats in ops.items():
            print(f"    {name:28s} p50={stats['p50_ms']:8.2f} мс  p95={stats['p95_ms']:8.2f} мс", flush=True)

    # --- сводка ---
    final = steps[-1]
    scaling = _scaling(steps)
    passed_ops = {
        op: stats["p95_ms"] < TARGET_MS for op, stats in final["ops"].items()
    }
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "postgres": version,
            "pgvector": pgv,
            "embedding_dim": dim,
            "note": "песочница: 2 vCPU, 3 ГБ RAM; эмбеддинги случайные (для замера скорости индекса)",
        },
        "config": {
            "sizes": sizes,
            "reps": reps,
            "target_p95_ms": TARGET_MS,
            "fts_candidate_limit": _settings_value("FTS_CANDIDATE_LIMIT"),
            "hnsw_ef_search": _settings_value("HNSW_EF_SEARCH"),
            "list_offset_max": _settings_value("LIST_OFFSET_MAX"),
            "count_estimate_threshold": _settings_value("COUNT_ESTIMATE_THRESHOLD"),
        },
        "steps": steps,
        "scaling": scaling,
        "storage_fanout_2m": _storage_fanout(),
        "passed_at_largest_size": passed_ops,
        "passed": all(passed_ops.values()),
        "limitations": [
            "замер до максимального размера из --sizes; 2 млн документов — экстраполяция по байтам на документ",
            "GIN для частого слова: стоимость растёт с числом совпадений, поэтому FTS ограничен FTS_CANDIDATE_LIMIT",
            "семантика: случайные векторы — замеряется скорость HNSW, а не качество выдачи",
            "текст синтетический: «договор» есть почти во всех страницах (худший случай для FTS), «редкийтермин» — в 20 из 100000 страниц, «оригинал» — в 20%",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"==> Отчёт: {out_path}", flush=True)
    print(f"==> Итог: {'ПРОЙДЕН' if report['passed'] else 'НЕ ПРОЙДЕН'} (p95 < {TARGET_MS:.0f} мс на {final['documents']} док.)", flush=True)

    if cleanup:
        async with SessionLocal() as db:
            await db.execute(text("TRUNCATE document_chunks, ocr_pages, documents CASCADE"))
            await db.commit()
        print("    тестовые данные удалены (--cleanup)", flush=True)
    return report


async def _counts_safe() -> dict:
    from app.core.database import SessionLocal

    async with SessionLocal() as db:
        return await _counts(db)


def _settings_value(name: str):
    from app.core.config import settings

    return getattr(settings, name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Масштабный замер списка, FTS и векторов (PostgreSQL)")
    parser.add_argument("--sizes", default="2000,20000,100000", help="Размеры базы через запятую, по возрастанию")
    parser.add_argument("--reps", type=int, default=30, help="Повторов на операцию")
    parser.add_argument("--dim", type=int, default=DIM_DEFAULT, help="Размерность векторов (EMBEDDING_DIM)")
    parser.add_argument("--out", type=Path, default=REPORT_DEFAULT)
    parser.add_argument("--cleanup", action="store_true", help="Очистить таблицы после замера")
    args = parser.parse_args(argv)
    sizes = sorted({int(x) for x in args.sizes.split(",") if x.strip()})
    if not sizes or sizes[0] < 100:
        print("Укажите размеры не меньше 100", file=sys.stderr)
        return 2
    asyncio.run(run(sizes, args.reps, args.dim, args.out, args.cleanup))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
