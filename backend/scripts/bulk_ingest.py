"""Потоковая пакетная загрузка документов (bulk ingestion) для больших объёмов.

Зачем: через браузерный drag-and-drop миллионы файлов не загрузить. Скрипт читает папку
(или NFS/смонтированный архив) потоком, режет на батчи и:

* проверяет каждый файл тем же кодом, что и API (расширение, размер, Magic Bytes);
* пишет оригиналы в шард-хранилище ``uploads/ab/cd/{id}_{имя}``;
* вставляет строки ``documents`` батчами: на PostgreSQL — через COPY (asyncpg), иначе — executemany;
* ставит документы в очередь Celery, но только пока очередь не превышает ``--max-queue``
  (backpressure): иначе миллионы задач переполнят Redis и вызовут OOM;
* пишет чекпоинт после каждого зафиксированного батча: прерванный запуск продолжается с места остановки.

Пример (из каталога backend, PostgreSQL и Redis подняты, воркер запущен с QUEUE_BACKEND=celery):

    make bulk-ingest SRC=/data/archive OWNER=admin@example.ru
    python -m scripts.bulk_ingest /data/archive --batch 500 --max-queue 2000

Регистрация без постановки в очередь (обработает воркер позже):

    python -m scripts.bulk_ingest /data/archive --register-only
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import insert, select

from app.core.config import settings
from app.core.database import SessionLocal, init_db, engine
from app.models.audit import AuditLog
from app.models.document import Document, DocumentStatus
from app.models.user import User, UserRole, new_uuid, utcnow
from app.services.storage_service import original_key, safe_filename, storage

logger = logging.getLogger("bulk_ingest")

DOCUMENT_COLUMNS = (
    "id", "user_id", "filename", "s3_key_original", "s3_key_processed", "mime_type",
    "file_size", "page_count", "status", "progress", "doc_type", "error_message",
    "created_at", "updated_at",
)


# ----------------------------- обход папки -----------------------------
def iter_files(source: Path) -> Iterator[Path]:
    """Потоковый обход: порядок детерминирован (сортировка внутри каталога), память — O(ширина каталога).

    Детерминированность нужна чекпоинту: при перезапуске пропускаем уже обработанные N файлов.
    """
    stack = [source]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(os.scandir(current), key=lambda e: e.name)
        except OSError as exc:
            logger.warning("Каталог недоступен, пропущен: %s (%s)", current, exc)
            continue
        subdirs: list[Path] = []
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                subdirs.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                yield Path(entry.path)
        stack.extend(reversed(subdirs))


def batched(items: Iterable[Path], max_count: int, max_bytes: int) -> Iterator[list[Path]]:
    """Режет поток на батчи по количеству файлов и по суммарному объёму.

    Объём считается по stat() до чтения содержимого, поэтому память ограничена ``max_bytes``
    плюс один крупный файл (его размер проверяется API-правилами: не больше MAX_UPLOAD_MB).
    """
    batch: list[Path] = []
    batch_bytes = 0
    for item in items:
        try:
            size = item.stat().st_size
        except OSError:
            size = 0
        if batch and (len(batch) >= max_count or batch_bytes + size > max_bytes):
            yield batch
            batch, batch_bytes = [], 0
        batch.append(item)
        batch_bytes += size
    if batch:
        yield batch


# ----------------------------- backpressure -----------------------------
def wait_for_queue_capacity(
    queue_length: Callable[[], int | None],
    max_length: int,
    *,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = 2.0,
    timeout_seconds: float = 6 * 3600,
) -> float:
    """Ждёт, пока длина очереди не опустится ниже ``max_length``. Возвращает время ожидания, сек.

    ``queue_length`` возвращает None, если очередь недоступна (например, inline-режим): тогда не ждём.
    """
    waited = 0.0
    while True:
        length = queue_length()
        if length is None or length < max_length:
            return waited
        if waited >= timeout_seconds:
            raise TimeoutError(f"Очередь не разгрузилась за {int(timeout_seconds)} с (длина {length})")
        logger.info("Backpressure: в очереди %s задач (порог %s), пауза %.0f с", length, max_length, poll_seconds)
        sleep(poll_seconds)
        waited += poll_seconds


def redis_queue_length() -> Callable[[], int | None]:
    """Длина очереди Celery в Redis. Если Redis недоступен — None (backpressure отключён)."""
    try:
        import redis  # noqa: PLC0415

        client = redis.Redis.from_url(settings.REDIS_URL, socket_timeout=5)
    except Exception:  # noqa: BLE001
        return lambda: None

    def _length() -> int | None:
        try:
            return int(client.llen(settings.CELERY_QUEUE_NAME))
        except Exception:  # noqa: BLE001 - Redis недоступен: не блокируем загрузку
            return None

    return _length


# ----------------------------- чекпоинт -----------------------------
@dataclass
class Checkpoint:
    path: Path | None
    consumed: int = 0  # сколько файлов из источника уже зафиксировано в БД

    @classmethod
    def load(cls, path: Path | None) -> "Checkpoint":
        if path is None or not path.exists():
            return cls(path=path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(path=path, consumed=int(data.get("consumed", 0)))

    def save(self) -> None:
        if self.path is None:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"consumed": self.consumed, "saved_at": utcnow().isoformat()}), encoding="utf-8")
        tmp.replace(self.path)  # атомарная замена: сбой не портит чекпоинт


# ----------------------------- статистика -----------------------------
@dataclass
class IngestStats:
    scanned: int = 0
    accepted: int = 0
    rejected: list[str] = field(default_factory=list)  # до 50 примеров причин
    rejected_count: int = 0
    batches: int = 0
    bytes_written: int = 0
    enqueued: int = 0
    waited_seconds: float = 0.0
    started: float = field(default_factory=time.perf_counter)

    def reject(self, path: Path, reason: str) -> None:
        self.rejected_count += 1
        if len(self.rejected) < 50:
            self.rejected.append(f"{path}: {reason}")


# ----------------------------- вставка -----------------------------
def _csv_value(value) -> str:
    if value is None:
        return ""  # в CSV-формате PostgreSQL пустое поле без кавычек = NULL
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


async def insert_documents(rows: list[dict]) -> None:
    """Вставка батча строк documents: COPY на PostgreSQL, executemany — на SQLite."""
    if not rows:
        return
    async with engine.connect() as conn:
        if engine.dialect.name == "postgresql":
            buffer = io.StringIO()
            writer = csv.writer(buffer, lineterminator="\n")
            for row in rows:
                writer.writerow([_csv_value(row[column]) for column in DOCUMENT_COLUMNS])
            payload = buffer.getvalue().encode("utf-8")
            raw = await conn.get_raw_connection()
            # COPY на порядки быстрее построчных INSERT: одна команда на весь батч
            await raw.driver_connection.copy_to_table(
                "documents",
                source=io.BytesIO(payload),
                columns=list(DOCUMENT_COLUMNS),
                format="csv",
            )
            await conn.commit()
        else:
            await conn.execute(insert(Document.__table__), rows)
            await conn.commit()


# ----------------------------- основной цикл -----------------------------
def _validate(path: Path) -> tuple[str, bytes, str] | str:
    """Возвращает (filename, data, suffix) либо причину отказа. Правила — как у API загрузки."""
    from app.api.v1.documents import ALLOWED_EXTENSIONS, validate_upload  # noqa: PLC0415

    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return f"расширение «{suffix or '—'}» не поддерживается"
    if path.stat().st_size > settings.max_upload_bytes:
        return f"размер больше {settings.MAX_UPLOAD_MB} МБ"
    data = path.read_bytes()
    try:
        validate_upload(path.name, data)
    except HTTPException as exc:
        return str(exc.detail)
    return path.name, data, suffix


async def ingest_batch(
    paths: list[Path],
    owner: User,
    stats: IngestStats,
    *,
    concurrency: int = 8,
) -> list[str]:
    """Принимает батч файлов: пишет оригиналы в хранилище, вставляет строки. Возвращает id документов."""
    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(concurrency)

    async def prepare(path: Path):
        async with semaphore:
            return path, await loop.run_in_executor(None, _validate, path)

    prepared = await asyncio.gather(*(prepare(p) for p in paths))

    now = utcnow()
    rows: list[dict] = []
    ids: list[str] = []

    async def store(row: dict, data: bytes) -> None:
        async with semaphore:
            await storage.put(row["s3_key_original"], data)

    writes = []
    for path, result in prepared:
        if isinstance(result, str):
            stats.reject(path, result)
            continue
        filename, data, suffix = result
        document_id = str(uuid.uuid4())
        stored_name = safe_filename(Path(filename).stem) + suffix
        row = {
            "id": document_id,
            "user_id": owner.id,
            "filename": stored_name,
            "s3_key_original": original_key(document_id, stored_name),
            "s3_key_processed": None,
            "mime_type": _mime(suffix),
            "file_size": len(data),
            "page_count": 0,
            "status": DocumentStatus.PENDING,
            "progress": 0,
            "doc_type": None,
            "error_message": None,
            "created_at": now,
            "updated_at": now,
        }
        rows.append(row)
        ids.append(document_id)
        stats.bytes_written += len(data)
        writes.append(store(row, data))
    await asyncio.gather(*writes)

    await insert_documents(rows)
    async with SessionLocal() as session:
        # один агрегированный след в аудите на батч: 2 млн строк аудита не нужны для расследования
        session.add(
            AuditLog(
                id=new_uuid(),
                user_id=owner.id,
                user_email=owner.email,
                action="bulk_ingest_batch",
                document_name="",
                details={"accepted": len(ids), "rejected": len(paths) - len(ids)},
            )
        )
        await session.commit()
    stats.accepted += len(ids)
    return ids


def _mime(suffix: str) -> str:
    import mimetypes  # noqa: PLC0415

    return mimetypes.guess_type(f"x{suffix}")[0] or "application/octet-stream"


async def resolve_owner(email: str | None) -> User:
    async with SessionLocal() as session:
        if email:
            user = await session.scalar(select(User).where(User.email == email))
        else:
            user = await session.scalar(
                select(User).where(User.role == UserRole.ADMIN).order_by(User.created_at).limit(1)
            )
        if user is None:
            raise SystemExit(
                "Владелец не найден. Укажите --owner существующего пользователя "
                "или сначала зарегистрируйте администратора в приложении."
            )
        return user


def enqueue_documents(ids: list[str]) -> None:
    from app.services import pipeline_service  # noqa: PLC0415

    for document_id in ids:
        pipeline_service.enqueue(document_id)


async def run(
    source: Path,
    *,
    owner_email: str | None,
    batch_size: int,
    batch_bytes: int,
    max_queue: int,
    checkpoint_path: Path | None,
    register_only: bool,
    limit: int | None,
    progress_every: int = 10_000,
    queue_length: Callable[[], int | None] | None = None,
) -> IngestStats:
    await init_db()
    owner = await resolve_owner(owner_email)
    checkpoint = Checkpoint.load(checkpoint_path)
    stats = IngestStats()
    queue_length = queue_length or (redis_queue_length() if not register_only else (lambda: None))
    enqueue = not register_only and settings.QUEUE_BACKEND == "celery"
    if not register_only and not enqueue:
        logger.warning(
            "QUEUE_BACKEND=%s: документы зарегистрированы, но обрабатывать их будет только Celery-воркер "
            "(нужен QUEUE_BACKEND=celery). Используйте --register-only, чтобы не видеть это предупреждение.",
            settings.QUEUE_BACKEND,
        )

    files = iter_files(source)
    # пропуск уже зафиксированных файлов (возобновление после остановки)
    for _ in range(checkpoint.consumed):
        if next(files, None) is None:
            break
    stats.scanned = checkpoint.consumed
    if checkpoint.consumed:
        logger.info("Возобновление: пропущено %s уже обработанных файлов", checkpoint.consumed)

    def limited() -> Iterator[Path]:
        for number, path in enumerate(files, start=1):
            if limit is not None and number > limit:
                return
            yield path

    for batch in batched(limited(), batch_size, batch_bytes):
        if enqueue:
            stats.waited_seconds += wait_for_queue_capacity(queue_length, max_queue)
        ids = await ingest_batch(batch, owner, stats)
        if enqueue:
            enqueue_documents(ids)
            stats.enqueued += len(ids)
        stats.batches += 1
        stats.scanned += len(batch)
        checkpoint.consumed = stats.scanned
        checkpoint.save()
        if stats.scanned // progress_every != (stats.scanned - len(batch)) // progress_every:
            _report(stats)
    _report(stats, final=True)
    return stats


def _report(stats: IngestStats, *, final: bool = False) -> None:
    elapsed = max(time.perf_counter() - stats.started, 1e-9)
    rate = stats.scanned / elapsed * 60
    prefix = "ИТОГ" if final else "ход"
    print(
        f"[{prefix}] просмотрено {stats.scanned}, принято {stats.accepted}, отклонено {stats.rejected_count}, "
        f"батчей {stats.batches}, в очередь {stats.enqueued}, ожидание {stats.waited_seconds:.0f} с, "
        f"{rate:.0f} файлов/мин",
        flush=True,
    )
    if final and stats.rejected:
        print("Примеры отказов:", flush=True)
        for line in stats.rejected[:10]:
            print(f"  - {line}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Пакетная потоковая загрузка документов")
    parser.add_argument("source", type=Path, help="Папка с документами (обход рекурсивный)")
    parser.add_argument("--owner", default=None, help="E-mail владельца (по умолчанию первый администратор)")
    parser.add_argument("--batch", type=int, default=500, help="Файлов в батче (по умолчанию 500)")
    parser.add_argument("--batch-mb", type=int, default=256, help="Лимит объёма батча, МБ")
    parser.add_argument("--max-queue", type=int, default=2000, help="Пауза, если в очереди больше задач")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Файл чекпоинта для возобновления")
    parser.add_argument("--register-only", action="store_true", help="Не ставить в очередь обработки")
    parser.add_argument("--limit", type=int, default=None, help="Обработать не более N файлов (для проверки)")
    args = parser.parse_args(argv)

    if not args.source.is_dir():
        print(f"Не папка: {args.source}", file=sys.stderr)
        return 2
    checkpoint = args.checkpoint or (args.source / ".bulk_ingest_checkpoint.json")
    stats = asyncio.run(
        run(
            args.source,
            owner_email=args.owner,
            batch_size=max(1, args.batch),
            batch_bytes=max(1, args.batch_mb) * 1024 * 1024,
            max_queue=max(1, args.max_queue),
            checkpoint_path=checkpoint,
            register_only=args.register_only,
            limit=args.limit,
        )
    )
    return 0 if stats.accepted or stats.scanned else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    raise SystemExit(main())
