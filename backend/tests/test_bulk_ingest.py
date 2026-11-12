"""Пакетный загрузчик: батчи, отказы по правилам API, возобновление по чекпоинту, backpressure."""
from __future__ import annotations

import uuid

from sqlalchemy import select

from app.core.database import SessionLocal
from app.models.document import Document
from app.models.user import User
from app.services.storage_service import storage
from scripts import bulk_ingest as bi


def _owner_email() -> str:
    return f"bulk-{uuid.uuid4().hex[:8]}@example.ru"


async def _make_owner() -> str:
    email = _owner_email()
    async with SessionLocal() as session:
        session.add(User(id=str(uuid.uuid4()), email=email, password_hash="x", full_name="bulk"))
        await session.commit()
    return email


def _make_source(root, good: int) -> None:
    for i in range(good):
        sub = root / f"part-{i % 3}"
        sub.mkdir(parents=True, exist_ok=True)
        (sub / f"bulk-{i:03}.txt").write_text(f"Договор номер {i}, сумма {i * 100} рублей", encoding="utf-8")
    (root / "bad.exe").write_bytes(b"MZ\x90\x00")  # расширение не поддерживается
    (root / "fake.pdf").write_bytes(b"this is not a pdf")  # Magic Bytes не совпадают


async def _count_bulk_docs() -> int:
    async with SessionLocal() as session:
        rows = (await session.execute(select(Document.id).where(Document.filename.like("bulk-%")))).all()
    return len(rows)


async def test_bulk_ingest_batches_and_rejects(tmp_path, monkeypatch):
    monkeypatch.setattr(bi.settings, "QUEUE_BACKEND", "inline")
    _make_source(tmp_path / "src", good=25)
    email = await _make_owner()

    before = await _count_bulk_docs()
    stats = await bi.run(
        tmp_path / "src",
        owner_email=email,
        batch_size=10,
        batch_bytes=10**9,
        max_queue=100,
        checkpoint_path=tmp_path / "ck.json",
        register_only=True,
        limit=None,
    )
    assert stats.accepted == 25
    assert stats.rejected_count == 2
    assert stats.batches == 3  # 27 файлов по 10 → 3 батча
    assert await _count_bulk_docs() - before == 25

    async with SessionLocal() as session:
        doc = await session.scalar(select(Document).where(Document.filename == "bulk-000.txt"))
    assert doc.s3_key_original.startswith("uploads/")
    assert (storage.root / doc.s3_key_original).read_text(encoding="utf-8").startswith("Договор номер 0")
    assert doc.status == "pending"


async def test_bulk_ingest_resumes_from_checkpoint_without_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(bi.settings, "QUEUE_BACKEND", "inline")
    _make_source(tmp_path / "src", good=12)
    email = await _make_owner()
    ck = tmp_path / "ck.json"
    before = await _count_bulk_docs()

    first = await bi.run(
        tmp_path / "src", owner_email=email, batch_size=4, batch_bytes=10**9, max_queue=100,
        checkpoint_path=ck, register_only=True, limit=6,
    )
    assert first.scanned == 6
    assert ck.exists()

    second = await bi.run(
        tmp_path / "src", owner_email=email, batch_size=4, batch_bytes=10**9, max_queue=100,
        checkpoint_path=ck, register_only=True, limit=None,
    )
    # во втором запуске — только файлы, которых не было в первом; всего 12 хороших, без дублей
    assert first.accepted + second.accepted == 12
    assert await _count_bulk_docs() - before == 12


def test_batches_split_by_count_and_bytes(tmp_path):
    files = []
    for i in range(7):
        path = tmp_path / f"f{i}.bin"
        path.write_bytes(b"x" * (100 if i != 3 else 900))
        files.append(path)
    batches = list(bi.batched(files, max_count=10, max_bytes=500))
    assert sum(len(b) for b in batches) == 7  # ни один файл не потерян
    assert all(sum(p.stat().st_size for p in b) <= 500 or len(b) == 1 for b in batches)
    assert len(list(bi.batched(files, max_count=3, max_bytes=10**9))) == 3


def test_backpressure_waits_until_queue_drains():
    lengths = iter([5000, 5000, 2500, 100])
    sleeps: list[float] = []
    waited = bi.wait_for_queue_capacity(
        lambda: next(lengths), max_length=2000, sleep=sleeps.append, poll_seconds=2.0
    )
    assert waited == 6.0  # 5000, 5000 и 2500 выше порога 2000 → три паузы по 2 с
    assert sleeps == [2.0, 2.0, 2.0]


def test_backpressure_disabled_without_queue():
    assert bi.wait_for_queue_capacity(lambda: None, 10, sleep=lambda s: 1 / 0) == 0.0


def test_backpressure_timeout_raises():
    import pytest

    with pytest.raises(TimeoutError):
        bi.wait_for_queue_capacity(
            lambda: 10**6, 10, sleep=lambda s: None, poll_seconds=1.0, timeout_seconds=3.0
        )


def test_iter_files_is_deterministic(tmp_path):
    for name in ["b/2.txt", "a/1.txt", "a/0.txt", "c.txt"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    first = [p.relative_to(tmp_path).as_posix() for p in bi.iter_files(tmp_path)]
    second = [p.relative_to(tmp_path).as_posix() for p in bi.iter_files(tmp_path)]
    assert first == second
    assert len(first) == 4
