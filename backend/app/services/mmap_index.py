"""Дисковый векторный индекс на memory-mapped файлах для слабых машин (Ultra-Low-Spec).

Вместо кэша float32 в куче Python (4 КБ на чанк) хранятся три плоских файла на чанк:

* ``signs.bin`` — 1-битная сигнатура SimHash, 16×uint64 = 128 Б (POPCNT-скан Хэмминга). Биты — знаки
  случайной проекции (``proj.f32``, dim×dim, фиксированный seed): расстояние Хэмминга оценивает угол;
* ``int8.bin``  — единичный вектор int8, 1024 Б (доранжирование кандидатов);
* ``ids.bin``   — UUID чанка, 36 Б (связь с таблицей document_chunks).

Файлы читаются через ``numpy.memmap``: C-ядро получает прямой указатель на отображение
(zero-copy), поэтому Python не копирует векторы. Расход RAM — страничный кэш ОС, который
ядро может вытеснить; собственная куча процесса не растёт с размером базы.

Файлы только дописываются; удалённые чанки остаются в индексе и отфильтровываются при выборке
по таблице (см. vector_store.top_chunks_mmap). Полная пересборка — ``rebuild()``.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.native.loader import get_kernels

META_FILE = "meta.json"
PROJ_FILE = "proj.f32"
PROJ_SEED = 20261112  # фиксированный seed: построение и запрос обязаны использовать одну матрицу
SIGNS_FILE = "signs.bin"
INT8_FILE = "int8.bin"
IDS_FILE = "ids.bin"
ID_BYTES = 36
FORMAT_VERSION = 2  # v2: SimHash-сигнатуры (v1 — знаки координат, несовместимы)


@dataclass
class Candidate:
    chunk_id: str
    score: float  # приближение косинуса по int8 (1.0 = совпадение направления)


class MmapVectorIndex:
    def __init__(self, directory: Path, dim: int) -> None:
        self.dir = Path(directory)
        self.dim = dim
        self.words = (dim + 63) // 64
        self._lock = threading.Lock()
        self.kernels = get_kernels()
        self.count = 0
        self.cursor = 0  # максимальный rowid, уже попавший в индекс (для дозаписи новых чанков)
        self._load_meta()

    # ---------- проекция SimHash ----------
    def projection(self) -> np.ndarray:
        """Матрица dim×dim случайных гиперплоскостей. Создаётся один раз и хранится в каталоге индекса."""
        path = self.dir / PROJ_FILE
        if not path.exists() or path.stat().st_size != self.dim * self.dim * 4:
            self.dir.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(PROJ_SEED)
            matrix = rng.standard_normal((self.dim, self.dim)).astype(np.float32)
            tmp = path.with_suffix(".tmp")
            matrix.tofile(tmp)
            os.replace(tmp, path)
        return np.memmap(path, dtype=np.float32, mode="r", shape=(self.dim, self.dim))

    # ---------- метаданные ----------
    @property
    def meta_path(self) -> Path:
        return self.dir / META_FILE

    def _load_meta(self) -> None:
        if not self.meta_path.exists():
            return
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if meta.get("version") != FORMAT_VERSION or meta.get("dim") != self.dim:
            # другой формат или размерность — индекс несовместим, начинаем заново
            self.count, self.cursor = 0, 0
            return
        self.count = int(meta.get("count", 0))
        self.cursor = int(meta.get("cursor", 0))
        # страховка от рассинхронизации файлов (например, прерванная запись): берём минимум
        self.count = min(self.count, self._file_count(self.dir / SIGNS_FILE, self.words * 8),
                         self._file_count(self.dir / INT8_FILE, self.dim),
                         self._file_count(self.dir / IDS_FILE, ID_BYTES))

    @staticmethod
    def _file_count(path: Path, record_bytes: int) -> int:
        if not path.exists():
            return 0
        return path.stat().st_size // record_bytes

    def _save_meta(self) -> None:
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"version": FORMAT_VERSION, "dim": self.dim, "count": self.count, "cursor": self.cursor}),
            encoding="utf-8",
        )
        os.replace(tmp, self.meta_path)

    def rebuild(self) -> None:
        """Удаляет индекс: следующая дозапись начнёт с нуля."""
        with self._lock:
            if self.dir.exists():
                shutil.rmtree(self.dir)
            self.count, self.cursor = 0, 0

    # ---------- запись ----------
    def append(self, chunk_ids: list[str], vectors: np.ndarray, cursor: int | None = None) -> None:
        """Дописывает чанки. vectors — (n, dim) float32. cursor — новый rowid-курсор (если задан)."""
        if not chunk_ids:
            if cursor is not None:
                with self._lock:
                    self.cursor = max(self.cursor, cursor)
                    self._save_meta()
            return
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs.shape != (len(chunk_ids), self.dim):
            raise ValueError(f"ожидалась форма ({len(chunk_ids)}, {self.dim}), получено {vecs.shape}")
        signs = np.zeros((len(chunk_ids), self.words), dtype=np.uint64)
        int8 = np.zeros((len(chunk_ids), self.dim), dtype=np.int8)
        proj = self.projection()
        for row in range(len(chunk_ids)):
            signs[row] = self.kernels.simhash(vecs[row], proj)
            int8[row] = self.kernels.quantize_unit_i8(vecs[row])
        ids = np.array([cid.encode("ascii").ljust(ID_BYTES, b" ")[:ID_BYTES] for cid in chunk_ids],
                       dtype=f"S{ID_BYTES}")
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            # выравниваем файлы по count на случай прерванной прошлой записи
            self._truncate_to(self.count)
            with open(self.dir / SIGNS_FILE, "ab") as f:
                signs.tofile(f)
            with open(self.dir / INT8_FILE, "ab") as f:
                int8.tofile(f)
            with open(self.dir / IDS_FILE, "ab") as f:
                ids.tofile(f)
            self.count += len(chunk_ids)
            if cursor is not None:
                self.cursor = max(self.cursor, cursor)
            self._save_meta()

    def _truncate_to(self, count: int) -> None:
        for name, rec in ((SIGNS_FILE, self.words * 8), (INT8_FILE, self.dim), (IDS_FILE, ID_BYTES)):
            path = self.dir / name
            if path.exists() and path.stat().st_size != count * rec:
                with open(path, "r+b") as f:
                    f.truncate(count * rec)

    # ---------- чтение (zero-copy) ----------
    def _maps(self):
        if self.count == 0:
            return None
        signs = np.memmap(self.dir / SIGNS_FILE, dtype=np.uint64, mode="r", shape=(self.count, self.words))
        int8 = np.memmap(self.dir / INT8_FILE, dtype=np.int8, mode="r", shape=(self.count, self.dim))
        ids = np.memmap(self.dir / IDS_FILE, dtype=f"S{ID_BYTES}", mode="r", shape=(self.count,))
        return signs, int8, ids

    def search(self, query: np.ndarray, k: int, rerank_factor: int) -> list[Candidate]:
        """Top-k по косинусу: POPCNT-скан по сигнатурам → int8-доранжирование кандидатов."""
        maps = self._maps()
        if maps is None or k <= 0:
            return []
        signs, int8, ids = maps
        q = np.asarray(query, dtype=np.float32)
        q_sig = self.kernels.simhash(q, self.projection())
        q_i8 = self.kernels.quantize_unit_i8(q)
        cand = min(max(k * rerank_factor, 200), self.count)
        idx, _dist = self.kernels.topk_hamming(signs, q_sig, cand)
        if idx.size == 0:
            return []
        scores = self.kernels.rerank_i8(int8, idx, q_i8).astype(np.float64) / (127.0 * 127.0)
        order = np.argsort(-scores, kind="stable")[:k]
        out: list[Candidate] = []
        for pos in order:
            raw = bytes(ids[int(idx[pos])]).rstrip(b" ").decode("ascii")
            out.append(Candidate(chunk_id=raw, score=float(scores[pos])))
        return out

    def disk_bytes(self) -> int:
        return sum((self.dir / name).stat().st_size for name in (SIGNS_FILE, INT8_FILE, IDS_FILE)
                   if (self.dir / name).exists())
