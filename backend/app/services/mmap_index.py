"""Дисковый векторный индекс на memory-mapped файлах для слабых машин (Ultra-Low-Spec).

Вместо кэша float32 в куче Python (4 КБ на чанк) хранятся три плоских файла на чанк:

* ``signs.bin`` — 1-битная сигнатура SimHash, 16×uint64 = 128 Б (POPCNT-скан Хэмминга). Биты — знаки
  случайной проекции (``proj.f32``, dim×dim, фиксированный seed): расстояние Хэмминга оценивает угол;
* ``int8.bin``  — единичный вектор int8, 1024 Б (доранжирование кандидатов);
* ``ids.bin``   — UUID чанка, 36 Б (связь с таблицей document_chunks).

IVF (этап 1 итерации 5). ``build_ivf`` обучает сферический K-Means (K ≈ 256) на подвыборке int8-векторов,
назначает каждую строку ближайшему центроиду и ПЕРЕУКЛАДЫВАЕТ все три файла так, что векторы одного кластера
лежат подряд. ``offsets.i64`` (K+1) — границы кластеров, ``centroids.f32`` (K×dim) — центроиды.
Поиск: запрос сравнивается с K центроидами, берутся ``nprobe`` ближайших кластеров (и ещё, пока суммарный
размер меньше числа кандидатов), каждый — непрерывный срез файла сигнатур. Строки после последней
перестройки (хвост «дельта») дописываются в конец файлов и сканируются линейно, поэтому дозапись по курсору
работает как раньше. Перестройка нужна, когда дельта выросла (см. ``ivf_needs_build``).

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
CENTROIDS_FILE = "centroids.f32"
OFFSETS_FILE = "offsets.i64"
FORMAT_VERSION = 2  # v2: SimHash-сигнатуры (v1 — знаки координат, несовместимы). IVF-поля необязательны:
#                    индекс v2 без них читается, IVF просто не построен (линейный скан).
IVF_BLOCK = 8192  # строк за раз при назначении кластеров и переукладке (ограничение RAM)
KMEANS_SAMPLE = 20_000
KMEANS_ITERS = 8


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
        self.ivf_count = 0  # строки [0, ivf_count) уложены по кластерам; дальше — хвост-дельта
        self.ivf_clusters = 0
        self._ivf: tuple[np.ndarray, np.ndarray] | None = None  # (centroids K×dim f32, offsets K+1 int64)
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
        self._load_ivf(int(meta.get("ivf_count", 0)), int(meta.get("ivf_clusters", 0)))

    def _load_ivf(self, ivf_count: int, clusters: int) -> None:
        self.ivf_count, self.ivf_clusters, self._ivf = 0, 0, None
        if ivf_count <= 0 or clusters <= 0 or ivf_count > self.count:
            return  # IVF не построен или файлы урезаны — работаем линейно
        c_path, o_path = self.dir / CENTROIDS_FILE, self.dir / OFFSETS_FILE
        if not (c_path.exists() and o_path.exists()):
            return
        if c_path.stat().st_size != clusters * self.dim * 4 or o_path.stat().st_size != (clusters + 1) * 8:
            return
        centroids = np.fromfile(c_path, dtype=np.float32).reshape(clusters, self.dim)
        offsets = np.fromfile(o_path, dtype=np.int64)
        if offsets[0] != 0 or offsets[-1] != ivf_count:
            return
        self.ivf_count, self.ivf_clusters, self._ivf = ivf_count, clusters, (centroids, offsets)

    @staticmethod
    def _file_count(path: Path, record_bytes: int) -> int:
        if not path.exists():
            return 0
        return path.stat().st_size // record_bytes

    def _save_meta(self) -> None:
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"version": FORMAT_VERSION, "dim": self.dim, "count": self.count, "cursor": self.cursor,
                        "ivf_count": self.ivf_count, "ivf_clusters": self.ivf_clusters}),
            encoding="utf-8",
        )
        os.replace(tmp, self.meta_path)

    def rebuild(self) -> None:
        """Удаляет индекс: следующая дозапись начнёт с нуля."""
        with self._lock:
            if self.dir.exists():
                shutil.rmtree(self.dir)
            self.count, self.cursor = 0, 0
            self.ivf_count, self.ivf_clusters, self._ivf = 0, 0, None

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

    def search(self, query: np.ndarray, k: int, rerank_factor: int, nprobe: int = 16) -> list[Candidate]:
        """Top-k по косинусу: POPCNT-скан сигнатур (всех или только выбранных IVF-кластеров) → int8-доранжирование."""
        if k <= 0:
            return []
        # Снимок состояния под блокировкой: файлы и границы кластеров должны относиться к одной перестройке.
        with self._lock:
            maps = self._maps()
            if maps is None:
                return []
            ivf = self._ivf if self.ivf_count > 0 else None
            ivf_count, count = self.ivf_count, self.count
        signs, int8, ids = maps
        q = np.asarray(query, dtype=np.float32)
        q_sig = self.kernels.simhash(q, self.projection())
        q_i8 = self.kernels.quantize_unit_i8(q)
        cand = min(max(k * rerank_factor, 200), count)
        if ivf is None:
            ranges = [(0, count)]
        else:
            ranges = self._probe_ranges(q, ivf, ivf_count, count, cand, nprobe)
        parts_idx, parts_dist = [], []
        for lo, hi in ranges:
            local_idx, local_dist = self.kernels.topk_hamming(signs[lo:hi], q_sig, cand)
            if local_idx.size:
                parts_idx.append(np.asarray(local_idx, dtype=np.int64) + lo)
                parts_dist.append(np.asarray(local_dist))
        if not parts_idx:
            return []
        all_idx = np.concatenate(parts_idx)
        all_dist = np.concatenate(parts_dist)
        if all_idx.size > cand:
            sel = np.argpartition(all_dist, cand - 1)[:cand]
            all_idx = all_idx[sel]
        scores = self.kernels.rerank_i8(int8, all_idx, q_i8).astype(np.float64) / (127.0 * 127.0)
        order = np.argsort(-scores, kind="stable")[:k]
        out: list[Candidate] = []
        for pos in order:
            raw = bytes(ids[int(all_idx[pos])]).rstrip(b" ").decode("ascii")
            out.append(Candidate(chunk_id=raw, score=float(scores[pos])))
        return out

    @staticmethod
    def _probe_ranges(q, ivf, ivf_count: int, count: int, cand: int, nprobe: int) -> list[tuple[int, int]]:
        """Диапазоны строк для поиска: ближайшие кластеры (минимум nprobe и не меньше cand строк) + хвост-дельта."""
        centroids, offsets = ivf
        norm = float(np.linalg.norm(q))
        qn = q / norm if norm > 0 else q
        order = np.argsort(-(centroids @ qn), kind="stable")
        ranges: list[tuple[int, int]] = []
        total = 0
        for i, c in enumerate(order):
            if i >= max(1, nprobe) and total >= cand:
                break
            lo, hi = int(offsets[c]), int(offsets[c + 1])
            if hi > lo:
                ranges.append((lo, hi))
                total += hi - lo
        if ivf_count < count:
            ranges.append((ivf_count, count))
        return ranges

    # ---------- IVF ----------
    def ivf_needs_build(self, min_vectors: int, rebuild_ratio: float) -> bool:
        """Нужно построить/перестроить IVF: строк достаточно, а хвост-дельта вырос сверх доли от уложенной части."""
        if self.count < min_vectors:
            return False
        if self.ivf_count == 0:
            return True
        return (self.count - self.ivf_count) > rebuild_ratio * self.ivf_count

    def build_ivf(self, clusters: int) -> dict:
        """Обучает центроиды (сферический K-Means на подвыборке) и переукладывает файлы по кластерам."""
        with self._lock:
            n = self.count
            k = min(int(clusters), n)
            if n < 2 or k < 2:
                return {"built": False, "count": n}
            signs, int8, ids = self._maps()
            rng = np.random.default_rng(PROJ_SEED)
            m = min(n, max(KMEANS_SAMPLE, k))
            pick = np.sort(rng.choice(n, size=m, replace=False))
            x = _unit_rows(np.asarray(int8[pick], dtype=np.float32))
            cent = x[rng.choice(m, size=k, replace=False)].copy()
            for _ in range(KMEANS_ITERS):
                a = np.argmax(x @ cent.T, axis=1)
                counts = np.bincount(a, minlength=k)
                new = np.zeros_like(cent)
                by_cluster = x[np.argsort(a, kind="stable")]
                starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
                nonempty = counts > 0
                new[nonempty] = np.add.reduceat(by_cluster, starts[nonempty], axis=0)
                empty = np.flatnonzero(~nonempty)
                if empty.size:  # пустой кластер — переинициализируем случайной строкой выборки
                    new[empty] = x[rng.choice(m, size=empty.size, replace=False)]
                cent = _unit_rows(new)
            assign = np.empty(n, dtype=np.int64)
            for s in range(0, n, IVF_BLOCK):
                block = np.asarray(int8[s:s + IVF_BLOCK], dtype=np.float32)
                assign[s:s + IVF_BLOCK] = np.argmax(block @ cent.T, axis=1)
            order = np.argsort(assign, kind="stable")
            counts = np.bincount(assign, minlength=k)
            offsets = np.zeros(k + 1, dtype=np.int64)
            offsets[1:] = np.cumsum(counts)
            # переукладка: каждый файл пишем во временный, потом атомарно подменяем
            _permute_into(signs, order, self.dir / SIGNS_FILE)
            _permute_into(int8, order, self.dir / INT8_FILE)
            _permute_into(ids, order, self.dir / IDS_FILE)
            tmp = self.dir / (CENTROIDS_FILE + ".tmp")
            cent.astype(np.float32).tofile(tmp)
            os.replace(tmp, self.dir / CENTROIDS_FILE)
            tmp = self.dir / (OFFSETS_FILE + ".tmp")
            offsets.tofile(tmp)
            os.replace(tmp, self.dir / OFFSETS_FILE)
            self.ivf_count, self.ivf_clusters = n, k
            self._ivf = (cent.astype(np.float32), offsets)
            self._save_meta()
            return {"built": True, "count": n, "clusters": k,
                    "min_cluster": int(counts.min()), "max_cluster": int(counts.max())}

    def disk_bytes(self) -> int:
        names = (SIGNS_FILE, INT8_FILE, IDS_FILE, CENTROIDS_FILE, OFFSETS_FILE)
        return sum((self.dir / name).stat().st_size for name in names if (self.dir / name).exists())


def _unit_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


def _permute_into(src: np.ndarray, order: np.ndarray, path: Path) -> None:
    """Пишет строки src в порядке order во временный файл и атомарно заменяет path (блоками, без полной копии в RAM)."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        for s in range(0, order.size, IVF_BLOCK):
            src[order[s:s + IVF_BLOCK]].tofile(f)
    os.replace(tmp, path)
