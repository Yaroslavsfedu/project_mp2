"""Дисковый векторный индекс на memory-mapped файлах для слабых машин (Ultra-Low-Spec).

Вместо кэша float32 в куче Python (4 КБ на чанк) хранятся плоские файлы, по строке на чанк:

* ``signs.bin`` — 1-битная сигнатура SimHash, 16×uint64 = 128 Б (POPCNT-скан Хэмминга). Биты — знаки
  H·D·v (Адамар-вращение со случайными знаками D, seed PROJ_SEED, O(D log D) на вектор; формат v3);
* доранжирование, один из двух вариантов (``rerank_bits``):
  - ``int8.bin``  — единичный вектор int8, 1024 Б на 1024 измерения (rerank_bits = 8);
  - ``int4.bin``  — 4-битные полубайты, 512 Б, + ``scale.f32`` (масштаб строки, 4 Б) (rerank_bits = 4);
* ``ids.bin``   — UUID чанка в бинарном виде, 16 Б (связь с таблицей document_chunks; строка восстанавливается при выборке).

IVF. ``build_ivf`` обучает сферический K-Means (K ≈ 256) на подвыборке, назначает строки ближайшему центроиду
и ПЕРЕУКЛАДЫВАЕТ файлы так, что векторы одного кластера лежат подряд (``offsets.i64`` — границы,
``centroids.f32`` — центроиды). Поиск: ближайшие ``nprobe`` кластеров (и ещё, пока суммарный размер
меньше числа кандидатов) плюс хвост-дельта — строки после последней перестройки, дописанные в конец файлов.

Файлы только дописываются; удалённые чанки остаются в индексе и отфильтровываются при выборке
по таблице (см. vector_store.top_chunks_mmap). Полная пересборка — ``rebuild()``.
"""
from __future__ import annotations

import json
import mmap as mmap_module
import os
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.native.loader import get_kernels, sign_vector

META_FILE = "meta.json"
PROJ_SEED = 20261112  # фиксированный seed: построение и запрос обязаны использовать одни и те же знаки
SIGNS_FILE = "signs.bin"
INT8_FILE = "int8.bin"
INT4_FILE = "int4.bin"
SCALE_FILE = "scale.f32"
IDS_FILE = "ids.bin"
ID_BYTES = 16  # бинарный UUID (v4): вместо 36-символьной строки экономит 20 Б на чанк (~400 МБ на 20M)
CENTROIDS_FILE = "centroids.f32"
OFFSETS_FILE = "offsets.i64"
FORMAT_VERSION = 4  # v4: ids.bin — 16-байтные UUID (v3 — строки по 36 Б, несовместимы → переиндексация).
#                    v3: SimHash через FWHT-вращение (v2 — плотная проекция, несовместима → переиндексация).
#                    IVF-поля и поля rerank в meta необязательны: отсутствующие значат «нет IVF» и int8.
IVF_BLOCK = 8192  # строк за раз при назначении кластеров и переукладке (ограничение RAM)
KMEANS_SAMPLE = 20_000
KMEANS_ITERS = 8


def _uuid_bytes(chunk_id: str) -> bytes:
    """16 байт UUID для записи в ids.bin; нестроковый UUID — явная ошибка, а не тихая порча."""
    try:
        return uuid.UUID(chunk_id).bytes
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"chunk_id должен быть UUID, получено {chunk_id!r}") from exc


@dataclass
class Candidate:
    chunk_id: str
    score: float  # косинус (приближённо) по int8 или int4 (1.0 = совпадение направления)


class MmapVectorIndex:
    def __init__(self, directory: Path, dim: int, rerank_bits: int = 8) -> None:
        if rerank_bits not in (4, 8):
            raise ValueError("rerank_bits должен быть 4 или 8")
        if rerank_bits == 4 and dim % 2:
            raise ValueError("int4 требует чётной размерности")
        self.dir = Path(directory)
        self.dim = dim
        self.words = (dim + 63) // 64
        self.rerank_bits = rerank_bits
        self._rerank_file = INT4_FILE if rerank_bits == 4 else INT8_FILE
        self._rerank_rec = dim // 2 if rerank_bits == 4 else dim  # байт на строку доранжирования
        self._lock = threading.Lock()
        self.kernels = get_kernels()
        self.count = 0
        self.cursor = 0  # максимальный rowid, уже попавший в индекс (для дозаписи новых чанков)
        self.ivf_count = 0  # строки [0, ivf_count) уложены по кластерам; дальше — хвост-дельта
        self.ivf_clusters = 0
        self._ivf: tuple[np.ndarray, np.ndarray] | None = None  # (centroids K×dim f32, offsets K+1 int64)
        self._rot: tuple[np.ndarray, int] | None = None
        self._load_meta()

    # ---------- SimHash через вращение H·D ----------
    @property
    def rotation(self) -> tuple[np.ndarray, int]:
        """Знаки D и размер pad (степень двойки ≥ dim). Детерминированы seed-ом: построение и запрос совпадают."""
        if self._rot is None:
            self._rot = sign_vector(self.dim, PROJ_SEED)
        return self._rot

    # ---------- метаданные ----------
    @property
    def meta_path(self) -> Path:
        return self.dir / META_FILE

    def _load_meta(self) -> None:
        if not self.meta_path.exists():
            return
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        bits = int(meta.get("rerank_bits", 8))  # индексы без поля созданы int8
        if meta.get("version") != FORMAT_VERSION or meta.get("dim") != self.dim or bits != self.rerank_bits:
            # другой формат, размерность или способ доранжирования — индекс несовместим, начинаем заново
            self.count, self.cursor = 0, 0
            return
        self.count = int(meta.get("count", 0))
        self.cursor = int(meta.get("cursor", 0))
        # страховка от рассинхронизации файлов (например, прерванная запись): берём минимум
        counts = [self.count,
                  self._file_count(self.dir / SIGNS_FILE, self.words * 8),
                  self._file_count(self.dir / self._rerank_file, self._rerank_rec),
                  self._file_count(self.dir / IDS_FILE, ID_BYTES)]
        if self.rerank_bits == 4:
            counts.append(self._file_count(self.dir / SCALE_FILE, 4))
        self.count = min(counts)
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
        self.dir.mkdir(parents=True, exist_ok=True)
        # уникальное имя временного файла: два объекта на одном каталоге не должны мешать друг другу
        tmp = self.dir / f".meta.{os.getpid()}.{threading.get_ident()}.tmp"
        tmp.write_text(
            json.dumps({"version": FORMAT_VERSION, "dim": self.dim, "count": self.count, "cursor": self.cursor,
                        "rerank_bits": self.rerank_bits,
                        "ivf_count": self.ivf_count, "ivf_clusters": self.ivf_clusters}),
            encoding="utf-8",
        )
        os.replace(tmp, self.meta_path)

    def _reset_if_removed(self) -> None:
        """Каталог индекса удалён извне (rebuild другого объекта, очистка данных): строк больше нет.
        Вызывать под self._lock. Без этого закэшированный объект сообщает строки, которых на диске нет."""
        if self.count > 0 and not self.meta_path.exists():
            self.count, self.cursor = 0, 0
            self.ivf_count, self.ivf_clusters, self._ivf = 0, 0, None

    def check_consistency(self) -> None:
        """Публичная проверка: вызывать ДО чтения курсора (иначе курсор от удалённого индекса пропустит строки)."""
        with self._lock:
            self._reset_if_removed()

    def rebuild(self) -> None:
        """Удаляет индекс: следующая дозапись начнёт с нуля."""
        with self._lock:
            if self.dir.exists():
                shutil.rmtree(self.dir)
            self.count, self.cursor = 0, 0
            self.ivf_count, self.ivf_clusters, self._ivf = 0, 0, None

    def _layout(self) -> list[tuple[str, int]]:
        """(имя файла, байт на строку) для всех файлов по строкам. Порядок важен только внутри одного индекса."""
        layout = [(SIGNS_FILE, self.words * 8), (self._rerank_file, self._rerank_rec), (IDS_FILE, ID_BYTES)]
        if self.rerank_bits == 4:
            layout.append((SCALE_FILE, 4))
        return layout

    # ---------- запись ----------
    def append(self, chunk_ids: list[str], vectors: np.ndarray, cursor: int | None = None) -> None:
        """Дописывает чанки. vectors — (n, dim) float32. cursor — новый rowid-курсор (если задан)."""
        if not chunk_ids:
            if cursor is not None:
                with self._lock:
                    self._reset_if_removed()
                    self.cursor = max(self.cursor, cursor)
                    self._save_meta()
            return
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs.shape != (len(chunk_ids), self.dim):
            raise ValueError(f"ожидалась форма ({len(chunk_ids)}, {self.dim}), получено {vecs.shape}")
        dsign, pad = self.rotation
        signs = self.kernels.simhash_fwht(vecs, dsign, pad)  # пакетно (OpenMP, если доступно)
        if self.rerank_bits == 4:
            rerank, scales = self.kernels.quantize_i4_batch(vecs)
        else:
            rerank, scales = self.kernels.quantize_unit_i8_batch(vecs), None
        ids = np.frombuffer(b"".join(_uuid_bytes(cid) for cid in chunk_ids), dtype=np.uint8).reshape(len(chunk_ids), ID_BYTES)
        payload = {SIGNS_FILE: signs, self._rerank_file: rerank, IDS_FILE: ids}
        if scales is not None:
            payload[SCALE_FILE] = scales
        with self._lock:
            self._reset_if_removed()
            self.dir.mkdir(parents=True, exist_ok=True)
            # выравниваем файлы по count на случай прерванной прошлой записи
            self._truncate_to(self.count)
            for name, _rec in self._layout():
                with open(self.dir / name, "ab") as f:
                    np.ascontiguousarray(payload[name]).tofile(f)
            self.count += len(chunk_ids)
            if cursor is not None:
                self.cursor = max(self.cursor, cursor)
            self._save_meta()

    def _truncate_to(self, count: int) -> None:
        for name, rec in self._layout():
            path = self.dir / name
            if path.exists() and path.stat().st_size != count * rec:
                with open(path, "r+b") as f:
                    f.truncate(count * rec)

    # ---------- чтение (zero-copy) ----------
    def _maps(self):
        """(signs, rerank, ids, scales|None) как memmap; None, если строк нет."""
        if self.count == 0:
            return None
        signs = np.memmap(self.dir / SIGNS_FILE, dtype=np.uint64, mode="r", shape=(self.count, self.words))
        if self.rerank_bits == 4:
            rerank = np.memmap(self.dir / INT4_FILE, dtype=np.uint8, mode="r", shape=(self.count, self._rerank_rec))
            scales = np.memmap(self.dir / SCALE_FILE, dtype=np.float32, mode="r", shape=(self.count,))
        else:
            rerank = np.memmap(self.dir / INT8_FILE, dtype=np.int8, mode="r", shape=(self.count, self.dim))
            scales = None
        ids = np.memmap(self.dir / IDS_FILE, dtype=np.uint8, mode="r", shape=(self.count, ID_BYTES))
        return signs, rerank, ids, scales

    def search(self, query: np.ndarray, k: int, rerank_factor: int, nprobe: int = 16, *,
               early_words: int = 4, early_z: float = 2.0, prefetch: bool = True) -> list[Candidate]:
        """Top-k по косинусу: POPCNT-скан сигнатур (всех или только выбранных IVF-кластеров) → доранжирование."""
        if k <= 0:
            return []
        # Снимок состояния под блокировкой: файлы и границы кластеров должны относиться к одной перестройке.
        with self._lock:
            self._reset_if_removed()
            maps = self._maps()
            if maps is None:
                return []
            ivf = self._ivf if self.ivf_count > 0 else None
            ivf_count, count = self.ivf_count, self.count
        signs, rerank, ids, scales = maps
        q = np.asarray(query, dtype=np.float32)
        dsign, pad = self.rotation
        q_sig = self.kernels.simhash_fwht(q, dsign, pad)
        q_i8 = self.kernels.quantize_unit_i8(q)
        cand = min(max(k * rerank_factor, 200), count)
        if ivf is None:
            ranges = [(0, count)]
        else:
            ranges = self._probe_ranges(q, ivf, ivf_count, count, cand, nprobe)
        early = early_words if self.words >= 8 else 0  # ранний выход имеет смысл для длинных сигнатур
        if prefetch:
            for lo, hi in ranges:
                _advise_willneed(signs, lo * self.words * 8, hi * self.words * 8)
        # одна куча на все выбранные кластеры и дельту: общий порог быстро ужесточается
        all_idx, _ = self.kernels.topk_hamming_ranges(signs, q_sig, ranges, cand,
                                                      early_words=early, z=early_z)
        if all_idx.size == 0:
            return []
        if self.rerank_bits == 4:
            scores = self.kernels.rerank_i4(rerank, all_idx, q_i8, scales).astype(np.float64)
        else:
            scores = self.kernels.rerank_i8(rerank, all_idx, q_i8).astype(np.float64) / (127.0 * 127.0)
        order = np.argsort(-scores, kind="stable")[:k]
        out: list[Candidate] = []
        for pos in order:
            raw = str(uuid.UUID(bytes=bytes(ids[int(all_idx[pos])])))
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
            signs, rerank, ids, scales = self._maps()
            rng = np.random.default_rng(PROJ_SEED)
            m = min(n, max(KMEANS_SAMPLE, k))
            pick = np.sort(rng.choice(n, size=m, replace=False))
            x = _unit_rows(self._dequantize(rerank, scales, pick))
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
                block = _unit_rows(self._dequantize(rerank, scales, np.arange(s, min(s + IVF_BLOCK, n))))
                assign[s:s + IVF_BLOCK] = np.argmax(block @ cent.T, axis=1)
            order = np.argsort(assign, kind="stable")
            counts = np.bincount(assign, minlength=k)
            offsets = np.zeros(k + 1, dtype=np.int64)
            offsets[1:] = np.cumsum(counts)
            # переукладка: каждый файл пишем во временный, потом атомарно подменяем
            sources = {SIGNS_FILE: signs, self._rerank_file: rerank, IDS_FILE: ids}
            if scales is not None:
                sources[SCALE_FILE] = scales
            for name, _rec in self._layout():
                _permute_into(sources[name], order, self.dir / name)
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

    def _dequantize(self, rerank, scales, rows) -> np.ndarray:
        """Приблизительные float-векторы строк (для K-Means и назначения кластеров)."""
        rows = np.asarray(rows, dtype=np.int64)
        if self.rerank_bits == 4:
            packed = np.asarray(rerank[rows], dtype=np.int32)
            u = np.concatenate([(packed & 0x0F) - 8, (packed >> 4) - 8], axis=1).astype(np.float32)
            return u * np.asarray(scales[rows], dtype=np.float32)[:, None]
        return np.asarray(rerank[rows], dtype=np.float32)

    def disk_bytes(self) -> int:
        names = [SIGNS_FILE, INT8_FILE, INT4_FILE, SCALE_FILE, IDS_FILE, CENTROIDS_FILE, OFFSETS_FILE]
        return sum((self.dir / name).stat().st_size for name in names if (self.dir / name).exists())


def _unit_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


def _permute_into(src: np.ndarray, order: np.ndarray, path: Path) -> None:
    """Пишет строки src в порядке order во временный файл и атомарно заменяет path (блоками, без полной копии в RAM)."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        for s in range(0, order.size, IVF_BLOCK):
            np.ascontiguousarray(src[order[s:s + IVF_BLOCK]]).tofile(f)
    os.replace(tmp, path)


def _advise_willneed(arr: np.ndarray, byte_lo: int, byte_hi: int) -> None:
    """madvise(MADV_WILLNEED) для диапазона отображения: ядро начинает подгружать страницы кластера заранее.

    Подсказка необязательна: без поддержки (не Linux, нет _mmap) тихо ничего не делает.
    """
    mm = getattr(arr, "_mmap", None)
    if mm is None or byte_hi <= byte_lo or not hasattr(mm, "madvise"):
        return
    page = mmap_module.PAGESIZE
    start = (byte_lo // page) * page
    length = min(byte_hi - start, len(mm) - start)
    if length <= 0:
        return
    try:
        mm.madvise(mmap_module.MADV_WILLNEED, start, length)
    except (OSError, ValueError):
        pass
