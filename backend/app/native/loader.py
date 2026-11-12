"""Загрузка нативных ядер fastkernels.c через ctypes с прозрачным откатом на NumPy.

* Сборка: ``gcc -O3 -march=native -ffast-math -funroll-loops -fopenmp -shared -fPIC`` при первом использовании
  или если исходник новее библиотеки. Артефакты ``libfastidp.so`` (с OpenMP) и ``libfastidp_st.so`` (без OpenMP,
  если libgomp недоступна) лежат рядом и не попадают в Git.
* Нет gcc / сборка упала / ``FASTIDP_DISABLE_NATIVE=1`` → класс ``FastKernels`` работает на NumPy.
  Результаты совпадают (тесты проверяют это), меняется только скорость.
"""
from __future__ import annotations

import ctypes
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

NATIVE_DIR = Path(__file__).resolve().parent
SOURCE = NATIVE_DIR / "fastkernels.c"
LIBRARY_OMP = NATIVE_DIR / "libfastidp.so"
LIBRARY_ST = NATIVE_DIR / "libfastidp_st.so"
BASE_FLAGS = ["-O3", "-march=native", "-ffast-math", "-funroll-loops", "-shared", "-fPIC"]

HASH_MD5 = 0
HASH_WYHASH = 1

_lock = threading.Lock()
_lib: ctypes.CDLL | None = None
_load_attempted = False
_build_error: str | None = None
_openmp = False


def _needs_build(path: Path) -> bool:
    if not path.exists():
        return True
    return SOURCE.stat().st_mtime > path.stat().st_mtime


def build_library(force: bool = False, openmp: bool = True) -> Path | None:
    """Собирает библиотеку (с OpenMP или без). Возвращает путь или None (тогда работает NumPy)."""
    global _build_error
    path = LIBRARY_OMP if openmp else LIBRARY_ST
    if not force and not _needs_build(path):
        return path
    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        _build_error = "компилятор gcc/cc не найден"
        return None
    flags = BASE_FLAGS + (["-fopenmp"] if openmp else [])
    cmd = [compiler, *flags, str(SOURCE), "-o", str(path), "-lm"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        _build_error = f"сборка не запущена: {exc}"
        return None
    if proc.returncode != 0:
        _build_error = f"ошибка сборки: {proc.stderr.strip()[:500]}"
        return None
    return path


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    P = ctypes.c_void_p
    i64 = ctypes.c_int64
    i32 = ctypes.c_int32
    lib.fk_has_avx2.restype = ctypes.c_int
    lib.fk_has_openmp.restype = i32
    lib.fk_pack_bits.argtypes = [P, i64, P]
    lib.fk_simhash_bits.argtypes = [P, i64, P, i64, P]
    lib.fk_hamming_scan.argtypes = [P, i64, i64, P, P]
    lib.fk_topk_hamming.argtypes = [P, i64, i64, P, i64, P, P]
    lib.fk_topk_hamming.restype = i64
    lib.fk_quantize_unit_i8.argtypes = [P, i64, P]
    lib.fk_quantize_unit_i8_batch.argtypes = [P, i64, i64, P]
    lib.fk_dot_i8.argtypes = [P, P, i64]
    lib.fk_dot_i8.restype = ctypes.c_int32
    lib.fk_rerank_i8.argtypes = [P, i64, P, i64, P, P]
    lib.fk_hash_embed.argtypes = [P, i64, ctypes.c_int32, P]
    lib.fk_hash_embed.restype = i64
    lib.fk_hash_embed_batch.argtypes = [P, P, i64, i32, i32, P]
    lib.fk_fwht.argtypes = [P, i64]
    lib.fk_simhash_fwht.argtypes = [P, i64, P, i64, i64, P]
    lib.fk_simhash_fwht_batch.argtypes = [P, i64, i64, P, i64, i64, P]
    return lib


def _try_load(path: Path) -> ctypes.CDLL | None:
    try:
        return _bind(ctypes.CDLL(str(path)))
    except (OSError, AttributeError) as exc:
        logger.warning("Не удалось загрузить %s: %s", path, exc)
        return None


def _get_lib() -> ctypes.CDLL | None:
    global _lib, _load_attempted, _openmp
    if os.environ.get("FASTIDP_DISABLE_NATIVE") == "1":
        return None
    with _lock:
        if _load_attempted:
            return _lib
        _load_attempted = True
        # сначала с OpenMP; если сборка или загрузка не удалась (нет libgomp) — однопоточная версия
        for openmp in (True, False):
            path = build_library(openmp=openmp)
            if path is None:
                continue
            lib = _try_load(path)
            if lib is not None:
                _lib, _openmp = lib, bool(lib.fk_has_openmp())
                return _lib
        logger.warning("Нативные ядра недоступны (%s), используется NumPy", _build_error)
        return None


def _ptr(arr: np.ndarray) -> ctypes.c_void_p:
    return ctypes.c_void_p(arr.ctypes.data)


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):  # NumPy ≥ 2.0
        return np.bitwise_count(x).astype(np.int32)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
    return table[x.view(np.uint8)].reshape(*x.shape, 8).sum(axis=-1).astype(np.int32)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def sign_vector(dim: int, seed: int) -> tuple[np.ndarray, int]:
    """Случайные знаки D (длина pad = степень двойки ≥ dim) для вращения SimHash. Детерминированы по seed."""
    pad = _next_pow2(dim)
    rng = np.random.default_rng(seed)
    dsign = np.where(rng.integers(0, 2, size=pad) == 1, 1.0, -1.0).astype(np.float32)
    return dsign, pad


def _fwht_numpy(x: np.ndarray) -> np.ndarray:
    """FWHT по последней оси (длина — степень двойки), без нормировки. Та же математика, что fk_fwht."""
    shape = x.shape
    pad = shape[-1]
    y = x.reshape(-1, pad).astype(np.float32, copy=True)
    h = 1
    while h < pad:
        y = y.reshape(-1, pad // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = np.stack([a + b, a - b], axis=2).reshape(-1, pad)
        h *= 2
    return y.reshape(shape)


class FastKernels:
    """Единая точка входа. Методы одинаковы для C и NumPy; ``native`` говорит, что активно."""

    def __init__(self) -> None:
        self._lib = _get_lib()

    @property
    def native(self) -> bool:
        return self._lib is not None

    @property
    def openmp(self) -> bool:
        return self._lib is not None and _openmp

    @property
    def backend(self) -> str:
        if self._lib is None:
            return "numpy"
        isa = "c-avx2" if self._lib.fk_has_avx2() else "c-scalar"
        return isa + ("-omp" if self.openmp else "")

    # ---- 1. 1-битные сигнатуры ----
    @staticmethod
    def words_for(dim: int) -> int:
        return (dim + 63) // 64

    def pack_bits(self, vec: np.ndarray) -> np.ndarray:
        v = np.ascontiguousarray(vec, dtype=np.float32)
        dim = v.shape[0]
        out = np.zeros(self.words_for(dim), dtype=np.uint64)
        if self._lib is not None:
            self._lib.fk_pack_bits(_ptr(v), dim, _ptr(out))
            return out
        mean = np.float32(v.astype(np.float64).sum() / max(dim, 1))
        bits = np.zeros(out.size * 64, dtype=bool)
        bits[:dim] = v > mean
        return np.packbits(bits, bitorder="little").view("<u8").astype(np.uint64)

    def simhash(self, vec: np.ndarray, proj: np.ndarray) -> np.ndarray:
        """SimHash с плотной матрицей проекции proj (dim×bits). Оставлен для совместимости и сравнения."""
        v = np.ascontiguousarray(vec, dtype=np.float32)
        r = np.ascontiguousarray(proj, dtype=np.float32)
        dim, bits = r.shape
        out = np.zeros(self.words_for(bits), dtype=np.uint64)
        if self._lib is not None:
            self._lib.fk_simhash_bits(_ptr(v), dim, _ptr(r), bits, _ptr(out))
            return out
        acc = v.astype(np.float64) @ r.astype(np.float64)
        bools = np.zeros(out.size * 64, dtype=bool)
        bools[:bits] = acc > 0
        return np.packbits(bools, bitorder="little").view("<u8").astype(np.uint64)

    def simhash_fwht(self, vecs: np.ndarray, dsign: np.ndarray, pad: int) -> np.ndarray:
        """SimHash через вращение H·D (O(D log D)). vecs: (dim,) или (n, dim). Возвращает (words,) или (n, words)."""
        single = vecs.ndim == 1
        batch = np.ascontiguousarray(np.atleast_2d(vecs), dtype=np.float32)
        n, dim = batch.shape
        bits = dim
        words = self.words_for(bits)
        out = np.zeros((n, words), dtype=np.uint64)
        if self._lib is not None and n:
            if n == 1:
                self._lib.fk_simhash_fwht(_ptr(batch), dim, _ptr(dsign), pad, bits, _ptr(out))
            else:
                self._lib.fk_simhash_fwht_batch(_ptr(batch), n, dim, _ptr(dsign), pad, bits, _ptr(out))
        elif n:
            padded = np.zeros((n, pad), dtype=np.float32)
            padded[:, :dim] = batch * dsign[:dim]
            rotated = _fwht_numpy(padded)[:, :bits]
            bools = np.zeros((n, words * 64), dtype=bool)
            bools[:, :bits] = rotated > 0
            out = np.packbits(bools, axis=1, bitorder="little").view("<u8").astype(np.uint64)
        return out[0] if single else out

    def topk_hamming(self, sigs: np.ndarray, query_sig: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k по расстоянию Хэмминга. sigs — (n, words) uint64, может быть memmap (без копий)."""
        n, words = sigs.shape
        k = min(k, n)
        if k <= 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int32)
        q = np.ascontiguousarray(query_sig, dtype=np.uint64)
        if self._lib is not None:
            idx = np.zeros(k, dtype=np.int64)
            dist = np.zeros(k, dtype=np.int32)
            got = self._lib.fk_topk_hamming(_ptr(sigs), n, words, _ptr(q), k, _ptr(idx), _ptr(dist))
            if got < 0:
                raise MemoryError("fk_topk_hamming: нехватка памяти")
            return idx[:got], dist[:got]
        # NumPy: блоками, чтобы не держать весь скан в памяти
        best_d = np.zeros(0, np.int32)
        best_i = np.zeros(0, np.int64)
        block = 65536
        for start in range(0, n, block):
            chunk = np.asarray(sigs[start:start + block])
            d = _popcount(chunk ^ q).sum(axis=1).astype(np.int32)
            cand_d = np.concatenate([best_d, d])
            cand_i = np.concatenate([best_i, np.arange(start, start + d.size, dtype=np.int64)])
            order = np.lexsort((cand_i, cand_d))[:k]
            best_d, best_i = cand_d[order], cand_i[order]
        return best_i, best_d

    # ---- 2. int8 SQ8 ----
    def quantize_unit_i8(self, vec: np.ndarray) -> np.ndarray:
        v = np.ascontiguousarray(vec, dtype=np.float32)
        out = np.zeros(v.shape[0], dtype=np.int8)
        if self._lib is not None:
            self._lib.fk_quantize_unit_i8(_ptr(v), v.shape[0], _ptr(out))
            return out
        norm = float(np.sqrt(np.sum(v.astype(np.float64) ** 2)))
        if norm == 0.0:
            return out
        x = v.astype(np.float64) * (127.0 / norm)
        r = np.sign(x) * np.floor(np.abs(x) + 0.5)  # округление как lround: half away from zero
        return np.clip(r, -127, 127).astype(np.int8)

    def quantize_unit_i8_batch(self, vecs: np.ndarray) -> np.ndarray:
        """Батч квантования (n, dim) → (n, dim) int8; с OpenMP, если доступно."""
        batch = np.ascontiguousarray(vecs, dtype=np.float32)
        n, dim = batch.shape
        out = np.zeros((n, dim), dtype=np.int8)
        if self._lib is not None and n:
            self._lib.fk_quantize_unit_i8_batch(_ptr(batch), n, dim, _ptr(out))
            return out
        for i in range(n):
            out[i] = self.quantize_unit_i8(batch[i])
        return out

    def rerank_i8(self, base: np.ndarray, idx: np.ndarray, query_i8: np.ndarray) -> np.ndarray:
        """Скалярные произведения int8 для выбранных строк base (n, dim) → int32 (m,)."""
        m = idx.shape[0]
        scores = np.zeros(m, dtype=np.int32)
        if m == 0:
            return scores
        q = np.ascontiguousarray(query_i8, dtype=np.int8)
        if self._lib is not None:
            ids = np.ascontiguousarray(idx, dtype=np.int64)
            self._lib.fk_rerank_i8(_ptr(base), base.shape[1], _ptr(ids), m, _ptr(q), _ptr(scores))
            return scores
        rows = np.asarray(base[np.asarray(idx, dtype=np.int64)], dtype=np.int32)
        return (rows @ q.astype(np.int32)).astype(np.int32)

    # ---- 3. хэширование текста ----
    def hash_embedding(self, text: str, dim: int) -> np.ndarray:
        """Побитово совместимо с ml_pipeline.text_utils.hashing_embedding (до округления до 5 знаков)."""
        raw = text.encode("utf-8")
        out = np.zeros(dim, dtype=np.float32)
        if self._lib is not None:
            buf = ctypes.create_string_buffer(raw, len(raw)) if raw else ctypes.create_string_buffer(b"", 1)
            self._lib.fk_hash_embed(ctypes.cast(buf, ctypes.c_void_p), len(raw), dim, _ptr(out))
            return out
        from app.ml_pipeline.text_utils import hashing_embedding  # noqa: PLC0415 - ленивый импорт

        return np.asarray(hashing_embedding(text, dim), dtype=np.float32)

    def hash_embedding_batch(self, texts: list[str], dim: int, method: int = HASH_MD5) -> np.ndarray:
        """Батч: (n, dim). method=HASH_WYHASH работает только нативно (с OpenMP-параллелизмом).

        Без нативной библиотеки wyhash недоступен: метод молча не подменяется, поэтому выбрасывается ошибка.
        """
        n = len(texts)
        out = np.zeros((n, dim), dtype=np.float32)
        if self._lib is None:
            if method == HASH_WYHASH:
                raise RuntimeError("wyhash доступен только с нативной библиотекой (см. loader.build_library)")
            for i, t in enumerate(texts):
                out[i] = self.hash_embedding(t, dim)
            return out
        encoded = [t.encode("utf-8") for t in texts]
        offs = np.zeros(n + 1, dtype=np.int64)
        offs[1:] = np.cumsum([len(b) for b in encoded])
        buf = np.frombuffer(b"".join(encoded) or b"\0", dtype=np.uint8).copy()
        if n:
            self._lib.fk_hash_embed_batch(_ptr(buf), _ptr(offs), n, dim, int(method), _ptr(out))
        return out


_kernels: FastKernels | None = None


def get_kernels() -> FastKernels:
    global _kernels
    if _kernels is None:
        _kernels = FastKernels()
    return _kernels
