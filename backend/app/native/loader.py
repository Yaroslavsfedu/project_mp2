"""Загрузка нативных ядер fastkernels.c через ctypes с прозрачным откатом на NumPy.

* Сборка: ``gcc -O3 -march=native -ffast-math -funroll-loops -shared -fPIC`` (при первом использовании
  или если исходник новее библиотеки). Артефакт ``libfastidp.so`` лежит рядом и не попадает в Git.
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
LIBRARY = NATIVE_DIR / "libfastidp.so"
COMPILE_FLAGS = ["-O3", "-march=native", "-ffast-math", "-funroll-loops", "-shared", "-fPIC"]

_lock = threading.Lock()
_lib: ctypes.CDLL | None = None
_load_attempted = False
_build_error: str | None = None


def _needs_build() -> bool:
    if not LIBRARY.exists():
        return True
    return SOURCE.stat().st_mtime > LIBRARY.stat().st_mtime


def build_library(force: bool = False) -> Path | None:
    """Собирает libfastidp.so. Возвращает путь или None (тогда работает NumPy)."""
    global _build_error
    if not force and not _needs_build():
        return LIBRARY
    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        _build_error = "компилятор gcc/cc не найден"
        return None
    cmd = [compiler, *COMPILE_FLAGS, str(SOURCE), "-o", str(LIBRARY), "-lm"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        _build_error = f"сборка не запущена: {exc}"
        return None
    if proc.returncode != 0:
        _build_error = f"ошибка сборки: {proc.stderr.strip()[:500]}"
        return None
    return LIBRARY


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    P = ctypes.c_void_p
    i64 = ctypes.c_int64
    lib.fk_has_avx2.restype = ctypes.c_int
    lib.fk_pack_bits.argtypes = [P, i64, P]
    lib.fk_simhash_bits.argtypes = [P, i64, P, i64, P]
    lib.fk_hamming_scan.argtypes = [P, i64, i64, P, P]
    lib.fk_topk_hamming.argtypes = [P, i64, i64, P, i64, P, P]
    lib.fk_topk_hamming.restype = i64
    lib.fk_quantize_unit_i8.argtypes = [P, i64, P]
    lib.fk_dot_i8.argtypes = [P, P, i64]
    lib.fk_dot_i8.restype = ctypes.c_int32
    lib.fk_rerank_i8.argtypes = [P, i64, P, i64, P, P]
    lib.fk_hash_embed.argtypes = [P, i64, ctypes.c_int32, P]
    lib.fk_hash_embed.restype = i64
    return lib


def _get_lib() -> ctypes.CDLL | None:
    global _lib, _load_attempted
    if os.environ.get("FASTIDP_DISABLE_NATIVE") == "1":
        return None
    with _lock:
        if _load_attempted:
            return _lib
        _load_attempted = True
        path = build_library()
        if path is None:
            logger.warning("Нативные ядра недоступны (%s), используется NumPy", _build_error)
            return None
        try:
            _lib = _bind(ctypes.CDLL(str(path)))
        except OSError as exc:
            logger.warning("Не удалось загрузить %s: %s — используется NumPy", path, exc)
            _lib = None
        return _lib


def _ptr(arr: np.ndarray) -> ctypes.c_void_p:
    return ctypes.c_void_p(arr.ctypes.data)


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):  # NumPy ≥ 2.0
        return np.bitwise_count(x).astype(np.int32)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
    return table[x.view(np.uint8)].reshape(*x.shape, 8).sum(axis=-1).astype(np.int32)


class FastKernels:
    """Единая точка входа. Методы одинаковы для C и NumPy; ``native`` говорит, что активно."""

    def __init__(self) -> None:
        self._lib = _get_lib()

    @property
    def native(self) -> bool:
        return self._lib is not None

    @property
    def backend(self) -> str:
        if self._lib is None:
            return "numpy"
        return "c-avx2" if self._lib.fk_has_avx2() else "c-scalar"

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
        """SimHash-сигнатура: биты = знаки проекции на случайные гиперплоскости proj (dim×bits, float32)."""
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


_kernels: FastKernels | None = None


def get_kernels() -> FastKernels:
    global _kernels
    if _kernels is None:
        _kernels = FastKernels()
    return _kernels
