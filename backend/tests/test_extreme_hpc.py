"""Итерация 5: wyhash + CountSketch, пакетный OpenMP-режим, FWHT-вращение SimHash, квантование int8 батчем.

Проверяем совпадение C и NumPy там, где оно обязано быть (FWHT, SimHash, квантование), и
совместимость MD5-пути с Python. wyhash-эмбеддинги несовместимы с MD5 по определению — для них проверяем
детерминированность, батч = поштучно и полезность (близкие тексты ближе друг к другу, чем далёкие).
"""
from __future__ import annotations

import numpy as np
import pytest

from app.ml_pipeline.text_utils import hashing_embedding
from app.native.loader import HASH_MD5, HASH_WYHASH, FastKernels, _fwht_numpy, get_kernels, sign_vector

K = get_kernels()
NATIVE_ONLY = pytest.mark.skipif(not K.native, reason="нативная библиотека недоступна")


def _unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


# ---------- хэширование ----------

def test_md5_batch_matches_python_reference():
    texts = ["Договор поставки № 12/2026 от 1 марта", "invoice total 4500 RUB", "", "a b c"]
    batch = K.hash_embedding_batch(texts, 512, HASH_MD5)
    for row, text in zip(batch, texts):
        ref = np.asarray(hashing_embedding(text, 512), dtype=np.float32)
        assert np.allclose(row, ref, atol=2e-5), text


@NATIVE_ONLY
def test_wyhash_batch_equals_one_by_one_and_is_deterministic():
    texts = [f"документ номер {i} про договор поставки {i * 7}" for i in range(300)]
    batch = K.hash_embedding_batch(texts, 1024, HASH_WYHASH)
    single = np.stack([K.hash_embedding_batch([t], 1024, HASH_WYHASH)[0] for t in texts])
    assert np.array_equal(batch, single)  # OpenMP не меняет результат
    again = K.hash_embedding_batch(texts, 1024, HASH_WYHASH)
    assert np.array_equal(batch, again)


@NATIVE_ONLY
def test_wyhash_is_normalized_and_differs_from_md5():
    texts = ["поиск по документам и OCR распознавание", "vector index with popcount scan"]
    w = K.hash_embedding_batch(texts, 1024, HASH_WYHASH)
    m = K.hash_embedding_batch(texts, 1024, HASH_MD5)
    assert np.allclose(np.linalg.norm(w, axis=1), 1.0, atol=1e-5)
    assert not np.allclose(w, m)


@NATIVE_ONLY
def test_wyhash_keeps_similarity_structure():
    base = "договор поставки оборудования между сторонами срок оплаты"
    near = "договор поставки оборудования между сторонами срок платежа"
    far = "погода в регионе ожидается дождь и сильный ветер ночью"
    v = K.hash_embedding_batch([base, near, far], 1024, HASH_WYHASH)
    assert float(v[0] @ v[1]) > float(v[0] @ v[2]) + 0.2


@NATIVE_ONLY
def test_countsketch_signs_produce_negative_components():
    vec = K.hash_embedding_batch(["один два три четыре пять шесть семь восемь девять"], 256, HASH_WYHASH)[0]
    assert (vec < 0).any() and (vec > 0).any()


def test_wyhash_without_native_is_refused():
    fallback = FastKernels.__new__(FastKernels)
    fallback._lib = None
    with pytest.raises(RuntimeError):
        fallback.hash_embedding_batch(["текст"], 64, HASH_WYHASH)


def test_openmp_flag_is_boolean():
    assert isinstance(K.openmp, bool)
    assert K.backend.startswith(("numpy", "c-"))


# ---------- FWHT и SimHash через вращение ----------

@NATIVE_ONLY
@pytest.mark.parametrize("n", [8, 64, 1024])
def test_fwht_is_self_inverse_up_to_scale(n):
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n).astype(np.float32)
    y = x.copy()
    K._lib.fk_fwht(y.ctypes.data, n)
    K._lib.fk_fwht(y.ctypes.data, n)
    assert np.allclose(y / n, x, atol=1e-3)


@NATIVE_ONLY
def test_fwht_native_matches_numpy():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((5, 256)).astype(np.float32)
    ref = _fwht_numpy(x)
    got = x.copy()
    for row in got:
        K._lib.fk_fwht(row.ctypes.data, 256)
    assert np.allclose(got, ref, atol=1e-3)


@pytest.mark.parametrize("dim", [1024, 1000, 300])
def test_simhash_fwht_native_matches_numpy(dim):
    rng = np.random.default_rng(dim)
    vecs = _unit(rng.standard_normal((40, dim)).astype(np.float32))
    dsign, pad = sign_vector(dim, 20261112)
    native = K.simhash_fwht(vecs, dsign, pad)
    # эталон NumPy: те же знаки, то же вращение
    padded = np.zeros((40, pad), dtype=np.float32)
    padded[:, :dim] = vecs * dsign[:dim]
    rotated = _fwht_numpy(padded)[:, :dim]
    bits = np.unpackbits(native.view(np.uint8), axis=1, bitorder="little")[:, :dim]
    ref_bits = (rotated > 0).astype(np.uint8)
    # допускаем расхождение на координатах, где значение почти 0 (разный порядок float-сложений)
    mismatch = np.mean(bits != ref_bits)
    assert mismatch < 0.001, mismatch


def test_simhash_fwht_batch_equals_single_rows():
    rng = np.random.default_rng(5)
    vecs = _unit(rng.standard_normal((25, 512)).astype(np.float32))
    dsign, pad = sign_vector(512, 20261112)
    batch = K.simhash_fwht(vecs, dsign, pad)
    rows = np.stack([K.simhash_fwht(v, dsign, pad) for v in vecs])
    assert np.array_equal(batch, rows)


def test_simhash_hamming_estimates_angle():
    """Расстояние Хэмминга сигнатур оценивает угол: arccos(cos)/π ≈ hamming/bits (с разбросом случайного вращения)."""
    dim = 1024
    rng = np.random.default_rng(11)
    dsign, pad = sign_vector(dim, 20261112)
    a = _unit(rng.standard_normal(dim).astype(np.float32))
    noise = _unit(rng.standard_normal((30, dim)).astype(np.float32))
    mixes = [_unit(a * (1 - t) + n * t) for t, n in zip(np.linspace(0.05, 0.9, 30), noise)]
    sa = K.simhash_fwht(a, dsign, pad)
    errors = []
    for m in mixes:
        sm = K.simhash_fwht(m, dsign, pad)
        ham = int(np.unpackbits((sa ^ sm).view(np.uint8)).sum())
        est = ham / dim
        true = float(np.arccos(np.clip(float(a @ m), -1, 1)) / np.pi)
        errors.append(abs(est - true))
    assert float(np.mean(errors)) < 0.06


def test_sign_vector_is_deterministic_and_pow2():
    d1, p1 = sign_vector(1000, 7)
    d2, p2 = sign_vector(1000, 7)
    assert p1 == 1024 and np.array_equal(d1, d2)
    assert set(np.unique(d1).tolist()) == {-1.0, 1.0}


# ---------- квантование int8 батчем ----------

def test_quantize_batch_equals_rows():
    rng = np.random.default_rng(9)
    vecs = rng.standard_normal((50, 128)).astype(np.float32)
    batch = K.quantize_unit_i8_batch(vecs)
    rows = np.stack([K.quantize_unit_i8(v) for v in vecs])
    assert np.array_equal(batch, rows)
