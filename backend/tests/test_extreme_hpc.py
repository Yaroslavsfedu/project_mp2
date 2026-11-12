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


# ---------- ранний выход (cascade) и подсказки ядру ----------

def _clustered_sigs(n: int, words: int, seed: int):
    rng = np.random.default_rng(seed)
    centers = rng.integers(0, 2**63, size=(40, words), dtype=np.uint64)
    pick = rng.integers(0, 40, size=n)
    sigs = centers[pick].copy()
    flips = rng.random((n, words * 64)) < 0.18  # шум: 18 % битов
    bits = np.unpackbits(sigs.view(np.uint8), axis=1, bitorder="little")
    bits ^= flips.astype(np.uint8)
    return np.packbits(bits, axis=1, bitorder="little").view(np.uint64).reshape(n, words)


@NATIVE_ONLY
def test_early_exit_with_huge_z_is_exact():
    sigs = _clustered_sigs(20000, 16, seed=1)
    q = sigs[123].copy()
    exact_i, exact_d = K.topk_hamming(sigs, q, 50)
    ee_i, ee_d = K.topk_hamming(sigs, q, 50, early_words=4, z=1e9)  # отсев выключен
    assert np.array_equal(exact_i, ee_i) and np.array_equal(exact_d, ee_d)


@NATIVE_ONLY
def test_early_exit_keeps_most_of_top_k():
    sigs = _clustered_sigs(50000, 16, seed=2)
    overlap = []
    for qi in (5, 900, 4321, 33000):
        q = sigs[qi].copy()
        exact_i, _ = K.topk_hamming(sigs, q, 100)
        ee_i, _ = K.topk_hamming(sigs, q, 100, early_words=4, z=2.0)
        overlap.append(len(set(exact_i.tolist()) & set(ee_i.tolist())) / 100)
    assert float(np.mean(overlap)) >= 0.9, overlap


@NATIVE_ONLY
def test_early_exit_result_is_sorted_and_distances_true():
    sigs = _clustered_sigs(8000, 16, seed=3)
    q = sigs[77].copy()
    idx, dist = K.topk_hamming(sigs, q, 30, early_words=4, z=2.0)
    assert np.all(np.diff(dist) >= 0)
    true = np.array([int(np.unpackbits((sigs[i] ^ q).view(np.uint8)).sum()) for i in idx])
    assert np.array_equal(true, dist)


def test_willneed_hint_is_harmless():
    import tempfile
    from pathlib import Path

    from app.services.mmap_index import _advise_willneed

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.bin"
        np.arange(4096, dtype=np.uint64).tofile(path)
        mm = np.memmap(path, dtype=np.uint64, mode="r", shape=(4096,))
        _advise_willneed(mm, 0, 8 * 100)
        _advise_willneed(mm, 10_000_000, 10_000_100)  # за пределами файла — молча
        _advise_willneed(np.zeros(4), 0, 8)  # без _mmap — молча
    assert True


@NATIVE_ONLY
def test_multi_range_topk_equals_exact_over_union():
    sigs = _clustered_sigs(30000, 16, seed=4)
    q = sigs[4242].copy()
    ranges = [(0, 4000), (10000, 12500), (25000, 30000)]
    got_i, got_d = K.topk_hamming_ranges(sigs, q, ranges, 60)
    union = np.concatenate([np.arange(lo, hi) for lo, hi in ranges])
    dist = np.array([int(np.unpackbits((sigs[i] ^ q).view(np.uint8)).sum()) for i in union])
    order = np.lexsort((union, dist))[:60]
    assert np.array_equal(got_i, union[order]) and np.array_equal(got_d, dist[order])
