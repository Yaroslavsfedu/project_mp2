"""IVF-кластеризация mmap-индекса (итерация 5, этап 1): укладка по кластерам, recall, дельта, перезагрузка."""
from __future__ import annotations

import json
import uuid

import numpy as np
import pytest

from app.services.mmap_index import META_FILE, MmapVectorIndex

DIM = 128


def _uid(i: int) -> str:
    """Детерминированный UUID для строки i (ids.bin хранит только UUID)."""
    return str(uuid.UUID(int=i + 1))


def _num(chunk_id: str) -> int:
    return uuid.UUID(chunk_id).int - 1


def _clustered(n: int, centers: int, seed: int, noise: float = 0.6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((centers, DIM)).astype(np.float32)
    v = c[rng.integers(0, centers, size=n)] + rng.standard_normal((n, DIM)).astype(np.float32) * noise
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def _exact_top(vectors: np.ndarray, q: np.ndarray, k: int) -> set[int]:
    scores = vectors @ (q / np.linalg.norm(q))
    return set(np.argsort(-scores, kind="stable")[:k].tolist())


def _index_with(tmp_path, vectors: np.ndarray, name: str = "ivf") -> MmapVectorIndex:
    index = MmapVectorIndex(tmp_path / name, DIM)
    index.append([_uid(i) for i in range(len(vectors))], vectors)
    return index


def test_build_ivf_layout_is_contiguous_and_complete(tmp_path):
    vecs = _clustered(3000, 30, seed=1)
    index = _index_with(tmp_path, vecs)
    stats = index.build_ivf(clusters=32)
    assert stats["built"] and stats["clusters"] == 32
    centroids, offsets = index._ivf
    assert offsets[0] == 0 and offsets[-1] == index.count == index.ivf_count == 3000
    assert np.all(np.diff(offsets) >= 0)
    assert centroids.shape == (32, DIM)


def test_ids_stay_attached_to_vectors_after_relayout(tmp_path):
    vecs = _clustered(2000, 20, seed=2)
    index = _index_with(tmp_path, vecs)
    index.build_ivf(clusters=20)
    # точный запрос по вектору i должен вернуть его собственный chunk_id первым
    for i in (0, 17, 999, 1999):
        top = index.search(vecs[i], 1, rerank_factor=20, nprobe=20)
        assert top and top[0].chunk_id == _uid(i)
        assert top[0].score > 0.95


def test_ivf_recall_on_clustered_data(tmp_path):
    vecs = _clustered(6000, 60, seed=3)
    index = _index_with(tmp_path, vecs)
    index.build_ivf(clusters=64)
    rng = np.random.default_rng(4)
    hits = 0
    total = 0
    for _ in range(40):
        q = vecs[int(rng.integers(0, len(vecs)))] + rng.standard_normal(DIM).astype(np.float32) * 0.05
        truth = _exact_top(vecs, q, 10)
        got = {_num(c.chunk_id) for c in index.search(q, 10, rerank_factor=20, nprobe=6)}
        hits += len(truth & got)
        total += 10
    recall = hits / total
    # просматривается около 6/64 ≈ 10 % строк; на кластерных данных recall должен оставаться высоким
    assert recall >= 0.85, recall


def test_probe_scans_only_a_fraction_of_rows(tmp_path):
    vecs = _clustered(4000, 40, seed=5)
    index = _index_with(tmp_path, vecs)
    index.build_ivf(clusters=64)
    ranges = MmapVectorIndex._probe_ranges(vecs[0], index._ivf, index.ivf_count, index.count, cand=200, nprobe=4)
    scanned = sum(hi - lo for lo, hi in ranges)
    assert scanned < index.count * 0.5  # 4 из 40 кластеров — заметно меньше половины базы
    assert scanned >= 200  # кандидатов хватает для top-k с доранжированием


def test_delta_rows_after_build_are_found(tmp_path):
    vecs = _clustered(2500, 25, seed=6)
    index = _index_with(tmp_path, vecs)
    index.build_ivf(clusters=25)
    fresh = _clustered(50, 25, seed=7)
    index.append([_uid(2500 + i) for i in range(50)], fresh)
    assert index.ivf_count == 2500 and index.count == 2550
    top = index.search(fresh[13], 1, rerank_factor=20, nprobe=4)
    assert top[0].chunk_id == _uid(2500 + 13)


def test_ivf_survives_reopen_and_meta_is_backward_compatible(tmp_path):
    vecs = _clustered(1500, 15, seed=8)
    index = _index_with(tmp_path, vecs)
    index.build_ivf(clusters=16)
    again = MmapVectorIndex(tmp_path / "ivf", DIM)
    assert again.ivf_count == 1500 and again.ivf_clusters == 16
    assert again.search(vecs[5], 1, rerank_factor=20, nprobe=16)[0].chunk_id == _uid(5)

    # meta формата v2 без IVF-полей — читается, IVF считается не построенным
    meta = json.loads((tmp_path / "ivf" / META_FILE).read_text(encoding="utf-8"))
    meta.pop("ivf_count")
    meta.pop("ivf_clusters")
    (tmp_path / "ivf" / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    old = MmapVectorIndex(tmp_path / "ivf", DIM)
    assert old.count == 1500 and old.ivf_count == 0
    assert old.search(vecs[5], 1, rerank_factor=20)[0].chunk_id == _uid(5)


def test_ivf_needs_build_thresholds(tmp_path):
    index = MmapVectorIndex(tmp_path / "thr", DIM)
    assert not index.ivf_needs_build(min_vectors=100, rebuild_ratio=0.3)  # мало строк
    index.append([_uid(i) for i in range(200)], _clustered(200, 5, seed=9))
    assert index.ivf_needs_build(min_vectors=100, rebuild_ratio=0.3)  # IVF ещё не построен
    index.build_ivf(clusters=8)
    assert not index.ivf_needs_build(min_vectors=100, rebuild_ratio=0.3)
    index.append([_uid(200 + i) for i in range(70)], _clustered(70, 5, seed=10))  # дельта 35 %
    assert index.ivf_needs_build(min_vectors=100, rebuild_ratio=0.3)


def test_rebuild_clears_ivf(tmp_path):
    index = _index_with(tmp_path, _clustered(800, 8, seed=11), name="rb")
    index.build_ivf(clusters=8)
    index.rebuild()
    assert index.count == 0 and index.ivf_count == 0 and index._ivf is None


@pytest.mark.parametrize("clusters", [2, 5])
def test_small_cluster_counts(tmp_path, clusters):
    vecs = _clustered(60, 3, seed=12)
    index = _index_with(tmp_path, vecs, name=f"s{clusters}")
    assert index.build_ivf(clusters=clusters)["built"]
    assert index.search(vecs[7], 1, rerank_factor=20, nprobe=clusters)[0].chunk_id == _uid(7)


def test_ids_file_stores_16_byte_uuids(tmp_path):
    vecs = _clustered(300, 5, seed=13)
    index = _index_with(tmp_path, vecs, name="uuid16")
    assert (tmp_path / "uuid16" / "ids.bin").stat().st_size == 300 * 16
    assert top_id(index, vecs[42]) == _uid(42)


def top_id(index: MmapVectorIndex, q: np.ndarray) -> str:
    return index.search(q, 1, rerank_factor=20, nprobe=5)[0].chunk_id


def test_non_uuid_chunk_id_is_rejected(tmp_path):
    index = MmapVectorIndex(tmp_path / "bad", DIM)
    with pytest.raises(ValueError, match="UUID"):
        index.append(["not-a-uuid"], _clustered(1, 1, seed=14))
    assert index.count == 0


def test_index_survives_external_directory_removal(tmp_path):
    """Каталог индекса удалён извне (rebuild другого объекта): закэшированный объект сбрасывается, а не падает."""
    import shutil

    vecs = _clustered(40, 4, seed=15)
    index = _index_with(tmp_path, vecs, name="gone")
    assert index.count == 40
    shutil.rmtree(tmp_path / "gone")
    index.check_consistency()  # так делает sync_mmap_index перед чтением курсора
    assert index.count == 0 and index.cursor == 0
    assert index.search(vecs[0], 1, rerank_factor=20) == []
    # запись в удалённый каталог не падает (раньше — FileNotFoundError при записи meta)
    index.append([_uid(1)], vecs[1:2], cursor=1)
    assert index.count == 1
    index.append([_uid(0)], vecs[:1], cursor=1)  # и снова пишется с нуля
    assert index.search(vecs[0], 1, rerank_factor=20)[0].chunk_id == _uid(0)
