"""Ultra-Low-Spec бенчмарк: до/после для хэширования, векторного поиска и FTS на SQLite.

Запуск (из backend/):
    python -m scripts.ultra_benchmark --sizes 100000,1000000 --queries 30 --out ../reports/ultra_opt_benchmark.json

Что сравнивается:
  1. Хэширование текста: Python ``hashing_embedding`` против C ``fast_hashing_embedding`` (docs/сек).
  2. Векторный поиск на тех же синтетических документах:
       • «стандарт» — float32 (нормированные вектора, полный перебор с NumPy; для 1M — из memmap-файла);
       • «ultra»    — mmap-индекс: 1-бит POPCNT-скан Хэмминга + int8-доранжирование (C).
     Замеряются задержка, RSS процесса (отдельный процесс на вариант) и recall@10 относительно float32.
  3. FTS на SQLite: старый LIKE по тексту против FTS5 (bm25/snippet) на 100 000 чанков.
  4. Бюджет RAM/диска на 2 000 000 документов (20 млн векторов) по измеренным байтам.

Артефакты (float32, индекс, база) пишутся в --workdir (по умолчанию /tmp/idp-ultra), в Git не попадают.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sqlite3
import statistics
import subprocess
import uuid
import sys
import time
from pathlib import Path

import numpy as np

DIM = 1024
ENGINE = "fallback"
REPO_BACKEND = Path(__file__).resolve().parents[1]
DEFAULT_WORKDIR = Path("/tmp/idp-ultra")

# ------------------------------------------------------------------ данные

_VOCAB_CACHE: list[str] | None = None


def synthetic_vocab(size: int = 4000, seed: int = 1) -> list[str]:
    """Псевдо-слова из кириллицы и латиницы: достаточно разнообразно, чтобы хэши различались."""
    rng = np.random.default_rng(seed)
    letters = list("абвгдежзиклмнопрстуфхцчшщыэюя") + list("abcdefghijklmnopqrstuvwxyz") + list("0123456789")
    words = set()
    while len(words) < size:
        n = int(rng.integers(3, 11))
        words.add("".join(letters[int(i)] for i in rng.integers(0, len(letters), size=n)))
    return sorted(words)


TOPICS = 300  # тематические кластеры (договоры, акты, счета…): у каждого свой словарь
TOPIC_WORDS = 150
TOPIC_SHARE = 0.7  # доля слов документа из словаря его темы; остальное — общая лексика


def synthetic_texts(n: int, start: int, words_per_doc: int = 40) -> list[str]:
    """Документы с тематической структурой и Zipf-распределением слов.

    Без тем все документы почти одинаково похожи друг на друга, и recall теряет смысл (проверено:
    top-10 и top-100 косинусов отличались на 0.01). С темами соседи реальны, как у деловых документов.
    """
    global _VOCAB_CACHE
    if _VOCAB_CACHE is None:
        _VOCAB_CACHE = synthetic_vocab()
    vocab = np.array(_VOCAB_CACHE)
    size = len(vocab)
    ranks = np.arange(1, size + 1, dtype=np.float64)
    global_p = 1.0 / ranks**1.07
    global_p /= global_p.sum()
    topic_rng = np.random.default_rng(7)
    topic_words = topic_rng.integers(0, size, size=(TOPICS, TOPIC_WORDS))
    local_p = 1.0 / np.arange(1, TOPIC_WORDS + 1)
    local_p /= local_p.sum()
    rng = np.random.default_rng(10_000 + start)
    topic = rng.integers(0, TOPICS, size=n)
    from_topic = rng.random((n, words_per_doc)) < TOPIC_SHARE
    topical = topic_words[topic[:, None], rng.choice(TOPIC_WORDS, size=(n, words_per_doc), p=local_p)]
    general = rng.choice(size, size=(n, words_per_doc), p=global_p)
    ids = np.where(from_topic, topical, general)
    return [" ".join(row) for row in vocab[ids]]


def doc_id(i: int) -> str:
    return str(uuid.UUID(int=i))  # UUID: ids.bin хранит 16 байт (формат v4)


# ------------------------------------------------------------------ этап 1: хэширование

def bench_hashing(docs: int) -> dict:
    from app.ml_pipeline.text_utils import hashing_embedding  # noqa: PLC0415
    from app.native.loader import get_kernels  # noqa: PLC0415

    kernels = get_kernels()
    texts = synthetic_texts(docs, start=77)
    sample = texts[:200]
    max_diff = 0.0
    for text in sample[:50]:
        py = np.asarray(hashing_embedding(text, DIM), dtype=np.float32)
        max_diff = max(max_diff, float(np.max(np.abs(py - kernels.hash_embedding(text, DIM)))))

    t0 = time.perf_counter()
    for text in texts:
        hashing_embedding(text, DIM)
    py_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for text in texts:
        kernels.hash_embedding(text, DIM)
    c_s = time.perf_counter() - t0

    return {
        "docs": docs,
        "words_per_doc": 40,
        "python_docs_per_s": round(docs / py_s, 1),
        "c_docs_per_s": round(docs / c_s, 1),
        "speedup": round(py_s / c_s, 2),
        "max_abs_diff_vs_python": max_diff,
        "backend": kernels.backend,
        "note": "синтетические документы по 40 слов (~250 символов); для чанков по 900 символов ускорение выше",
    }


# ------------------------------------------------------------------ этап 2: сборка данных

def build_artifacts(workdir: Path, n: int, queries: int) -> dict:
    from app.native.loader import get_kernels  # noqa: PLC0415
    from app.services.mmap_index import MmapVectorIndex  # noqa: PLC0415

    kernels = get_kernels()
    base = workdir / f"n{n}"
    base.mkdir(parents=True, exist_ok=True)
    f32_path = base / "vectors.f32"
    index_dir = base / "mmap_index"
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index = MmapVectorIndex(index_dir, DIM)

    t0 = time.perf_counter()
    batch = 50_000
    with open(f32_path, "wb") as f32:
        for start in range(0, n, batch):
            count = min(batch, n - start)
            texts = synthetic_texts(count, start=start)
            vecs = np.empty((count, DIM), dtype=np.float32)
            for row, text in enumerate(texts):
                vecs[row] = kernels.hash_embedding(text, DIM)
            vecs.tofile(f32)  # нормированные вектора: косинус = скалярное произведение
            index.append([doc_id(start + row) for row in range(count)], vecs, cursor=start + count)
    build_s = time.perf_counter() - t0

    # набор запросов: случайные документы + небольшой шум (как поиск похожих)
    rng = np.random.default_rng(42)
    picks = rng.integers(0, n, size=queries)
    qvecs = np.empty((queries, DIM), dtype=np.float32)
    f32 = np.memmap(f32_path, dtype=np.float32, mode="r", shape=(n, DIM))
    for i, p in enumerate(picks):
        # шум нормы ~0.16 (0.005 на компоненту): запрос — почти тот же документ
        v = np.asarray(f32[p]) + rng.standard_normal(DIM).astype(np.float32) * 0.005
        qvecs[i] = v / np.linalg.norm(v)
    del f32
    np.save(base / "queries.npy", qvecs)
    return {
        "documents": n,
        "build_seconds": round(build_s, 1),
        "float32_bytes": n * DIM * 4,
        "mmap_index_bytes": index.disk_bytes(),
        "signs_bytes": n * 128,
        "int8_bytes": n * DIM,
        "ids_bytes": n * 36,
        "base_dir": str(base),
    }


# ------------------------------------------------------------------ этап 3: рабочие процессы (отдельный процесс = честный RSS)

def _mem_breakdown() -> dict:
    """Память процесса по /proc/self/status: пик, собственная куча (RssAnon) и отображённые файлы (RssFile).
    Файловые страницы mmap ОС может вытеснить, поэтому «куча» — честная цифра для собственного расхода."""
    out = {"peak_mb": None, "heap_anon_mb": None, "mapped_file_mb": None}
    try:
        names = {"VmHWM": "peak_mb", "RssAnon": "heap_anon_mb", "RssFile": "mapped_file_mb"}
        for line in Path("/proc/self/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in names:
                out[names[key]] = round(int(value.split()[0]) / 1024.0, 1)
    except OSError:
        pass
    return out


def _rss_mb() -> float:
    """Пиковый RSS именно этого процесса (VmHWM). ru_maxrss здесь не годится: дочерний процесс после
    vfork/exec наследует пик родителя (тот строит индекс), и замер получается завышенным."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        pass
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def worker_float(base: Path, n: int, limit: int, k: int = 10) -> dict:
    queries = np.load(base / "queries.npy")[:limit]
    vec_file = base / "vectors.f32"
    # в RAM целиком только если помещается; иначе — memmap с потоковым перебором
    fits = n * DIM * 4 <= 1.5 * 1024**3
    if fits:
        mat = np.fromfile(vec_file, dtype=np.float32).reshape(n, DIM)
    else:
        mat = np.memmap(vec_file, dtype=np.float32, mode="r", shape=(n, DIM))
    chunk = 16_384
    latencies, tops = [], []
    for qi, q in enumerate(queries):
        t0 = time.perf_counter()
        best_s = np.empty(0, np.float32)
        best_i = np.empty(0, np.int64)
        for start in range(0, n, chunk):
            block = np.asarray(mat[start:start + chunk])
            s = block @ q
            kk = min(k, s.size)
            top = np.argpartition(-s, kk - 1)[:kk]
            best_s = np.concatenate([best_s, s[top]])
            best_i = np.concatenate([best_i, top.astype(np.int64) + start])
            order = np.argsort(-best_s)[:k]
            best_s, best_i = best_s[order], best_i[order]
        latencies.append((time.perf_counter() - t0) * 1000)
        tops.append([int(x) for x in best_i])
    return {"latencies_ms": latencies, "tops": tops, "rss_mb": _rss_mb(), "memory": _mem_breakdown(),
            "mode": "float32 in RAM" if fits else "float32 memmap (потоковый перебор)"}


def worker_mmap(base: Path, n: int, limit: int, k: int = 10) -> dict:
    from app.services.mmap_index import MmapVectorIndex  # noqa: PLC0415

    queries = np.load(base / "queries.npy")[:limit]
    index = MmapVectorIndex(base / "mmap_index", DIM)
    assert index.count == n, (index.count, n)
    latencies, tops = [], []
    for q in queries[:3]:  # прогрев страничного кэша
        index.search(q, k, 20)
    for q in queries:
        t0 = time.perf_counter()
        cands = index.search(q, k, 20)
        latencies.append((time.perf_counter() - t0) * 1000)
        tops.append([uuid.UUID(c.chunk_id).int for c in cands])
    return {"latencies_ms": latencies, "tops": tops, "rss_mb": _rss_mb(), "memory": _mem_breakdown(),
            "mode": "mmap 1-bit POPCNT + int8 rerank"}


def _run_worker(args: list[str]) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_BACKEND)
    proc = subprocess.run([sys.executable, "-m", "scripts.ultra_benchmark", *args], cwd=REPO_BACKEND,
                          capture_output=True, text=True, env=env, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _pct(values: list[float], p: float) -> float:
    return round(float(np.percentile(values, p)), 2)


def bench_vectors(workdir: Path, n: int, queries: int) -> dict:
    build = build_artifacts(workdir, n, queries)
    base = Path(build["base_dir"])
    # полный перебор float32 с диска на 1M занимает минуты: для больших размеров берём подвыборку запросов
    limit = queries if n <= 200_000 else min(queries, 8)
    flt = _run_worker(["--worker", "float", "--base", str(base), "--n", str(n), "--limit", str(limit)])
    ultra = _run_worker(["--worker", "mmap", "--base", str(base), "--n", str(n), "--limit", str(limit)])
    recalls = [len(set(a) & set(b)) / 10 for a, b in zip(flt["tops"], ultra["tops"], strict=True)]
    return {
        "documents": n,
        "chunks": n,
        "build": build,
        "standard_float32": {
            "mode": flt["mode"],
            "p50_ms": _pct(flt["latencies_ms"], 50),
            "p95_ms": _pct(flt["latencies_ms"], 95),
            "rss_mb": flt["rss_mb"],
            "memory": flt["memory"],
            "ram_needed_mb_if_loaded": round(n * DIM * 4 / 1024**2, 1),
        },
        "ultra_mmap_binary": {
            "mode": ultra["mode"],
            "p50_ms": _pct(ultra["latencies_ms"], 50),
            "p95_ms": _pct(ultra["latencies_ms"], 95),
            "rss_mb": ultra["rss_mb"],
            "memory": ultra["memory"],
            "signs_mb": round(build["signs_bytes"] / 1024**2, 1),
            "int8_mb_on_disk": round(build["int8_bytes"] / 1024**2, 1),
        },
        "recall_at_10_vs_float32": round(float(statistics.mean(recalls)), 3),
        "recall_min": round(float(min(recalls)), 3),
        "queries": limit,
    }


# ------------------------------------------------------------------ этап 4: FTS на SQLite

def bench_fts(workdir: Path, n: int, queries: int) -> dict:
    db_path = workdir / f"fts_n{n}.db"
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("CREATE TABLE document_chunks (id TEXT PRIMARY KEY, document_id TEXT, page_number INT, chunk_text TEXT)")
    con.execute("CREATE TABLE ocr_pages (id TEXT PRIMARY KEY, document_id TEXT, page_number INT, raw_text TEXT)")
    step = 20_000
    for start in range(0, n, step):
        texts = synthetic_texts(min(step, n - start), start=start)
        rows = [(doc_id(start + i), doc_id((start + i) // 5), 1, t) for i, t in enumerate(texts)]
        con.executemany("INSERT INTO document_chunks VALUES (?,?,?,?)", rows)
        con.executemany("INSERT INTO ocr_pages VALUES (?,?,?,?)", [(r[0], r[1], 1, r[3]) for r in rows])
    con.commit()

    fts_ok = True
    try:
        con.execute("CREATE VIRTUAL TABLE chunk_fts USING fts5(chunk_text, content='document_chunks', "
                    "content_rowid='rowid', tokenize='unicode61')")
        con.execute("INSERT INTO chunk_fts(chunk_fts) VALUES ('rebuild')")
        con.commit()
    except sqlite3.OperationalError:
        fts_ok = False

    vocab = synthetic_vocab()
    # термины: частое (высокий Zipf-ранг) и средней частоты (≈1%) слова
    terms = [vocab[0], vocab[50], vocab[400], vocab[1500], vocab[3000]]
    results = {}
    for term in terms:
        stem = term[:5]
        like_sql = "SELECT id FROM ocr_pages WHERE raw_text LIKE ? ORDER BY rowid DESC LIMIT 300"
        fts_sql = ("SELECT c.id, snippet(chunk_fts, 0, '', '', '…', 24), bm25(chunk_fts) AS rank "
                   "FROM chunk_fts JOIN document_chunks c ON c.rowid = chunk_fts.rowid "
                   "WHERE chunk_fts MATCH ? ORDER BY rank LIMIT 300")
        like_t = _time_query(con, like_sql, (f"%{stem}%",), queries)
        entry = {"like_p50_ms": like_t["p50"], "like_p95_ms": like_t["p95"], "like_hits": like_t["rows"]}
        if fts_ok:
            fts_t = _time_query(con, fts_sql, (f'"{stem}"*',), queries)
            entry.update({"fts5_p50_ms": fts_t["p50"], "fts5_p95_ms": fts_t["p95"], "fts5_hits": fts_t["rows"]})
        results[term] = entry
    con.close()
    p95_like = [v["like_p95_ms"] for v in results.values()]
    p95_fts = [v.get("fts5_p95_ms", float("nan")) for v in results.values()]
    return {
        "chunks": n,
        "fts5_available": fts_ok,
        "terms": results,
        "like_worst_p95_ms": round(max(p95_like), 2),
        "fts5_worst_p95_ms": round(max(p95_fts), 2) if fts_ok else None,
        "note": "LIKE ищет подстроку по всему тексту (как прежний путь); FTS5 — токены по префиксу стема",
    }


def _time_query(con: sqlite3.Connection, sql: str, params: tuple, repeats: int) -> dict:
    rows = 0
    for _ in range(2):
        rows = len(con.execute(sql, params).fetchall())
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        con.execute(sql, params).fetchall()
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50": _pct(times, 50), "p95": _pct(times, 95), "rows": rows}


# ------------------------------------------------------------------ этап 5: бюджет на 2M

def budget_2m(measured_bytes_per_vector_sig: float, measured_int8: float, scan_ms_per_million: float | None) -> dict:
    vectors = 2_000_000 * 10  # 10 чанков на документ
    gb = 1024**3
    return {
        "documents": 2_000_000,
        "chunks_per_document": 10,
        "vectors": vectors,
        "sign_index_gb": round(vectors * 128 / gb, 2),
        "sign_index_gb_decimal": round(vectors * 128 / 1e9, 2),
        "int8_on_disk_gb": round(vectors * DIM / gb, 1),
        "ids_gb": round(vectors * 36 / gb, 2),
        "hnsw_float32_gb_for_reference": round(vectors * 7946 / gb, 1),
        "ram_for_signs_if_fully_cached_gb": round(vectors * 128 / gb, 2),
        "process_rss_target_gb": 0.5,
        "scan_seconds_per_query_estimate": (
            round(scan_ms_per_million * vectors / 1e6 / 1000, 2) if scan_ms_per_million else None
        ),
        "measured_bytes_per_vector_sign": measured_bytes_per_vector_sig,
        "measured_bytes_per_vector_int8": measured_int8,
        "notes": [
            "RAM: сигнатуры 2.4 ГБ читаются из страничного кэша ОС; ядро может их вытеснить, процесс не держит копию.",
            "int8 (20 ГБ на диске) трогается только для кандидатов доранжирования: сотни строк на запрос.",
            "Полный POPCNT-скан читает все сигнатуры на каждый запрос: время растёт линейно с числом векторов.",
            "Оценка времени — линейная экстраполяция по измеренному скану; на 2M без индекса по кластерам "
            "поиск может не уложиться в 200 мс на 2 ядрах.",
        ],
    }


# ------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default="100000,1000000", help="размеры векторного теста, через запятую")
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--fts-size", type=int, default=100_000)
    parser.add_argument("--hash-docs", type=int, default=3000)
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    parser.add_argument("--out", default=str(REPO_BACKEND.parent / "reports" / "ultra_opt_benchmark.json"))
    parser.add_argument("--keep", action="store_true", help="не удалять артефакты после замера")
    parser.add_argument("--worker", choices=["float", "mmap"])
    parser.add_argument("--base")
    parser.add_argument("--n", type=int)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    if args.worker:  # внутренний режим: отдельный процесс для замера RSS
        base = Path(args.base)
        limit = args.limit or 10**9
        result = worker_float(base, args.n, limit) if args.worker == "float" else worker_mmap(base, args.n, limit)
        print(json.dumps(result))
        return 0

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    started = time.time()
    report: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": _environment(),
        "hashing": None,
        "vectors": [],
        "fts_sqlite": None,
        "budget_2m": None,
        "limitations": [
            "синтетические документы: слова из псевдо-словаря с Zipf-распределением; качество выдачи не оценивается",
            "запросы — документы с шумом; recall измерен относительно полного перебора float32 на тех же данных",
            "RSS измерен отдельным процессом (ru_maxrss); страницы mmap входят в RSS, но могут быть вытеснены ОС",
            "float32 при 1M не помещается в RAM песочницы (3.9 ГБ): замер через memmap, RSS и задержка — для этого режима",
        ],
    }
    from app.native.loader import get_kernels  # noqa: PLC0415

    print(f"backend: {get_kernels().backend}", flush=True)
    print("==> 1. хэширование текста", flush=True)
    report["hashing"] = bench_hashing(args.hash_docs)
    print(json.dumps(report["hashing"], ensure_ascii=False), flush=True)

    for n in sizes:
        print(f"==> 2. векторный поиск, N={n}", flush=True)
        vec = bench_vectors(workdir, n, args.queries)
        report["vectors"].append(vec)
        print(json.dumps({k: vec[k] for k in ("standard_float32", "ultra_mmap_binary",
                                              "recall_at_10_vs_float32")}, ensure_ascii=False), flush=True)

    print("==> 3. FTS на SQLite", flush=True)
    report["fts_sqlite"] = bench_fts(workdir, args.fts_size, max(args.queries, 20))
    print(json.dumps({k: report["fts_sqlite"][k] for k in ("like_worst_p95_ms", "fts5_worst_p95_ms")}), flush=True)

    biggest = report["vectors"][-1]
    per_vec_sig = biggest["build"]["signs_bytes"] / biggest["documents"]
    per_vec_i8 = biggest["build"]["int8_bytes"] / biggest["documents"]
    scan_ms = biggest["ultra_mmap_binary"]["p50_ms"] / biggest["documents"] * 1e6
    report["budget_2m"] = budget_2m(per_vec_sig, per_vec_i8, scan_ms)

    report["targets"] = _targets(report)
    report["elapsed_seconds"] = round(time.time() - started, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"==> отчёт: {out}", flush=True)

    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


def _targets(report: dict) -> dict:
    hashing = report["hashing"]
    big = report["vectors"][-1]
    fts = report["fts_sqlite"]
    return {
        "hash_speedup_target_10x": {
            "measured": hashing["speedup"], "passed": hashing["speedup"] >= 10.0,
        },
        "recall_at_10_target_0.7": {
            "measured": big["recall_at_10_vs_float32"], "passed": big["recall_at_10_vs_float32"] >= 0.7,
        },
        "rss_under_1gb_for_ultra": {
            "measured_mb": big["ultra_mmap_binary"]["rss_mb"], "passed": big["ultra_mmap_binary"]["rss_mb"] < 1024,
        },
        "fts5_faster_than_like": {
            "like_worst_p95_ms": fts["like_worst_p95_ms"],
            "fts5_worst_p95_ms": fts["fts5_worst_p95_ms"],
            "passed": bool(fts["fts5_available"] and fts["fts5_worst_p95_ms"] < fts["like_worst_p95_ms"]),
        },
        "search_under_200ms_on_largest": {
            "measured_p95_ms": big["ultra_mmap_binary"]["p95_ms"], "documents": big["documents"],
            "passed": big["ultra_mmap_binary"]["p95_ms"] < 200,
        },
    }


def _environment() -> dict:
    import platform  # noqa: PLC0415

    cpu_flags = ""
    try:
        cpu_flags = next(line for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("flags"))
    except (OSError, StopIteration):
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "avx2": " avx2" in cpu_flags,
        "popcnt": " popcnt" in cpu_flags,
        "ram_total_mb": _meminfo_mb("MemTotal"),
        "note": "песочница: 2 vCPU, ~3.9 ГБ RAM; целевой класс — ноутбук 2 ядра / 4–8 ГБ",
    }


def _meminfo_mb(key: str) -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(key + ":"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        return None
    return None


if __name__ == "__main__":
    sys.exit(main())
