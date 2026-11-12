"""Итерация 5: бенчмарк трёх поколений архитектуры на 100k и 1M документов → reports/extreme_hpc_benchmark.json.

Корпус — тексты синтетических документов (тематические кластеры, Zipf) → эмбеддинги нашей функцией
(hash_embedding_batch, MD5 — как в конвейере). Эталон recall — ТОЧНЫЙ float32-косинус (top-100 на запрос).

Поколения:
  G1  базовая:   float32 brute force (полный перебор по файлу float32), SQLite LIKE для текста;
  G2  итерация 4: линейный POPCNT-скан по сигнатурам (без IVF, rerank 20, без раннего выхода), int8, FTS5 (JOIN);
  G3  итерация 5: IVF (K=256, nprobe 8) + FWHT-SimHash + ранний выход + одна куча + int8 (G3a) или int4 (G3b),
      FTS5 двухшаговый; индексация пакетная (OpenMP).

Метрики recall: strict — совпадение позиций с точным top-10; tolerant — результат засчитан, если его точный
косинус ≥ 10-го точного − 0.01 (строгий recall штрафует за порядок почти равных скоров, см. docs/REPORT).

Запуск: ``python -m scripts.extreme_benchmark [--sizes 100000,1000000] [--queries 30] [--out ...]``.
Каждый вариант поиска замеряется отдельным процессом (память — пик этого процесса, VmHWM).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from scripts import ultra_benchmark as ub

DIM = 1024
TOP = 10
TRUTH_K = 100
TOL_EPS = 0.01
REPORT = Path(__file__).resolve().parents[2] / "reports" / "extreme_hpc_benchmark.json"


def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))], 2)


# ------------------------------------------------------------------ корпус и эталон

def build_corpus(workdir: Path, n: int, queries: int) -> dict:
    """Генерирует n документов и queries запросов, строит float32-файл, индексы int8/int4 и эталон top-100."""
    from app.native.loader import HASH_MD5, get_kernels
    from app.services.mmap_index import MmapVectorIndex

    kernels = get_kernels()
    workdir.mkdir(parents=True, exist_ok=True)
    f32_path = workdir / f"vec_{n}.f32"
    q_texts = ub.synthetic_texts(queries, start=900_000_000)
    qvecs = kernels.hash_embedding_batch(q_texts, DIM, HASH_MD5).astype(np.float32)
    np.save(workdir / f"queries_{n}.npy", qvecs)

    idx8 = MmapVectorIndex(workdir / f"idx8_{n}", DIM, rerank_bits=8)
    idx4 = MmapVectorIndex(workdir / f"idx4_{n}", DIM, rerank_bits=4)
    best_s = np.full((queries, TRUTH_K), -9.0, dtype=np.float32)
    best_i = np.zeros((queries, TRUTH_K), dtype=np.int64)
    t_hash = t_index = 0.0
    step = 50_000
    with open(f32_path, "wb") as f32:
        for start in range(0, n, step):
            m = min(step, n - start)
            texts = ub.synthetic_texts(m, start=start)
            t0 = time.perf_counter()
            vecs = kernels.hash_embedding_batch(texts, DIM, HASH_MD5).astype(np.float32)
            t_hash += time.perf_counter() - t0
            f32.write(np.ascontiguousarray(vecs).tobytes())
            ids = [ub.doc_id(start + i) for i in range(m)]
            t0 = time.perf_counter()
            idx8.append(ids, vecs)
            idx4.append(ids, vecs)
            t_index += time.perf_counter() - t0
            sc = qvecs @ vecs.T
            cs = np.concatenate([best_s, sc], axis=1)
            ci = np.concatenate([best_i, np.tile(np.arange(start, start + m), (queries, 1))], axis=1)
            sel = np.argpartition(-cs, TRUTH_K, axis=1)[:, :TRUTH_K]
            best_s = np.take_along_axis(cs, sel, 1)
            best_i = np.take_along_axis(ci, sel, 1)
    truth = []
    for q in range(queries):
        order = np.argsort(-best_s[q])
        truth.append({"ids": best_i[q][order].tolist(), "scores": best_s[q][order].tolist()})
    (workdir / f"truth_{n}.json").write_text(json.dumps(truth))

    t0 = time.perf_counter()
    ivf8 = idx8.build_ivf(clusters=256)
    ivf8_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    ivf4 = idx4.build_ivf(clusters=256)
    ivf4_s = time.perf_counter() - t0
    return {
        "documents_or_vectors": n,
        "hash_embedding_md5_batch_s": round(t_hash, 1),
        "hash_docs_per_s": round(n / t_hash) if t_hash else None,
        "index_append_int8_int4_s": round(t_index, 1),
        "index_append_docs_per_s": round(n / t_index) if t_index else None,
        "ivf_build_int8_s": round(ivf8_s, 1),
        "ivf_build_int4_s": round(ivf4_s, 1),
        "ivf_int8": ivf8,
        "ivf_int4": ivf4,
        "disk_mb": {
            "float32_brute_force": round(f32_path.stat().st_size / 1e6, 1),
            "int8_index": round(idx8.disk_bytes() / 1e6, 1),
            "int4_index": round(idx4.disk_bytes() / 1e6, 1),
        },
    }


# ------------------------------------------------------------------ воркеры поиска (отдельный процесс)

def worker_search(args: argparse.Namespace) -> dict:
    workdir = Path(args.workdir)
    n = args.n
    qvecs = np.load(workdir / f"queries_{n}.npy")
    truth = json.loads((workdir / f"truth_{n}.json").read_text())
    lat: list[float] = []
    warm: list[float] = []
    strict: list[float] = []
    tol: list[float] = []

    f32 = np.memmap(workdir / f"vec_{n}.f32", dtype=np.float32, mode="r", shape=(n, DIM))

    def score_row(got_ids: list[int], q: int) -> None:
        """strict: совпадение позиций с точным top-10; tolerant: точный косинус кандидата ≥ 10-й точный − ε.
        Точный косинус берётся из float32-файла для каждого возвращённого кандидата (даже вне эталонного top-100)."""
        tr = truth[q]
        top_ids = set(tr["ids"][:TOP])
        kth = tr["scores"][TOP - 1]
        uniq = list(dict.fromkeys(got_ids))[:TOP]
        strict.append(len(top_ids & set(uniq)) / TOP)
        exact = np.asarray(f32[np.array(uniq, dtype=np.int64)]) @ qvecs[q]
        tol.append(int(np.sum(exact >= kth - TOL_EPS)) / TOP)

    if args.mode == "float32_brute":
        for q in range(len(qvecs)):
            t0 = time.perf_counter()
            best_s = np.empty(0, np.float32)
            best_i = np.empty(0, np.int64)
            for s in range(0, n, 65536):
                blk = np.asarray(f32[s:s + 65536])
                sc = blk @ qvecs[q]
                cs = np.concatenate([best_s, sc])
                ci = np.concatenate([best_i, np.arange(s, s + sc.size)])
                sel = np.argpartition(-cs, TOP)[:TOP]
                best_s, best_i = cs[sel], ci[sel]
            lat.append((time.perf_counter() - t0) * 1000)
            score_row([int(i) for i in best_i], q)
    else:
        from app.services.mmap_index import MmapVectorIndex

        rerank_bits = 4 if args.mode.endswith("int4") else 8
        idx = MmapVectorIndex(workdir / f"idx{rerank_bits}_{n}", DIM, rerank_bits=rerank_bits)
        if args.mode.startswith("linear"):
            idx.ivf_count = 0  # линейный скан: кластеры не используются
            early, rr = 0, 20
        else:
            early, rr = 4, 100
        # два прохода: первый — как есть (страницы могут быть холодными), второй — тёплый кэш
        for pass_no in (1, 2):
            for q in range(len(qvecs)):
                t0 = time.perf_counter()
                got = [int(c.chunk_id) for c in idx.search(qvecs[q], TOP, rr, nprobe=8, early_words=early)]
                ms = (time.perf_counter() - t0) * 1000
                if pass_no == 1:
                    lat.append(ms)
                    score_row(got, q)
                else:
                    warm.append(ms)
    return {
        "mode": args.mode, "n": n, "queries": len(qvecs),
        "p50_ms": _pct(lat, 50), "p95_ms": _pct(lat, 95),
        "warm_p50_ms": _pct(warm, 50) if warm else None, "warm_p95_ms": _pct(warm, 95) if warm else None,
        "strict_recall_at_10": round(float(np.mean(strict)), 3),
        "tolerant_recall_at_10": round(float(np.mean(tol)), 3),
        "peak_rss_mb": ub._mem_breakdown().get("peak_mb"),
        "memory": ub._mem_breakdown(),
    }


def run_worker(workdir: Path, n: int, mode: str) -> dict:
    cmd = [sys.executable, "-m", "scripts.extreme_benchmark", "worker", "--workdir", str(workdir),
           "--n", str(n), "--mode", mode]
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env,
                          cwd=str(Path(__file__).resolve().parents[1]), check=False)
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{")), None)
    if proc.returncode != 0 or line is None:
        raise RuntimeError(f"воркер {mode} упал: {proc.stderr[-800:]}")
    return json.loads(line)


# ------------------------------------------------------------------ FTS: LIKE / FTS5 / FTS5 двухшаговый

FTS_ROWS = 100_000


def bench_fts(workdir: Path, queries: int) -> dict:
    db_path = workdir / "fts_100k.db"
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE document_chunks (id TEXT PRIMARY KEY, document_id TEXT, page_number INT, chunk_text TEXT)")
    con.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, filename TEXT, doc_type TEXT, status TEXT, created_at TEXT)")
    for start in range(0, FTS_ROWS, 20_000):
        texts = ub.synthetic_texts(min(20_000, FTS_ROWS - start), start=start)
        con.executemany("INSERT INTO document_chunks VALUES (?,?,?,?)",
                        [(ub.doc_id(start + i), ub.doc_id((start + i) // 5), 1, t) for i, t in enumerate(texts)])
    con.executemany("INSERT INTO documents VALUES (?,?,?,?,?)",
                    [(ub.doc_id(i), f"f{i}.pdf", "Договор", "completed", "2026-01-01 00:00:00")
                     for i in range(FTS_ROWS // 5 + 1)])
    con.commit()
    con.execute("CREATE VIRTUAL TABLE chunk_fts USING fts5(chunk_text, content='document_chunks', "
                "content_rowid='rowid', tokenize='unicode61')")
    con.execute("INSERT INTO chunk_fts(chunk_fts) VALUES ('rebuild')")
    con.commit()

    vocab = ub.synthetic_vocab()
    terms = {"частый": vocab[0], "средний": vocab[50], "редкий_средний": vocab[400],
             "редкий": vocab[1500], "очень_редкий": vocab[3000]}
    like_sql = "SELECT id FROM document_chunks WHERE chunk_text LIKE ? ORDER BY rowid DESC LIMIT 300"
    joined_sql = ("SELECT c.id, snippet(chunk_fts, 0, '', '', '…', 24), bm25(chunk_fts) AS rank "
                  "FROM chunk_fts JOIN document_chunks c ON c.rowid = chunk_fts.rowid "
                  "JOIN documents d ON d.id = c.document_id "
                  "WHERE chunk_fts MATCH ? ORDER BY rank LIMIT 300")
    rank_sql = ("SELECT rowid AS rid, bm25(chunk_fts) AS rank FROM chunk_fts "
                "WHERE chunk_fts MATCH ? ORDER BY rank LIMIT 300")

    def two_step(stem: str):
        top = con.execute(rank_sql, (stem,)).fetchall()
        rids = [r[0] for r in top]
        ph = ",".join("?" * len(rids))
        texts = dict(con.execute(f"SELECT rowid, chunk_text FROM document_chunks WHERE rowid IN ({ph})",
                                 tuple(rids)).fetchall())
        return [(rid, texts.get(rid)) for rid, _ in top]

    out: dict = {"chunks": FTS_ROWS, "terms": {}}
    for label, term in terms.items():
        stem = term[:5]
        row: dict = {"word": term}
        for name, fn in (
            ("like_p95_ms", lambda: con.execute(like_sql, (f"%{stem}%",)).fetchall()),
            ("fts5_join_p95_ms", lambda: con.execute(joined_sql, (f'"{stem}"*',)).fetchall()),
            ("fts5_two_step_p95_ms", lambda: two_step(f'"{stem}"*')),
        ):
            fn()
            times = []
            for _ in range(queries):
                t0 = time.perf_counter()
                fn()
                times.append((time.perf_counter() - t0) * 1000)
            row[name] = _pct(times, 95)
        out["terms"][label] = row
    con.close()
    worst = {k: max(v[k] for v in out["terms"].values()) for k in ("like_p95_ms", "fts5_join_p95_ms", "fts5_two_step_p95_ms")}
    out["worst_p95_ms"] = {k: round(v, 2) for k, v in worst.items()}
    out["note_like"] = ("LIKE не ранжирует и отдаёт первые 300 совпадений по rowid DESC: его время — "
                        "стоимость скана, а не эквивалент bm25-выдачи; ранжирование сравнивать по FTS5-колонкам")
    out["note"] = ("частый терм совпадает почти со всеми чанками: bm25 считается по всем совпадениям — это "
                   "предел FTS5, не обходится двухшаговым путём; средние термы — прямой выигрыш")
    return out


# ------------------------------------------------------------------ хэширование и бюджет

def bench_hashing_variants(docs: int) -> dict:
    from app.ml_pipeline.text_utils import hashing_embedding
    from app.native.loader import HASH_MD5, HASH_WYHASH, get_kernels

    kernels = get_kernels()
    texts = ub.synthetic_texts(docs, start=7_000_000)

    def rate(fn, count):
        t0 = time.perf_counter()
        fn()
        return round(count / (time.perf_counter() - t0))

    py_n = min(2000, docs)
    res = {
        "backend": kernels.backend,
        "python_md5_docs_per_s": rate(lambda: [hashing_embedding(t, DIM) for t in texts[:py_n]], py_n),
        "c_md5_batch_docs_per_s": rate(lambda: kernels.hash_embedding_batch(texts, DIM, HASH_MD5), docs),
    }
    if kernels.native:
        res["c_wyhash_batch_docs_per_s"] = rate(lambda: kernels.hash_embedding_batch(texts, DIM, HASH_WYHASH), docs)
    return res


def budget_2m(linear_scan_ms_per_million: float | None, bytes_per_vector: dict) -> dict:
    vectors = 2_000_000 * 10
    gb = 1024 ** 3
    return {
        "documents": 2_000_000,
        "chunks_per_document": 10,
        "vectors": vectors,
        "signs_gb": round(vectors * 128 / gb, 2),
        "int4_gb": round(vectors * DIM // 2 / gb, 2),
        "int8_gb": round(vectors * DIM / gb, 2),
        "scales_gb": round(vectors * 4 / gb, 3),
        "ids_gb": round(vectors * 36 / gb, 2),
        "index_disk_int4_gb": round(vectors * (128 + 512 + 36 + 4) / gb, 1),
        "index_disk_int8_gb": round(vectors * (128 + 1024 + 36) / gb, 1),
        "ivf_probe_fraction_nprobe8_of_256": round(8 / 256, 4),
        "ivf_rows_scanned_per_query": int(vectors * 8 / 256),
        "ivf_signs_read_per_query_mb": round(vectors * 8 / 256 * 128 / 1e6, 1),
        "rerank_rows_per_query": 1000,
        "rerank_bytes_per_query_int4_kb": round(1000 * 512 / 1024, 1),
        "linear_scan_ms_per_million_measured": linear_scan_ms_per_million,
        "ivf_scan_ms_estimate_2m_theoretical": round(linear_scan_ms_per_million * 2 * 8 / 256, 1) if linear_scan_ms_per_million else None,
        "measured_bytes_per_vector": bytes_per_vector,
        "ram_note": "процесс поиска: RSS по замеру на 1M (см. sizes.1000000.search); сигнатуры 2.4 ГБ и int8/int4 читаются через mmap и страничный кэш ОС, при нехватке памяти ядро их вытесняет",
        "build_ivf_note": "K-Means + назначение + переукладка: время построения замерено на 100k и 1M (build.ivf_build_*_s)",
        "caveats": [
            "оценка по линейной экстраполяции измеренных величин, не замеренный прогон на 2M",
            "холодный кэш (диск) увеличит время: ~80 МБ сигнатур на запрос при nprobe 8 из 256",
            "2 vCPU: сборка индекса и хэширование масштабируются ограниченно",
        ],
    }


# ------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")
    wp = sub.add_parser("worker")
    wp.add_argument("--workdir", required=True)
    wp.add_argument("--n", type=int, required=True)
    wp.add_argument("--mode", required=True)
    parser.add_argument("--sizes", default="100000,1000000")
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--brute-queries", type=int, default=10)
    parser.add_argument("--workdir-root", default="/tmp/extreme-bench")
    parser.add_argument("--out", default=str(REPORT))
    args = parser.parse_args(argv)

    if args.cmd == "worker":
        print(json.dumps(worker_search(args), ensure_ascii=False))
        return 0

    root = Path(args.workdir_root)
    report: dict = {"environment": ub._environment(), "generations": {
        "G1": "float32 brute force (полный перебор файла float32) + SQLite LIKE",
        "G2": "итерация 4: линейный POPCNT-скан, rerank 20, без IVF/раннего выхода, FTS5 (JOIN)",
        "G3a": "итерация 5: IVF(256, nprobe 8) + FWHT-SimHash + ранний выход + одна куча + int8 rerank",
        "G3b": "итерация 5: как G3a, int4 rerank (512 Б/вектор)",
    }}
    report["hashing"] = bench_hashing_variants(50_000)
    report["sizes"] = {}
    linear_ms_per_million = None
    for n in (int(x) for x in args.sizes.split(",") if x.strip()):
        wd = root / f"n{n}"
        if wd.exists():
            shutil.rmtree(wd)
        build = build_corpus(wd, n, args.queries)
        entry = {"build": build, "search": {}}
        if n <= 200_000:
            entry["search"]["G1_float32_brute"] = run_worker(wd, n, "float32_brute")
        else:
            entry["search"]["G1_float32_brute"] = run_worker_limited(wd, n, args.brute_queries)
        entry["search"]["G2_linear_int8_rr20"] = run_worker(wd, n, "linear_int8")
        entry["search"]["G3a_ivf_int8"] = run_worker(wd, n, "ivf_int8")
        entry["search"]["G3b_ivf_int4"] = run_worker(wd, n, "ivf_int4")
        report["sizes"][str(n)] = entry
        g2 = entry["search"]["G2_linear_int8_rr20"]
        if n >= 1_000_000:
            linear_ms_per_million = g2["p95_ms"] * 1_000_000 / n
        shutil.rmtree(wd, ignore_errors=True)
    report["fts"] = bench_fts(root, queries=20)
    bpv = {}
    if "1000000" in report["sizes"]:
        d = report["sizes"]["1000000"]["build"]["disk_mb"]
        vecs = 1_000_000  # в корпусе 1M векторов (один вектор на документ)
        bpv = {"int8_index_per_vector_bytes": round(d["int8_index"] * 1e6 / vecs, 1),
               "int4_index_per_vector_bytes": round(d["int4_index"] * 1e6 / vecs, 1)}
    report["budget_2m"] = budget_2m(linear_ms_per_million, bpv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"written": args.out}, ensure_ascii=False))
    return 0


def run_worker_limited(workdir: Path, n: int, queries: int) -> dict:
    """Brute force на больших N: ограниченное число запросов (каждый запрос читает весь файл)."""
    tmp = workdir / f"qlimit_{queries}"
    tmp.mkdir(exist_ok=True)
    qv = np.load(workdir / f"queries_{n}.npy")[:queries]
    tr = json.loads((workdir / f"truth_{n}.json").read_text())[:queries]
    np.save(tmp / f"queries_{n}.npy", qv)
    (tmp / f"truth_{n}.json").write_text(json.dumps(tr))
    try:
        os.symlink(workdir / f"vec_{n}.f32", tmp / f"vec_{n}.f32")
    except FileExistsError:
        pass
    return run_worker(tmp, n, "float32_brute")


if __name__ == "__main__":
    raise SystemExit(main())
