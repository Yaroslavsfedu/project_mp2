"""Нагрузочная проверка контура: массовая загрузка документов через API.

Скрипт поднимает N документов (демо-файлы под уникальными именами), ждёт
завершения обработки и считает пропускную способность, задержки и долю ошибок.
Работает против запущенного API — то есть проверяет весь стек целиком:
HTTP → очередь → OCR/LLM → БД → WebSocket-события.

Примеры:
    # 50 документов по 4 одновременно против локального API
    PYTHONPATH=. python -m scripts.load_test --count 50 --concurrency 4

    # 1000 документов против Docker-контура
    PYTHONPATH=. python -m scripts.load_test --count 1000 --concurrency 8 \
        --url http://localhost:8080/api/v1 --email demo@idp-project.ru

Замечания:
* для 1000+ документов используйте `QUEUE_BACKEND=celery` и GPU, иначе упрётесь в CPU;
* документы остаются в базе (имя начинается с `load_`), удалить их можно фильтром `q=load_`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]


async def login(client: httpx.AsyncClient, email: str, password: str) -> str:
    response = await client.post("/auth/login", json={"email": email, "password": password})
    if response.status_code != 200:
        response = await client.post(
            "/auth/register",
            json={"email": email, "password": password, "full_name": "Нагрузочный тест"},
        )
    response.raise_for_status()
    return response.json()["access_token"]


async def upload_batch(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    files: list[Path],
    start_index: int,
    semaphore: asyncio.Semaphore,
    results: dict,
) -> None:
    async with semaphore:
        chosen = files[start_index % len(files)]
        name = f"load_{start_index:04d}_{chosen.name}"
        payload = chosen.read_bytes()
        started = time.perf_counter()
        try:
            response = await client.post(
                "/documents/upload",
                headers=headers,
                files={"files": (name, payload, "application/octet-stream")},
            )
            response.raise_for_status()
            document_id = response.json()["documents"][0]["id"]
            results["uploaded"].append(time.perf_counter() - started)
            results["ids"].append(document_id)
        except httpx.HTTPError as exc:
            results["errors"].append(f"{name}: {exc}")


async def wait_processing(
    client: httpx.AsyncClient, headers: dict[str, str], document_ids: list[str], timeout: float
) -> dict[str, str]:
    statuses = {doc_id: "pending" for doc_id in document_ids}
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        pending = [doc_id for doc_id, status in statuses.items() if status in {"pending", "ocr_processing", "llm_processing"}]
        if not pending:
            break
        for doc_id in pending:
            try:
                response = await client.get(f"/documents/{doc_id}", headers=headers)
                if response.status_code == 200:
                    statuses[doc_id] = response.json()["status"]
            except httpx.HTTPError:
                pass
        await asyncio.sleep(1.0)
    return statuses


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * ratio), len(ordered) - 1)
    return ordered[index]


async def main() -> int:
    parser = argparse.ArgumentParser(description="Нагрузочный тест IDP-платформы")
    parser.add_argument("--url", default="http://localhost:8000/api/v1")
    parser.add_argument("--email", default="demo@idp-project.ru")
    parser.add_argument("--password", default="demo12345")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1800, help="ожидание обработки, секунд")
    parser.add_argument(
        "--samples",
        default=str(PROJECT_ROOT / "samples"),
        help="каталог с документами для загрузки",
    )
    parser.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "load_test.json"))
    args = parser.parse_args()

    files = sorted(
        path
        for path in Path(args.samples).iterdir()
        if path.suffix.lower() in {".pdf", ".docx", ".txt"} and not path.name.startswith("scan_")
    )
    if not files:
        print(f"Нет документов в {args.samples}")
        return 1

    results: dict = {"uploaded": [], "errors": [], "ids": []}
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(base_url=args.url, timeout=180.0) as client:
        token = await login(client, args.email, args.password)
        headers = {"Authorization": f"Bearer {token}"}

        started = time.perf_counter()
        await asyncio.gather(
            *(
                upload_batch(client, headers, files, index, semaphore, results)
                for index in range(args.count)
            )
        )
        upload_duration = time.perf_counter() - started

        print(f"Загружено: {len(results['ids'])} / {args.count} за {upload_duration:.1f} с")
        if results["errors"]:
            print(f"Ошибок загрузки: {len(results['errors'])}; первая: {results['errors'][0]}")

        processing_started = time.perf_counter()
        statuses = await wait_processing(client, headers, results["ids"], args.timeout)
        processing_duration = time.perf_counter() - processing_started
        done = sum(1 for status in statuses.values() if status in {"completed", "verified"})
        failed = sum(1 for status in statuses.values() if status == "failed")
        stuck = len(statuses) - done - failed

    uploads = results["uploaded"]
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "target": args.url,
        "requested": args.count,
        "uploaded": len(results["ids"]),
        "completed": done,
        "failed": failed,
        "stuck": stuck,
        "upload_errors": results["errors"][:10],
        "concurrency": args.concurrency,
        "upload_seconds": round(upload_duration, 2),
        "processing_seconds": round(processing_duration, 2),
        "upload_rps": round(len(results["ids"]) / upload_duration, 2) if upload_duration else None,
        "docs_per_minute": round(done / (processing_duration / 60), 2) if processing_duration and done else None,
        "upload_latency_ms": {
            "p50": round(percentile(uploads, 0.5) * 1000, 1),
            "p95": round(percentile(uploads, 0.95) * 1000, 1),
            "max": round(max(uploads) * 1000, 1) if uploads else None,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=== Итог ===")
    print(f"Обработано: {done} · ошибок: {failed} · в очереди: {stuck}")
    print(f"Скорость загрузки: {report['upload_rps']} док/с (p95 {report['upload_latency_ms']['p95']} мс)")
    print(f"Пропускная способность обработки: {report['docs_per_minute']} док/мин")
    print(f"Отчёт: {out}")

    return 0 if failed == 0 and stuck == 0 else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
