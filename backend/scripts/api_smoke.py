"""Сквозная проверка API: маршруты, роли, владение, негативные сценарии, граничные значения, пагинация.

Запуск (backend запущен, RATE_LIMIT_ENABLED=false — сценарий делает больше попыток входа, чем лимит 10/мин):
    make smoke                    # по умолчанию http://127.0.0.1:8000
    python -m scripts.api_smoke http://127.0.0.1:8010

Сценарий создаёт тестовых пользователей и документы в текущей БД (с уникальными e-mail),
временный документ удаляет сам; демо-документы остаются.
"""
import os
import sys, time, uuid
from pathlib import Path

import httpx

BASE = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("IDP_API_URL", "http://127.0.0.1:8000")).rstrip("/")
SAMPLES = Path(__file__).resolve().parents[2] / "samples"
api = BASE + "/api/v1"
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (("  -- " + str(detail)[:300]) if (detail and not cond) else ""), flush=True)


c = httpx.Client(timeout=120)

# --- служебные ---
r = c.get(BASE + "/api/docs"); check("swagger /api/docs 200", r.status_code == 200, r.status_code)
r = c.get(api + "/health"); check("health 200", r.status_code == 200, r.text)
r = c.get(api + "/system/status"); check("system/status без токена -> 401", r.status_code == 401, r.status_code)

# --- вход демо-админа ---
r = c.post(api + "/auth/login", json={"email": "demo@idp-project.ru", "password": "demo12345"})
check("login демо-админа 200", r.status_code == 200, r.text)
ADMIN = {"Authorization": "Bearer " + r.json()["access_token"]}
r = c.post(api + "/auth/login", json={"email": "demo@idp-project.ru", "password": "неверный"})
check("login с неверным паролем 401", r.status_code == 401, r.status_code)
r = c.post(api + "/auth/login", json={"email": "nobody@idp-project.ru", "password": "xxxxxx"})
check("login несуществующего 401 (не 500)", r.status_code == 401, r.status_code)
r = c.get(api + "/auth/me", headers=ADMIN); check("me admin", r.status_code == 200 and r.json()["role"] == "admin", r.text)
r = c.get(api + "/auth/me", headers={"Authorization": "Bearer junk.token.value"}); check("me с мусорным токеном 401", r.status_code == 401, r.status_code)

# --- регистрация и валидация ---
uid = uuid.uuid4().hex[:8]
OP_EMAIL, VW_EMAIL = f"op-{uid}@example.ru", f"vw-{uid}@example.ru"
r = c.post(api + "/auth/register", json={"email": OP_EMAIL, "password": "секрет123", "full_name": "Оператор"})
check("register 201/200", r.status_code in (200, 201), r.text)
r = c.post(api + "/auth/register", json={"email": OP_EMAIL, "password": "секрет123"})
check("register дубликат 400/409", r.status_code in (400, 409), r.status_code)
r = c.post(api + "/auth/register", json={"email": "не-почта", "password": "секрет123"})
check("register невалидный email 422", r.status_code == 422, r.status_code)
r = c.post(api + "/auth/register", json={"email": f"s-{uid}@example.ru", "password": "123"})
check("register короткий пароль 422", r.status_code == 422, r.status_code)
r = c.post(api + "/auth/register", json={"email": f"role-{uid}@example.ru", "password": "секрет123", "role": "admin"})
check("register не назначает роль admin", r.status_code in (200, 201) and r.json()["user"]["role"] != "admin", r.text)
r = c.post(api + "/auth/login", json={"email": OP_EMAIL, "password": "секрет123"})
OP = {"Authorization": "Bearer " + r.json()["access_token"]}
r = c.post(api + "/auth/register", json={"email": VW_EMAIL, "password": "секрет123", "full_name": "Наблюдатель"})
vw_id = r.json()["user"]["id"]
r = c.patch(f"{api}/admin/users/{vw_id}", json={"role": "viewer"}, headers=ADMIN)
check("admin меняет роль на viewer", r.status_code == 200, r.text)
r = c.post(api + "/auth/login", json={"email": VW_EMAIL, "password": "секрет123"})
VW = {"Authorization": "Bearer " + r.json()["access_token"]}

# --- RBAC админки ---
r = c.get(api + "/admin/users", headers=OP); check("operator не видит admin/users (403)", r.status_code == 403, r.status_code)
r = c.get(api + "/admin/users", headers=ADMIN); check("admin видит пользователей", r.status_code == 200, r.status_code)
r = c.get(api + "/admin/audit", headers=ADMIN); check("admin audit 200", r.status_code == 200, r.status_code)
r = c.get(api + "/admin/overview", headers=ADMIN); check("admin overview 200", r.status_code == 200, r.status_code)
r = c.get(api + "/admin/quality", headers=ADMIN); check("admin quality 200", r.status_code == 200, r.status_code)
r = c.get(api + "/admin/quality", headers=OP); check("operator quality 403", r.status_code == 403, r.status_code)
r = c.patch(f"{api}/admin/users/{uuid.uuid4()}", json={"role": "viewer"}, headers=ADMIN)
check("patch несуществующего пользователя 404", r.status_code == 404, r.status_code)
r = c.patch(f"{api}/admin/users/{vw_id}", json={"role": "boss"}, headers=ADMIN)
check("patch недопустимой роли 422", r.status_code == 422, r.status_code)

# --- загрузка ---
names = ["akt_15.pdf", "dogovor_postavki_47-2024.pdf", "prikaz_112-OD.pdf", "schet_208.pdf",
         "nauchnaya_statya_IDP.txt", "sluzhebnaya_zapiska.docx", "scan_akt_15.jpg"]
files = {n: open(f"{SAMPLES}/{n}", "rb").read() for n in names}
payload = [("files", (n, b, "application/octet-stream")) for n, b in files.items()]
r = c.post(api + "/documents/upload", files=payload, headers=OP)
check("upload 7 файлов 201", r.status_code == 201, r.text[:300])
docs = r.json()["documents"] if r.status_code == 201 else []
check("upload вернул 7 документов", len(docs) == 7, len(docs))
r = c.post(api + "/documents/upload", files=[("files", ("bad.exe", b"MZ\x90\x00", "application/octet-stream"))], headers=OP)
check("upload неподдерживаемого типа отклонён (415)", r.status_code == 415, r.status_code)
r = c.post(api + "/documents/upload", files=[("files", ("empty.txt", b"", "text/plain"))], headers=OP)
check("upload пустого файла отклонён", r.status_code in (400, 422), r.status_code)
r = c.post(api + "/documents/upload", files=[("files", ("fake.pdf", b"not a pdf at all", "application/pdf"))], headers=OP)
check("upload PDF с подменённой сигнатурой отклонён", r.status_code == 400, r.status_code)
r = c.post(api + "/documents/upload", files=[("files", ("big.txt", b"a" * (51 * 1024 * 1024), "text/plain"))], headers=OP)
check("upload >50 МБ отклонён (413/400)", r.status_code in (400, 413), r.status_code)
r = c.post(api + "/documents/upload", files=[("files", ("x.txt", b"hi", "text/plain"))], headers=VW)
check("VIEWER не может загружать (403)", r.status_code == 403, f"получен {r.status_code}: {r.text[:120]}")

# --- ожидание обработки ---
def wait_done(doc_id, timeout=240):
    t0 = time.time()
    d = {}
    while time.time() - t0 < timeout:
        d = c.get(f"{api}/documents/{doc_id}", headers=ADMIN).json()
        if d["status"] in ("completed", "failed"):
            return d
        time.sleep(1)
    return d


final = {}
for d in docs:
    final[d["filename"]] = wait_done(d["id"])
    print("      ", d["filename"], "->", final[d["filename"]]["status"], "|", final[d["filename"]].get("error_message"), flush=True)
ok_names = [n for n, d in final.items() if d["status"] == "completed"]
text_names = [n for n in final if not n.startswith("scan_")]
check("текстовые документы обработаны (completed)", all(final[n]["status"] == "completed" for n in text_names), ok_names)
scan = final.get("scan_akt_15.jpg")
check("скан: completed (есть OCR) или failed с понятным сообщением",
      bool(scan and (scan["status"] == "completed" or (scan["status"] == "failed" and scan.get("error_message")))),
      scan and scan.get("status"))

DOC = next(d for d in docs if d["filename"] == "dogovor_postavki_47-2024.pdf")["id"]
r = c.get(f"{api}/documents/{DOC}", headers=VW); check("VIEWER читает карточку", r.status_code == 200, r.status_code)
r = c.get(f"{api}/documents/{uuid.uuid4()}", headers=ADMIN); check("несуществующий документ 404", r.status_code == 404, r.status_code)
r = c.get(f"{api}/documents/not-a-uuid", headers=ADMIN); check("невалидный id не даёт 500", r.status_code in (404, 422), r.status_code)
det = c.get(f"{api}/documents/{DOC}", headers=ADMIN).json()
check("карточка содержит страницы и сущности", bool(det["pages"]) and det["entities"] is not None, list(det.keys()))

# --- список: page/limit и курсор ---
r = c.get(api + "/documents", params={"page": 1, "limit": 3}, headers=ADMIN)
check("список page/limit", r.status_code == 200 and len(r.json()["items"]) == 3, r.status_code)
r = c.get(api + "/documents", params={"page": 1, "limit": 3, "status": "completed"}, headers=ADMIN)
check("фильтр status", r.status_code == 200 and all(i["status"] == "completed" for i in r.json()["items"]), r.status_code)
r = c.get(api + "/documents", params={"limit": 101}, headers=ADMIN)
check("limit>100 отклонён 422", r.status_code == 422, r.status_code)
r = c.get(api + "/documents", params={"page": 0}, headers=ADMIN)
check("page=0 отклонён 422", r.status_code == 422, r.status_code)
seen, cursor, pages = [], None, 0
while pages < 50:
    params = {"limit": 2}
    if cursor:
        params["cursor"] = cursor
    body = c.get(api + "/documents", params=params, headers=ADMIN).json()
    seen += [i["id"] for i in body["items"]]
    pages += 1
    if not body["has_more"]:
        break
    cursor = body["next_cursor"]
check("курсорная пагинация без дублей", len(seen) == len(set(seen)) and len(seen) >= 7, (len(seen), len(set(seen))))
r = c.get(api + "/documents", params={"q": "договор"}, headers=ADMIN); check("список с q=", r.status_code == 200, r.status_code)
r = c.get(api + "/documents", params={"sort": "мусор"}, headers=ADMIN); check("неизвестный sort не даёт 500", r.status_code in (200, 422), r.status_code)

# --- файлы и страницы ---
r = c.get(f"{api}/documents/{DOC}/file", headers=ADMIN)
check("оригинал отдаётся 200 и совпадает байт в байт", r.status_code == 200 and r.content == files["dogovor_postavki_47-2024.pdf"], r.status_code)
check("оригинал: nosniff", r.headers.get("x-content-type-options") == "nosniff", dict(r.headers))
r = c.get(f"{api}/documents/{DOC}/pages/1/image", headers=ADMIN)
check("изображение страницы 200 (image/*)", r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"), r.status_code)
r = c.get(f"{api}/documents/{DOC}/pages/99/image", headers=ADMIN); check("страница 99 -> 404", r.status_code == 404, r.status_code)
r = c.get(f"{api}/files/..%2F..%2Fetc%2Fpasswd", headers=ADMIN)
check("path traversal через /files отклонён", r.status_code in (400, 403, 404), r.status_code)
r = c.get(f"{api}/files/..%2F..%2F..%2F..%2Fetc%2Fpasswd", headers=ADMIN)
check("path traversal (глубже) не отдаёт файл", r.status_code in (400, 403, 404) and b"root:" not in r.content, r.status_code)

# --- верификация, reprocess, экспорт ---
r = c.put(f"{api}/documents/{DOC}/verify", json={"summary": "Проверка", "is_verified": True}, headers=VW)
check("VIEWER не может верифицировать (403)", r.status_code == 403, r.status_code)
r = c.put(f"{api}/documents/{DOC}/verify", json={"summary": "Проверено оператором", "is_verified": True,
          "entities": {"persons": ["Иванов И.И."], "organizations": ["ООО Тест"], "dates": ["01.02.2024"],
                       "amounts": ["100 руб."], "inn_ogrn": [], "addresses": []}}, headers=OP)
check("верификация оператором 200", r.status_code == 200 and r.json()["is_verified"], r.text[:200])
r = c.put(f"{api}/documents/{DOC}/verify", json={"entities": {"persons": "не список"}}, headers=OP)
check("верификация с неверным типом сущностей 422", r.status_code == 422, r.status_code)
for fmt in ["json", "csv", "docx", "md", "txt"]:
    r = c.get(f"{api}/documents/{DOC}/export", params={"format": fmt}, headers=VW)
    check(f"экспорт {fmt} 200 и непустой", r.status_code == 200 and len(r.content) > 10, r.status_code)
r = c.get(f"{api}/documents/{DOC}/export", params={"format": "exe"}, headers=ADMIN)
check("экспорт неизвестного формата 422", r.status_code == 422, r.status_code)
r = c.post(api + "/documents/export", params={"format": "csv", "ids": [docs[0]["id"], docs[1]["id"]]}, headers=ADMIN)
check("пакетный csv 200", r.status_code == 200, r.status_code)
r = c.post(api + "/documents/export", params={"format": "xlsx", "ids": [docs[0]["id"]]}, headers=ADMIN)
check("пакетный xlsx 200 (zip-контейнер)", r.status_code == 200 and r.content[:2] == b"PK", r.status_code)
r = c.post(api + "/documents/export", params={"format": "csv", "ids": [str(uuid.uuid4())]}, headers=ADMIN)
check("пакетный экспорт без совпадений 404", r.status_code == 404, r.status_code)

# --- поиск ---
for mode in ["fulltext", "semantic", "hybrid"]:
    r = c.post(api + "/search", json={"query": "договор поставки", "mode": mode, "limit": 5}, headers=VW)
    check(f"поиск {mode} 200", r.status_code == 200 and "hits" in r.json(), r.status_code)
r = c.post(api + "/search", json={"query": "", "mode": "hybrid"}, headers=ADMIN); check("пустой запрос 422", r.status_code == 422, r.status_code)
r = c.post(api + "/search", json={"query": "x", "mode": "magic"}, headers=ADMIN); check("неизвестный режим 422", r.status_code == 422, r.status_code)
r = c.post(api + "/search", json={"query": "x" * 501, "mode": "hybrid"}, headers=ADMIN); check("запрос >500 символов 422", r.status_code == 422, r.status_code)
r = c.post(api + "/search", json={"query": "договор", "mode": "fulltext", "filters": {"doc_type": "Договор"}}, headers=ADMIN)
check("поиск с фильтром doc_type 200", r.status_code == 200, r.status_code)
r = c.post(api + "/search", json={"query": "%_'\" ; DROP TABLE users; --", "mode": "fulltext"}, headers=ADMIN)
check("SQL-подобная строка не ломает поиск (200)", r.status_code == 200, r.status_code)
r = c.post(api + "/search", json={"query": "договор", "mode": "fulltext", "limit": 0}, headers=ADMIN)
check("limit=0 отклонён 422", r.status_code == 422, r.status_code)

# --- чат ---
r = c.post(f"{api}/documents/{DOC}/chat", json={"question": "Какая сумма договора?"}, headers=VW)
check("чат по документу 200 (VIEWER может читать)", r.status_code == 200 and "answer" in r.json(), r.status_code)
r = c.post(f"{api}/documents/{DOC}/chat", json={"question": ""}, headers=VW); check("пустой вопрос 422", r.status_code == 422, r.status_code)
r = c.post(f"{api}/documents/{uuid.uuid4()}/chat", json={"question": "а?"}, headers=VW); check("чат несуществующего 404", r.status_code == 404, r.status_code)

# --- reprocess ---
r = c.post(f"{api}/documents/{DOC}/reprocess", headers=VW)
check("VIEWER не может перезапустить обработку (403)", r.status_code == 403, r.status_code)
r = c.post(f"{api}/documents/{DOC}/reprocess", headers=OP)
check("reprocess оператором 200", r.status_code == 200, r.status_code)

# --- дашборд / система ---
r = c.get(api + "/dashboard/stats", headers=VW); check("dashboard stats 200", r.status_code == 200, r.status_code)
r = c.get(api + "/system/status", headers=VW); check("system status 200", r.status_code == 200, r.status_code)

# --- удаление: владелец/админ, viewer нельзя ---
TMP = c.post(api + "/documents/upload", files=[("files", ("temp.txt", "временный документ".encode(), "text/plain"))],
             headers=OP).json()["documents"][0]["id"]
r = c.delete(f"{api}/documents/{TMP}", headers=VW); check("VIEWER не может удалять (403)", r.status_code == 403, r.status_code)
r = c.delete(f"{api}/documents/{TMP}", headers=ADMIN); check("админ удаляет (204)", r.status_code == 204, r.status_code)
r = c.get(f"{api}/documents/{TMP}", headers=ADMIN); check("удалённый документ 404", r.status_code == 404, r.status_code)

# --- аудит ---
audit = c.get(api + "/admin/audit", params={"limit": 500}, headers=ADMIN).json()
items = audit if isinstance(audit, list) else audit.get("items", [])
actions = {a["action"] for a in items}
check("аудит фиксирует загрузку, вход и экспорт", "document_uploaded" in actions and "login" in actions and "document_exported" in actions, actions)


fails = [n for n, ok, _ in results if not ok]
print(f"\nИТОГО: {len(results) - len(fails)} PASS, {len(fails)} FAIL")
for n in fails:
    print("  FAIL:", n)
sys.exit(1 if fails else 0)
