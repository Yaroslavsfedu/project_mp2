# IDP Platform — оцифровка документов и интеллектуальная обработка информации

Веб-приложение (Intelligent Document Processing) для загрузки, распознавания и интеллектуальной
обработки документов **полностью в локальном контуре**: OCR с координатами слов, локальная LLM
(Ollama / Qwen 2.5), извлечение сущностей, гибридный поиск, ручная верификация и экспорт.

**Запуск на своём ПК:** `./scripts/bootstrap.sh` → `make dev` (подробно — раздел
[«Быстрый старт»](#быстрый-старт-локально)). Для машины с GPU — [`docs/GPU_DEPLOY.md`](docs/GPU_DEPLOY.md).

> Междисциплинарный проект Проектного офиса ИКТИБ ЮФУ.
> Наставник: Семенов В. Е., м.н.с. НИИ МВС ЮФУ, руководитель СНО «Экспонента».
> Полная инженерная спецификация — [`AI/context/project.md`](AI/context/project.md).

![Стек](https://img.shields.io/badge/stack-FastAPI%20%2B%20React%20%2B%20PostgreSQL%2Fpgvector-8430e0)
![AI](https://img.shields.io/badge/AI-Ollama%20%C2%B7%20Qwen%202.5%20%C2%B7%20bge--m3-9a4dfc)
![On-Premise](https://img.shields.io/badge/On--Premise-no%20external%20APIs-1f9d6b)

---

## Что уже реализовано (MVP+)

| Блок требований | Статус | Где смотреть |
|---|---|---|
| Пакетная загрузка PDF / JPG / PNG / TIFF / DOCX / TXT | ✅ | `frontend/src/components/upload`, `backend/app/api/v1/documents.py` |
| Валидация: расширение, размер, **Magic Bytes**, безопасные имена | ✅ | `validate_upload()`, `safe_filename()` |
| Гибридный PDF: нативный текстовый слой без OCR + рендер страниц 150 DPI | ✅ | `ml_pipeline/pdf_handler.py` |
| Препроцессинг CV: deskew, удаление синих/фиолетовых печатей, denoise, адаптивная бинаризация | ✅ | `ml_pipeline/preprocessor.py` |
| OCR-движок с координатами и confidence (Tesseract rus+eng), подключаемые PaddleOCR/EasyOCR | ✅ | `ml_pipeline/ocr_engine.py` |
| Локальная LLM: пост-коррекция, классификация, NER, суммаризация, строгий JSON | ✅ | `ml_pipeline/llm_service.py` |
| Эвристический fallback без GPU/Ollama (regex-NER, классификация по маркерам, extractive summary) | ✅ | `llm_service.py` |
| Интерактивный Split-Screen: скан + Bounding Boxes ↔ текст ↔ форма сущностей | ✅ | `frontend/src/components/viewer`, `DocumentViewPage.tsx` |
| Ручная верификация (Human-in-the-loop) с редактированием сущностей и статусом `verified` | ✅ | `PUT /api/v1/documents/{id}/verify` |
| Полнотекстовый поиск (стемминг + подсветка), семантический (эмбеддинги) и гибридный | ✅ | `services/search_service.py`, `SearchPage.tsx` |
| RAG-чат по документу с указанием источников | ✅ | `POST /api/v1/documents/{id}/chat` |
| Экспорт: JSON, CSV, DOCX, Markdown, TXT + пакетный XLSX | ✅ | `services/export_service.py` |
| WebSocket-статусы обработки (`pending → ocr_processing → llm_processing → completed`) | ✅ | `api/v1/ws.py`, `useDocumentEvents.ts` |
| JWT-аутентификация, bcrypt, роли `admin/operator/viewer` (RBAC) | ✅ | `core/security.py`, `api/v1/auth.py` |
| Docker Compose: frontend, backend, celery, postgres+pgvector, redis, minio, ollama | ✅ | `docker-compose.yml` |
| Миграции Alembic (включая `audit_logs` и GIN-индексы PostgreSQL) | ✅ | `backend/migrations/versions/` |
| Журнал аудита действий (кто, что, когда, IP) | ✅ | `services/audit_service.py`, `PUT /verify` и др. |
| Админ-панель: пользователи, роли, аудит, метрики, состояние стека | ✅ | `frontend/src/pages/AdminPage.tsx`, `api/v1/admin.py` |
| Метрики качества: CER, WER, NER P/R/F1, точность классификации + отчёт JSON/Markdown | ✅ | `services/quality_service.py`, `scripts/evaluate.py` |
| Замер производительности на конкретном железе | ✅ | `backend/scripts/benchmark.py`, `reports/benchmark.json` |
| Нагрузочный тест через API (rps, p95, док/мин) | ✅ | `backend/scripts/load_test.py`, `make load-test` |
| Rate limiting (перебор паролей, флуд загрузками) | ✅ | `core/ratelimit.py` |
| Полнотекстовый поиск PostgreSQL: `to_tsvector('russian')` + GIN + `ts_headline` | ✅ | `services/search_service.py`, миграция `7c1a4f9e2b10` |
| OCR через локальную VLM (`qwen2-vl`, `llava`) для сканов и рукописи | ✅ | `ml_pipeline/ocr_engine.py`, `LLM.vision_extract_text` |
| GPU-профиль (Ollama CUDA + VLM), авто-подбор модели по VRAM | ✅ | `docker-compose.gpu.yml`, `core/hardware.py` |
| Скрипты установки и диагностики окружения | ✅ | `scripts/bootstrap.sh`, `scripts/check_env.py`, `Makefile` |
| Тесты (unit + интеграционные: API, пайплайн, RBAC, аудит, метрики) | ✅ | `backend/tests` (44 теста) |

---

## Архитектура

```
┌──────────────────────────── Браузер (React 18 + TS + Tailwind) ────────────────────────────┐
│  Dashboard · Documents · Upload · DocumentView (scan + bbox + entities + chat) · Search    │
│  TanStack Query (кэш/инвалидация) · Zustand (auth, toasts) · WebSocket-статусы            │
└───────────────────────────────────────────┬────────────────────────────────────────────────┘
                                            │ /api/v1 (REST, JWT)   /api/v1/ws/documents
┌───────────────────────────────────────────▼────────────────────────────────────────────────┐
│                              FastAPI (async) · SQLAlchemy 2.0                              │
│  auth · documents · search · export · system · ws        services: storage/search/export/  │
│                                                                    pipeline/events       │
└───────┬───────────────────────┬────────────────────────┬───────────────────────┬────────────┘
        │                       │                        │                       │
  ┌─────▼─────┐         ┌───────▼───────┐        ┌───────▼────────┐      ┌───────▼────────┐
  │ PostgreSQL │         │ MinIO (S3)    │        │ Redis + Celery │      │ Ollama         │
  │ + pgvector │         │ оригиналы +   │        │ очередь OCR /  │      │ qwen2.5:7b     │
  │ FTS+JSONB  │         │ PNG страниц   │        │ LLM задач      │      │ bge-m3         │
  └────────────┘         └───────────────┘        └───────┬────────┘      └───────▲────────┘
                                                          │                       │
                                              ┌───────────▼───────────────────────┴──────┐
                                              │ ML-пайплайн: PyMuPDF → OpenCV → OCR →    │
                                              │ LLM/NLP → chunking → embeddings          │
                                              └──────────────────────────────────────────┘
```

### Локальные режимы

Приложение запускается **без внешней инфраструктуры**: если PostgreSQL/MinIO/Ollama/Celery
недоступны, используются SQLite, локальная ФС, inline-задачи и эвристический NLP-движок.
Переключение на продуктивный контур — только переменными окружения (`DATABASE_URL`,
`STORAGE_BACKEND`, `QUEUE_BACKEND`, `OLLAMA_URL`), код менять не нужно.

---

## Быстрый старт (локально)

Требования: Python 3.11+, Node.js 20+.

### Установка одной командой

```bash
python scripts/check_env.py      # что уже есть: GPU, Ollama, Tesseract, Node
./scripts/bootstrap.sh           # venv + зависимости + .env + демо-данные
WITH_VISION=1 ./scripts/pull_models.sh   # локальные модели (LLM + bge-m3 + VLM)
make dev                         # backend :8000 + frontend :5173 одновременно
```

Демо-доступ: `demo@idp-project.ru` / `demo12345`. Полезные команды: `make check`,
`make test`, `make evaluate`, `make benchmark`, `make load-test`, `make docker-gpu`.

### Вручную (если нужен контроль над каждым шагом)

```bash
# 1. Бэкенд
cd backend
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
PYTHONPATH=. uvicorn app.main:app --reload --port 8000
# Swagger: http://localhost:8000/api/docs

# 2. Фронтенд (в новом терминале)
cd frontend
npm install
npm run dev
# Приложение: http://localhost:5173
```

Демо-доступ (создаётся автоматически при `SEED_DEMO=true`):

```
email:    demo@idp-project.ru
password: demo12345
```

При первом запуске в систему загружаются демонстрационные документы из `samples/`
(договор поставки, приказ, счёт, акт, научная статья, служебная записка) — они проходят
полный пайплайн, чтобы интерфейс сразу был наполнен данными.

> Без Ollama анализ выполняется встроенным fallback-движком (regex-NER + классификация по маркерам).
> Интерфейс явно показывает, каким движком обработан документ.

### Подключение локальной LLM

```bash
ollama pull qwen2.5:7b-instruct      # основная модель (≈5 ГБ, Q4_K_M)
ollama pull bge-m3                   # эмбеддинги для семантического поиска
ollama serve                         # http://localhost:11434
```

`OLLAMA_URL`, `OLLAMA_MODEL`, `EMBEDDING_MODEL` — в `.env` (см. `.env.example`).
Для ПК с GPU используйте `.env.gpu.example` и [`docs/GPU_DEPLOY.md`](docs/GPU_DEPLOY.md).

### OCR для сканов

```bash
# Вариант 1: Tesseract (CPU, быстро, есть Bounding Boxes)
sudo apt-get install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng

# Вариант 2: локальная VLM (GPU) — лучше на фото, наклонах, рукописи
ollama pull qwen2-vl:7b
# в .env: OCR_ENGINE=ollama  (или auto: Tesseract → VLM → «нужен OCR»)
```

PDF с текстовым слоем, DOCX и TXT обрабатываются без OCR. Если OCR-движка нет,
страница помечается «нужен OCR» — текст и сущности не выдумываются.

---

## Метрики качества (измерены на эталонном датасете)

Отчёт формируется автоматически: `reports/quality/quality.json` + `quality.md`,
а также в интерфейсе **Администрирование → Качество ИИ**.

Нагрузочная проверка (через запущенный API, весь стек целиком):

```bash
cd backend && PYTHONPATH=. python -m scripts.load_test --count 50 --concurrency 4
# → reports/load_test.json: rps загрузки, p50/p95 задержки, док/мин обработки, ошибки
```

```bash
cd backend && PYTHONPATH=. python -m scripts.evaluate
```

| Метрика | Значение | Как считается |
|---|---|---|
| CER | **0,02 %** | символьная ошибка распознавания (Левенштейн / длина эталона) |
| WER | **0,19 %** | ошибка на уровне слов (Левенштейн по токенам) |
| NER F1 | **98,97 %** (P 98,0 / R 100,0) | микро-усреднение по ФИО, организациям, датам, суммам, ИНН/ОГРН, адресам, номеру документа |
| Точность классификации | **100 %** | 6 типов документов (договор, приказ, счёт, акт, статья, заявление) |
| Время обработки | **0,20 с/документ** | fallback-движок без LLM (CPU, песочница) |

Как читать эти цифры честно:

* датасет синтетический (6 текстовых документов + 2 «скана»), поэтому CER/WER
  показывают корректность пайплайна, а не качество OCR на реальных помятых сканах;
* сканы (`scan_*.jpg`) помечены `requires_ocr` и попадают в метрики автоматически,
  как только доступен Tesseract или VLM — без OCR они честно пропускаются с пометкой;
* NER F1 измеряется regex/LLM-извлечением на эталонной разметке; на реальных
  документах значение будет ниже — это и должен показать прогон на ваших данных;
* производительность на GPU измеряется `scripts/benchmark.py` (см. `docs/GPU_DEPLOY.md`).

---

## Локальный ИИ на GPU

Платформа рассчитана на полностью локальный инференс; GPU-контур включается без
изменения кода:

```bash
cp .env.gpu.example .env
WITH_VISION=1 ./scripts/pull_models.sh         # подбор модели по VRAM
make dev                                        # или make docker-gpu
```

| VRAM | LLM | Эмбеддинги | VLM-OCR |
|---|---|---|---|
| 6–8 ГБ | `qwen2.5:7b-instruct` | `bge-m3` | — |
| 10–12 ГБ | `qwen2.5:14b-instruct` | `bge-m3` | — |
| 12–16 ГБ | `qwen2.5:7b-instruct` | `bge-m3` | `qwen2-vl:7b` |
| 20–24 ГБ | `qwen2.5:14b-instruct` | `bge-m3` | `qwen2-vl:7b` |

Раздел **Администрирование → Система и GPU** показывает фактическое состояние:
какие модели установлены, используется ли GPU, какой OCR-движок активен и что
нужно донастроить. Полное руководство (Docker + CUDA, оптимизация, диагностика):
[`docs/GPU_DEPLOY.md`](docs/GPU_DEPLOY.md).

---

## Администрирование и аудит

| Возможность | Детали |
|---|---|
| Пользователи и роли | `admin` / `operator` / `viewer`, блокировка учётной записи, защита от снятия последнего администратора и самоблокировки |
| Журнал аудита | загрузка, верификация, правка, повторная обработка, удаление, изменения пользователей, запуск оценки качества; фильтры по действию и документу |
| Метрики качества | запуск прогона датасета из интерфейса, карточки CER/WER/F1, таблица по документам, Markdown-отчёт |
| Состояние стека | Ollama и список моделей, GPU/VRAM и рекомендация модели, OCR-движок, подсказки по настройке |

API: `GET /api/v1/admin/overview`, `/users`, `PATCH /users/{id}`, `/audit`,
`GET /api/v1/admin/quality`, `POST /api/v1/admin/quality/run` — только для роли `admin`.

---

## Запуск в Docker (продуктивный контур)

```bash
cp .env.example .env
# при необходимости измените SECRET_KEY и пароли

docker compose up -d --build
docker compose logs -f backend

# Frontend (nginx):  http://localhost:8080
# API / Swagger:     http://localhost:8000/api/docs
# MinIO Console:     http://localhost:9001
```

Состав стека: `frontend` (nginx + собранный SPA), `backend` (uvicorn, 2 воркера),
`celery_worker`, `postgres` (`pgvector/pgvector:pg16`), `redis`, `minio`, `ollama`
+ one-shot контейнер `ollama-models`, который скачивает веса.

Для GPU используйте overlay-файл: `make docker-gpu`
(`docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build`) —
он пробрасывает GPU в Ollama, задаёт `OLLAMA_KEEP_ALIVE`/`OLLAMA_NUM_PARALLEL`
и докачивает VLM для OCR сканов. Требуется `nvidia-container-toolkit`.

---

## API (кратко)

Полная спецификация — Swagger `/api/docs`, OpenAPI — `/api/openapi.json`.

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/api/v1/auth/register` / `login` | Регистрация / вход (JWT) |
| GET | `/api/v1/auth/me` | Профиль пользователя |
| POST | `/api/v1/documents/upload` | Пакетная загрузка (multipart) |
| GET | `/api/v1/documents` | Список: пагинация, статус, тип, дата, поиск |
| GET | `/api/v1/documents/{id}` | Карточка: страницы, Bounding Boxes, сущности, резюме |
| GET | `/api/v1/documents/{id}/pages/{n}/image` | PNG страницы (с JWT) |
| PUT | `/api/v1/documents/{id}/verify` | Сохранение правок оператора |
| POST | `/api/v1/documents/{id}/reprocess` | Повторный прогон пайплайна |
| DELETE | `/api/v1/documents/{id}` | Удаление из БД и хранилища |
| POST | `/api/v1/documents/{id}/chat` | RAG-вопрос по документу |
| POST | `/api/v1/search` | `hybrid` / `fulltext` / `semantic` + фильтры |
| GET | `/api/v1/documents/{id}/export?format=json\|csv\|docx\|md\|txt` | Экспорт карточки |
| POST | `/api/v1/documents/export?format=xlsx\|csv` | Пакетный экспорт сводной таблицы |
| GET | `/api/v1/dashboard/stats` | Статистика рабочего стола |
| GET | `/api/v1/admin/overview` · `/users` · `/audit` | Админ-панель (только `admin`) |
| PATCH | `/api/v1/admin/users/{id}` | Роль / блокировка пользователя |
| GET/POST | `/api/v1/admin/quality` · `/quality/run` | Метрики CER/WER/F1: чтение и запуск прогона |
| GET | `/api/v1/system/status` | Состояние стека: LLM и модели, GPU/VRAM, OCR, подсказки |
| WS | `/api/v1/ws/documents?token=…` | Статусы и прогресс обработки |

Пример:

```bash
TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"demo@idp-project.ru","password":"demo12345"}' | jq -r .access_token)

curl -s -X POST localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer $TOKEN" -F "files=@samples/dogovor_postavki_47-2024.pdf"

curl -s -X POST localhost:8000/api/v1/search \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"query":"неустойка за просрочку поставки","mode":"hybrid"}'
```

---

## Пайплайн обработки

1. **Валидация и приём**: расширение + Magic Bytes + лимит 50 МБ → оригинал в хранилище, запись `PENDING`.
2. **Извлечение страниц**: PDF с текстовым слоем → слова и координаты напрямую (без OCR);
   скан → рендер 150 DPI → `preprocess` (deskew по HoughLinesP, удаление печатей в HSV, denoise,
   адаптивный порог) → OCR (`image_to_data`); DOCX/TXT → рендер страницы для корректных Bounding Boxes.
3. **Группировка в строки** (`_line_boxes_from_words`) — компактные, читаемые рамки для UI.
4. **Локальный ИИ**: коррекция текста (склейка переносов, `0/О`, `1/l`), классификация,
   NER (ФИО с падежами, организации, даты, суммы, ИНН/КПП/ОГРН/БИК, номер, адреса), резюме, ключевые слова.
5. **Индексация**: чанки по абзацам + эмбеддинги (Ollama `bge-m3` либо hashing-fallback), сохранение в БД.
6. **Верификация**: оператор правит текст/сущности → `VERIFIED`, экспорт в нужный формат.

Статусы транслируются на фронтенд по WebSocket и через поллинг TanStack Query.

---

## Тесты

```bash
cd backend
PYTHONPATH=. pytest -q
```

Покрыто (44 теста): нормализация/стемминг/чанкинг, эвристический NER и классификация
(договор ≠ акт), экспортёры (JSON/CSV/DOCX/XLSX), интеграционный сценарий «загрузка → OCR →
NER → поиск → чат → экспорт → верификация → удаление», метрики CER/WER/F1 и
fuzzy-сопоставление сущностей, журнал аудита, RBAC админ-эндпоинтов, rate limiting,
формат `/system/status`.

CI (`.github/workflows/ci.yml`) дополнительно генерирует демо-данные, считает метрики
качества, проверяет окружение и собирает фронтенд.

```bash
cd frontend && npm run build   # проверка типов + production-сборка
```

---

## Структура репозитория

```
├── AI/context/project.md        # полная спецификация проекта (SSoT)
├── backend/
│   ├── app/
│   │   ├── api/v1/              # auth, documents, search, export, admin, system, ws
│   │   ├── core/                # config, security, database, ratelimit, hardware (GPU)
│   │   ├── models/ schemas/     # SQLAlchemy 2.0 и Pydantic v2 (+ audit_logs)
│   │   ├── services/            # storage, search, export, pipeline, audit, quality, seed
│   │   └── ml_pipeline/         # preprocessor, pdf_handler, ocr_engine (+VLM), llm_service, renderer, celery
│   ├── migrations/              # Alembic: схема, audit_logs, GIN-индексы PostgreSQL
│   ├── scripts/                 # generate_samples, evaluate (CER/WER/F1), benchmark, check_env
│   └── tests/
├── frontend/src/
│   ├── api/                     # axios, endpoints, WebSocket, authed images
│   ├── components/{layout,ui,upload,viewer,chat}
│   ├── pages/                   # Dashboard, Documents, Upload, DocumentView, Search, Admin, Login
│   ├── store/ types/ lib/
├── scripts/                     # bootstrap.sh, dev.sh, pull_models.sh, check_env.py
├── docs/GPU_DEPLOY.md           # развёртывание на GPU, модели, диагностика
├── samples/                     # демо-документы + ground_truth.json (эталон разметки)
├── reports/                     # quality.json/md, benchmark.json (генерируются)
├── Makefile · docker-compose.yml · docker-compose.gpu.yml
```

---

## Безопасность и соответствие требованиям

* Данные не передаются во внешние сети: LLM, OCR и эмбеддинги работают локально.
* Оригиналы и страницы доступны только по JWT (в prod — presigned URL MinIO, TTL 15 мин).
* Проверка реального типа файла по Magic Bytes, очистка имён от Path Traversal, лимит размера.
* Пароли — bcrypt; доступ к эндпоинтам — JWT; роли `admin/operator/viewer` (RBAC).
* Логи не содержат содержимого документов; ошибки обработки сохраняются в `documents.error_message`.
* Журнал аудита фиксирует все значимые действия (загрузка, правка, верификация, удаление,
  изменение ролей) с IP-адресом и доступен только администратору.
* Rate limiting: 10 попыток входа/мин и 30 загрузок/мин на IP (скользящее окно в памяти
  процесса; для нескольких воркеров лимиты считаются на каждый процесс — в проде
  дублируется на уровне nginx/API-gateway).
* Администратор не может снять роль с себя или заблокировать себя; система не даёт
  остаться без активного администратора.

---

## Roadmap

* **Этап 1–2 (готово):** архитектура, загрузка, OCR, Bounding Boxes, UI-каркас.
* **Этап 3 (готово):** локальная LLM (Ollama) + fallback, NER, классификация, суммаризация, RAG.
* **Этап 4 (готово):** гибридный поиск, Split-Screen верификация, экспорт JSON/CSV/DOCX/MD/TXT/XLSX.
* **Этап 5 (готово):** замер CER/WER/F1 на эталонном датасете + отчёт, журнал аудита,
  админ-панель с ролями, rate limiting, `tsvector`+GIN для PostgreSQL, VLM-OCR,
  GPU-профиль с авто-подбором модели, скрипты установки и диагностики.
* **Этап 6 (следующее):** нагрузочное тестирование на 1000+ документов, HNSW-индекс
  pgvector при больших объёмах, PaddleOCR/VLM для сложной вёрстки и таблиц, OSD-поворот
  страниц, fine-tune промптов на корпусе ИКТИБ, shadcn/ui для унификации компонентов.

### Ограничения текущей версии (честно)

* Метрики CER/WER в репозитории получены на синтетических документах с текстовым слоем;
  качество OCR на реальных сканах нужно измерить на вашем оборудовании (`make evaluate`).
* Docker-образы и GPU-профиль не собирались в среде разработки (нет Docker/NVIDIA):
  конфигурация проверена статически, первый запуск на целевом сервере — за вами.
* Rate limiting хранится в памяти процесса; в многопроцессном контуре нужен общий
  счётчик (Redis) или лимиты на reverse-proxy.

---

## Команда и роли

Аналитик · UI/UX-дизайнер · Фронтенд-разработчик · Бэкенд-разработчик · ML/OCR-инженер ·
Тестировщик · DevOps-инженер — зоны ответственности описаны в разделе 12 спецификации
[`AI/context/project.md`](AI/context/project.md).
