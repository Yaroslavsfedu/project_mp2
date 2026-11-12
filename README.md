# IDP Platform — оцифровка документов и интеллектуальная обработка информации

Веб-приложение (Intelligent Document Processing) для загрузки, распознавания и интеллектуальной
обработки документов **полностью в локальном контуре**: OCR с координатами слов, локальная LLM
(Ollama / Qwen 2.5), извлечение сущностей, гибридный поиск, ручная верификация и экспорт.

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
| Миграции Alembic | ✅ | `backend/migrations/versions/…initial…` |
| Тесты (unit + интеграционный пайплайн) | ✅ | `backend/tests` |

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

### OCR для сканов

```bash
# Debian/Ubuntu
sudo apt-get install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng
```

PDF с текстовым слоем, DOCX и TXT обрабатываются без OCR.

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
+ one-shot контейнер `ollama-models`, который скачивает веса. Для GPU раскомментируйте
`deploy.resources` у сервиса `ollama` (требуется `nvidia-container-toolkit`).

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
| GET | `/api/v1/system/status` | Состояние LLM/OCR/хранилища/очередей |
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

Покрыто: нормализация/стемминг/чанкинг, эвристический NER и классификация (договор ≠ акт),
экспортёры (JSON/CSV/DOCX/XLSX), а также интеграционный сценарий «загрузка → OCR → NER →
поиск → чат → экспорт → верификация → удаление» (27 тестов).

```bash
cd frontend && npm run build   # проверка типов + production-сборка
```

---

## Структура репозитория

```
├── AI/context/project.md        # полная спецификация проекта (SSoT)
├── backend/
│   ├── app/
│   │   ├── api/v1/              # auth, documents, search, export, system, ws
│   │   ├── core/                # config, security, database
│   │   ├── models/ schemas/     # SQLAlchemy 2.0 и Pydantic v2
│   │   ├── services/            # storage, search, export, pipeline, events, seed
│   │   └── ml_pipeline/         # preprocessor, pdf_handler, ocr_engine, llm_service, renderer, celery
│   ├── migrations/              # Alembic
│   ├── scripts/generate_samples.py
│   └── tests/
├── frontend/src/
│   ├── api/                     # axios, endpoints, WebSocket, authed images
│   ├── components/{layout,ui,upload,viewer,chat}
│   ├── pages/                   # Dashboard, Documents, Upload, DocumentView, Search, Login
│   ├── store/ types/ lib/
├── samples/                     # демонстрационные документы
└── docker-compose.yml
```

---

## Безопасность и соответствие требованиям

* Данные не передаются во внешние сети: LLM, OCR и эмбеддинги работают локально.
* Оригиналы и страницы доступны только по JWT (в prod — presigned URL MinIO, TTL 15 мин).
* Проверка реального типа файла по Magic Bytes, очистка имён от Path Traversal, лимит размера.
* Пароли — bcrypt; доступ к эндпоинтам — JWT; роли `admin/operator/viewer` (RBAC).
* Логи не содержат содержимого документов; ошибки обработки сохраняются в `documents.error_message`.

---

## Roadmap

* **Этап 1–2 (готово):** архитектура, загрузка, OCR, Bounding Boxes, UI-каркас.
* **Этап 3 (готово):** локальная LLM (Ollama) + fallback, NER, классификация, суммаризация, RAG.
* **Этап 4 (готово):** гибридный поиск, Split-Screen верификация, экспорт JSON/CSV/DOCX/MD/TXT/XLSX.
* **Этап 5 (в работе):** замер CER/WER и F1 на расширенном датасете, нагрузочное тестирование,
  подключение `pgvector`-индекса HNSW в продуктивном контуре, fine-tune промптов под предметную область,
  роли и аудит действий операторов.

---

## Команда и роли

Аналитик · UI/UX-дизайнер · Фронтенд-разработчик · Бэкенд-разработчик · ML/OCR-инженер ·
Тестировщик · DevOps-инженер — зоны ответственности описаны в разделе 12 спецификации
[`AI/context/project.md`](AI/context/project.md).
