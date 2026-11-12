# Локальный ИИ на GPU: развёртывание и настройка

Платформа полностью локальна: LLM, эмбеддинги и OCR работают на вашем оборудовании,
внешние API не используются. Документ описывает запуск на ПК/сервере с NVIDIA GPU,
подбор моделей под объём видеопамяти и замер производительности.

---

## 1. Минимальные требования

| Сценарий | CPU | RAM | Диск | GPU (VRAM) | Что работает |
|---|---|---|---|---|---|
| Демо без ИИ | 4 ядра | 8 ГБ | 5 ГБ | — | OCR текст-слоя, regex-NER, эвристический классификатор |
| Рабочий CPU-контур | 8 ядер | 16 ГБ | 25 ГБ | — | + LLM `qwen2.5:3b-instruct` (медленно: 15–40 с/документ) |
| Рабочий GPU-контур | 8 ядер | 16 ГБ | 40 ГБ | 8 ГБ | + `qwen2.5:7b-instruct`, `bge-m3` (~2–5 с/документ) |
| Полный контур + OCR сканов | 8 ядер | 32 ГБ | 60 ГБ | 12–16 ГБ | + `qwen2-vl:7b` (VLM-OCR), всё в VRAM одновременно |
| Максимум качества | 16 ядер | 64 ГБ | 100 ГБ | 24 ГБ+ | + `qwen2.5:14b-instruct` / `32b-instruct` |

Требования к ПО: Python 3.11+, Node.js 20+ (для фронтенда), NVIDIA Driver 535+,
для Docker — `nvidia-container-toolkit`.

---

## 2. Быстрый старт (нативная установка, рекомендуется для ПК с GPU)

```bash
git clone <repo> && cd project_mp2

python scripts/check_env.py     # покажет GPU, модели, OCR — что уже готово
./scripts/bootstrap.sh          # venv + зависимости + .env + демо-данные
./scripts/pull_models.sh        # LLM + bge-m3 (модель подберётся по VRAM)
WITH_VISION=1 ./scripts/pull_models.sh   # + VLM для распознавания сканов

make dev                        # backend :8000 + frontend :5173
```

Ollama 0.3+ автоматически использует CUDA, если драйвер установлен: отдельная
настройка не нужна. Проверить, что модель реально на GPU:

```bash
ollama ps
# NAME                    ID     SIZE   PROCESSOR
# qwen2.5:7b-instruct     ...    5.2 GB 100% GPU
```

Если в колонке `PROCESSOR` указано `100% CPU` — драйвер не виден Ollama,
переустановите Ollama с официального сайта (не snap/flatpak).

### Конфигурация `.env`

Готовый шаблон: `cp .env.gpu.example .env`. Ключевые параметры:

```ini
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
EMBEDDING_MODEL=bge-m3
OLLAMA_VISION_MODEL=qwen2-vl:7b
OCR_ENGINE=auto            # tesseract → VLM → «нужен OCR»
LLM_TIMEOUT_SECONDS=300    # первый ответ включает загрузку весов в VRAM
```

---

## 3. Docker с GPU

```bash
# NVIDIA Container Toolkit должен быть установлен:
#   sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi  # проверка

cp .env.gpu.example .env
make docker-gpu     # docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Что делает `docker-compose.gpu.yml`:

- пробрасывает все GPU в контейнер Ollama (`deploy.resources.reservations.devices`);
- ставит `OLLAMA_KEEP_ALIVE=30m` и `OLLAMA_NUM_PARALLEL=2`, чтобы модель не выгружалась между запросами;
- докачивает в `ollama-models` основную модель, `bge-m3` и (при `PULL_VISION=1`) VLM;
- передаёт выбранные модели backend'у и Celery-воркеру.

Интерфейс: http://localhost:8080, API: http://localhost:8000/api/docs.

---

## 4. Какая модель под ваш объём VRAM

| VRAM | LLM (выбирает `scripts/pull_models.sh`) | Эмбеддинги | VLM-OCR | Комментарий |
|---|---|---|---|---|
| меньше 4.5 ГБ или CPU | `qwen2.5:1.5b-instruct` | `bge-m3` (можно на CPU) | — | минимум, качество извлечения ниже |
| 4.5–7 ГБ | `qwen2.5:3b-instruct` | `bge-m3` | — | базовый вариант для слабой карты |
| 7–20 ГБ | `qwen2.5:7b-instruct` | `bge-m3` | `qwen2-vl:7b` (при `WITH_VISION=1`) | рабочий вариант, полный контур: анализ + OCR сканов |
| от 20 ГБ | `qwen2.5:14b-instruct` | `bge-m3` | `qwen2-vl:7b` | оптимум качества для вуза |

Объём VRAM — по `nvidia-smi`; значения — ориентир по весам Q4_K_M, реальный расход выше на 10–20 % за счёт KV-кэша.
Вручную модель задаётся через `OLLAMA_MODEL`; `pull_models.sh` не выбирает модели больше 14B.

Скрипты подбирают LLM автоматически по VRAM (`scripts/pull_models.sh`,
`GET /api/v1/system/status` → `gpu.recommended_llm`, поле появляется только при обнаруженной GPU), интерфейс показывает
рекомендацию в разделе **Администрирование → Система и GPU**.

> Модели с лицензией для коммерческого использования: Qwen2.5 (Apache-2.0),
> bge-m3 (MIT), qwen2-vl (Apache-2.0). Для русскоязычных документов альтернатива —
> `vikhr` / `saiga` (см. Ollama library), но качество на реквизитах и таблицах
> у Qwen2.5 обычно выше.

---

## 5. OCR сканов: Tesseract или VLM

| | Tesseract 5 (rus+eng) | VLM (`qwen2-vl:7b` / `llava:7b`) |
|---|---|---|
| Скорость | 0,3–1 с/страница на CPU | 2–8 с/страница на GPU |
| Координаты слов (bbox) | **есть** — подсветка на скане | нет (только текст) |
| Печатный текст | хорошо | хорошо |
| Фото, наклон, печати | средне (нужен препроцессинг) | **лучше** |
| Рукописный текст | плохо | **существенно лучше** |
| Ресурсы | 0 ГБ VRAM | 6–10 ГБ VRAM |

Рекомендуется `OCR_ENGINE=auto`: Tesseract обрабатывает обычные сканы и даёт
bounding boxes, VLM подключается, когда Tesseract не установлен, а GPU есть.
Принудительно VLM: `OCR_ENGINE=ollama`.

Препроцессинг (`PREPROCESS_ENABLED=true`) включает удаление синей печати,
выравнивание наклона, бинаризацию и повышение резкости — это заметно улучшает
Tesseract на «фотографиях» документов.

---

## 6. Производительность: как получить честные цифры

Готовый замер на вашем железе (включая холодный старт модели):

```bash
cd backend
PYTHONPATH=. python -m scripts.benchmark --limit 5 --repeat 2
# → reports/benchmark.json: секунды на документ/страницу, док/мин, время LLM и эмбеддингов
```

Метрики качества (CER/WER/NER F1) на эталонном датасете:

```bash
PYTHONPATH=. python -m scripts.evaluate
# либо в интерфейсе: Администрирование → Качество ИИ → «Запустить оценку»
```

Ориентиры для `qwen2.5:7b-instruct` (Q4, одна GPU, документ 1–2 страницы):

| Этап | CPU (8 ядер) | GPU 8 ГБ | GPU 16 ГБ (14B) |
|---|---|---|---|
| OCR текстового слоя | < 0,1 с | < 0,1 с | < 0,1 с |
| OCR скана (Tesseract) | 0,5–1 с | 0,5–1 с | 0,5–1 с |
| Анализ LLM (классификация + сущности + summary) | 15–40 с | 2–5 с | 4–8 с |
| Эмбеддинги (`bge-m3`) | 0,2–0,6 с | 0,05–0,15 с | 0,05–0,15 с |
| **Итого на документ** | **15–40 с** | **2,5–6 с** | **5–9 с** |

Это ориентиры, а не гарантия: реальные значения зависят от модели, драйвера,
длины документа и загрузки GPU. `scripts/benchmark.py` пишет фактические цифры —
их и стоит приводить в отчёте по НИР.

---

## 7. Оптимизация под GPU

```bash
# 1. Держать модель в памяти между запросами (сильно снижает первый ответ)
export OLLAMA_KEEP_ALIVE=30m

# 2. Параллельная обработка: 2 документа одновременно
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=2     # LLM + VLM одновременно

# 3. Уменьшить контекст, если документы короткие (экономит VRAM)
#    OLLAMA_CONTEXT_LENGTH=4096
```

В платформе:

- `QUEUE_BACKEND=celery` + `celery_worker --concurrency=2` — параллельная обработка
  нескольких документов (Redis уже в compose);
- `EMBEDDING_DIM=512` (по умолчанию) — экономия памяти в векторном индексе;
- пакетная загрузка: до 30 файлов в минуту на IP (`RATE_LIMIT_UPLOAD_PER_MINUTE`).

---

## 8. Диагностика

| Симптом | Причина | Решение |
|---|---|---|
| `LLM: fallback` в интерфейсе | Ollama недоступна | `ollama serve`, проверить `OLLAMA_URL` |
| Очень медленно, `100% CPU` в `ollama ps` | нет доступа к GPU | переустановить Ollama с официального сайта, обновить драйвер |
| `модель не загружена` | не выполнен `ollama pull` | `./scripts/pull_models.sh` |
| `OCR-движок: none` | нет Tesseract и VLM | `apt install tesseract-ocr tesseract-ocr-rus` или `WITH_VISION=1 ./scripts/pull_models.sh` |
| OOM / `CUDA out of memory` | модель не влезает в VRAM | взять модель меньше (7B вместо 14B), закрыть другие GPU-процессы |
| Пустые сущности | скан без OCR | проверить `ocr.engine` в `/api/v1/system/status` |

Полная самодиагностика: `python scripts/check_env.py` и раздел
**Администрирование → Система и GPU** (список подсказок формируется backend'ом).

---

## 9. Что делать, когда приедет оборудование

1. `python scripts/check_env.py` — зафиксировать GPU/VRAM/драйвер.
2. `./scripts/bootstrap.sh` → `WITH_VISION=1 ./scripts/pull_models.sh`.
3. `make dev`, загрузить 3–5 реальных документов (сканы и текстовые).
4. `cd backend && PYTHONPATH=. python -m scripts.benchmark --limit 5 --repeat 2` — цифры скорости.
5. `PYTHONPATH=. python -m scripts.evaluate` — CER/WER/F1 (в отчёт; сканы попадут
   в метрики автоматически, как только заработает OCR).
6. Администрирование → Качество ИИ → «Запустить оценку» — те же метрики из интерфейса.
