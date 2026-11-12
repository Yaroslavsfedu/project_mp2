# IDP-платформа — частые команды. `make help` — список.
PYTHON ?= python3
BACKEND := backend
FRONTEND := frontend
VENV := $(BACKEND)/.venv
ACTIVATE := . $(VENV)/bin/activate

.DEFAULT_GOAL := help
.PHONY: help install check samples api web dev test smoke evaluate benchmark search-bench bulk-ingest scale-bench extreme-bench load-test models migrate docker docker-gpu docker-down logs psql

help: ## показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## установить зависимости, создать .env, собрать демо-данные
	./scripts/bootstrap.sh

check: ## проверить окружение (Python, Node, GPU, Ollama, OCR)
	$(PYTHON) scripts/check_env.py

samples: ## (пере)сгенерировать демо-документы и эталонную разметку
	cd $(BACKEND) && PYTHONPATH=. $(PYTHON) -m scripts.generate_samples --out ../samples --force

api: ## запустить backend (http://localhost:8000/api/docs)
	cd $(BACKEND) && PYTHONPATH=. $(ACTIVATE) && uvicorn app.main:app --reload --no-proxy-headers --host 0.0.0.0 --port 8000

web: ## запустить frontend (http://localhost:5173)
	cd $(FRONTEND) && npm run dev -- --host

dev: ## запустить backend и frontend вместе
	./scripts/dev.sh

test: ## тесты backend
	cd $(BACKEND) && PYTHONPATH=. $(ACTIVATE) && pytest -q

smoke: ## сквозная проверка API: 24 маршрута, роли, владение, граничные значения (нужен запущенный backend с RATE_LIMIT_ENABLED=false)
	cd $(BACKEND) && PYTHONPATH=. $(ACTIVATE) && python -m scripts.api_smoke $(or $(URL),http://127.0.0.1:8000)

evaluate: ## метрики CER/WER/NER F1 на эталонном датасете
	cd $(BACKEND) && PYTHONPATH=. $(ACTIVATE) && python -m scripts.evaluate

benchmark: ## замер скорости на этом железе (reports/benchmark.json)
	cd $(BACKEND) && PYTHONPATH=. $(ACTIVATE) && python -m scripts.benchmark --limit 5

bulk-ingest: ## пакетная потоковая загрузка папки: make bulk-ingest SRC=/path [OWNER=email] [BATCH=500]
	cd $(BACKEND) && PYTHONPATH=. $(ACTIVATE) && python -m scripts.bulk_ingest "$(SRC)" $(if $(OWNER),--owner $(OWNER),) --batch $(or $(BATCH),500) $(if $(REGISTER_ONLY),--register-only,)

scale-bench: ## масштаб: 2k→100k документов на PostgreSQL (DATABASE_URL), замер списка/FTS/векторов
	cd $(BACKEND) && PYTHONPATH=. $(ACTIVATE) && python -m scripts.scale_benchmark $(if $(SIZES),--sizes $(SIZES),)

extreme-bench: ## итерация 5: 100k/1M — float32 vs линейный int8 vs IVF int8/int4, FTS5 (≈ 15 мин, 2 CPU)
	cd $(BACKEND) && PYTHONPATH=. $(ACTIVATE) && python -m scripts.extreme_benchmark $(if $(SIZES),--sizes $(SIZES),)

search-bench: ## поиск на 1000+ документах: замер fulltext/semantic/hybrid (DOCS=, API= для HTTP)
	cd $(BACKEND) && PYTHONPATH=. $(ACTIVATE) && python -m scripts.search_benchmark --docs $(or $(DOCS),1000) $(if $(API),--api-url $(API),)

load-test: ## нагрузка: 50 документов через запущенный API (COUNT= / CONCURRENCY=)
	cd $(BACKEND) && PYTHONPATH=. $(ACTIVATE) && python -m scripts.load_test --count $(or $(COUNT),50) --concurrency $(or $(CONCURRENCY),4)

models: ## скачать локальные модели в Ollama (авто-подбор под VRAM)
	./scripts/pull_models.sh

migrate: ## применить миграции Alembic
	cd $(BACKEND) && PYTHONPATH=. $(ACTIVATE) && alembic upgrade head

docker: ## весь стек в Docker (CPU)
	docker compose up -d --build

docker-gpu: ## стек с GPU: Ollama на CUDA + VLM для OCR
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

docker-down: ## остановить контейнеры
	docker compose down

logs: ## логи backend и worker
	docker compose logs -f backend celery_worker

psql: ## консоль PostgreSQL
	docker compose exec postgres psql -U idp -d idp
