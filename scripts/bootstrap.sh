#!/usr/bin/env bash
# Установка IDP-платформы на локальной машине (Linux / macOS; Windows — через WSL2).
#
#   ./scripts/bootstrap.sh              # базовая установка (без скачивания LLM)
#   ./scripts/bootstrap.sh --models     # + скачать локальные модели в Ollama
#   ./scripts/bootstrap.sh --help
#
# Что делает (идемпотентно, повторный запуск безопасен):
#   1. проверяет Python 3.11+ и Node.js 20+;
#   2. создаёт backend/.venv и ставит зависимости backend;
#   3. создаёт .env из .env.example и генерирует уникальный SECRET_KEY;
#   4. применяет миграции БД (SQLite по умолчанию, файл backend/data/idp.db);
#   5. генерирует демо-документы в samples/;
#   6. ставит зависимости frontend (npm);
#   7. (с --models) скачивает LLM и модель эмбеддингов через Ollama.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WITH_MODELS=0
for arg in "$@"; do
  case "$arg" in
    --models) WITH_MODELS=1 ;;
    -h|--help)
      sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Неизвестный параметр: $arg (см. --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\n==> %s\n' "$*"; }
warn() { printf '    [!] %s\n' "$*"; }
die()  { printf '\n[x] %s\n' "$*" >&2; exit 1; }

# ---------- 1. Проверка инструментов ----------
say "Проверка окружения"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
  done
fi
[ -n "$PYTHON" ] || die "Python не найден. Установите Python 3.11+ (https://www.python.org/downloads/)."
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
  || die "Нужен Python 3.11+, найден $("$PYTHON" -V 2>&1). Укажите путь: PYTHON=/usr/bin/python3.12 ./scripts/bootstrap.sh"
echo "    Python: $("$PYTHON" -V 2>&1)"

HAS_NPM=0
if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
  NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
  if [ "$NODE_MAJOR" -ge 20 ]; then
    HAS_NPM=1
    echo "    Node.js: $(node -v)"
  else
    warn "Найден Node.js $(node -v), нужен 20+. Фронтенд не будет установлен."
  fi
else
  warn "Node.js 20+ не найден. Backend установится, фронтенд — нет (https://nodejs.org)."
fi

OLLAMA_OK=0
if command -v ollama >/dev/null 2>&1; then
  OLLAMA_OK=1
  echo "    Ollama: найдена"
else
  warn "Ollama не найдена: система будет работать на эвристическом fallback без LLM (https://ollama.com)."
fi

# ---------- 2. Backend ----------
say "Backend: виртуальное окружение backend/.venv"
if [ ! -x "$ROOT/backend/.venv/bin/python" ]; then
  "$PYTHON" -m venv "$ROOT/backend/.venv"
fi
VENV_PY="$ROOT/backend/.venv/bin/python"
"$VENV_PY" -m pip install --quiet --upgrade pip wheel
"$VENV_PY" -m pip install --quiet -r backend/requirements.txt
echo "    зависимости установлены"

# ---------- 3. .env ----------
say "Конфигурация .env"
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "    создан .env из .env.example"
else
  echo "    .env уже существует — не перезаписываю"
fi
if grep -q '^SECRET_KEY=change-me' .env; then
  NEW_KEY="$("$VENV_PY" -c 'import secrets; print(secrets.token_hex(32))')"
  "$VENV_PY" - "$NEW_KEY" <<'PY'
import pathlib, re, sys
p = pathlib.Path(".env")
p.write_text(re.sub(r"^SECRET_KEY=.*$", f"SECRET_KEY={sys.argv[1]}", p.read_text(), flags=re.M))
PY
  echo "    SECRET_KEY сгенерирован случайно"
fi

# ---------- 4. Миграции БД ----------
say "База данных: миграции Alembic"
(cd backend && PYTHONPATH=. "$VENV_PY" -m alembic upgrade head) \
  || die "Миграции не применились. См. вывод выше."

# ---------- 5. Демо-данные ----------
say "Демо-документы и эталонная разметка (samples/)"
(cd backend && PYTHONPATH=. "$VENV_PY" -m scripts.generate_samples --out ../samples >/dev/null)
echo "    samples/: $(find samples -maxdepth 1 -type f | wc -l | tr -d ' ') файлов"

# ---------- 6. Frontend ----------
say "Frontend: npm"
if [ "$HAS_NPM" = 1 ]; then
  if [ -f frontend/package-lock.json ]; then
    (cd frontend && npm ci --no-audit --no-fund)
  else
    (cd frontend && npm install --no-audit --no-fund)
  fi
  echo "    frontend/node_modules готов"
else
  warn "пропущено (нужен Node.js 20+)"
fi

# ---------- 7. Ollama / модели ----------
say "Локальные LLM (Ollama)"
if [ "$WITH_MODELS" = 1 ]; then
  if [ "$OLLAMA_OK" = 1 ]; then
    ./scripts/pull_models.sh
  else
    die "Флаг --models требует установленной Ollama (https://ollama.com)."
  fi
else
  echo "    модели не скачивались (чтобы скачать: ./scripts/bootstrap.sh --models)"
fi

# ---------- Итог ----------
"$VENV_PY" scripts/check_env.py || true

cat <<'EOF'

============================================================
 Установка завершена.

 Запуск (два терминала или одна команда):
   make dev            backend :8000 и frontend :5173
   make api            только backend  -> http://localhost:8000/api/docs
   make web            только frontend -> http://localhost:5173

 Вход:  demo@idp-project.ru / demo12345   (создаётся при первом запуске)
 Проверка: make test   |   make evaluate   |   make check
 Пароль демо-пользователя и SECRET_KEY при продуктивном
 использовании обязательно замените (см. README, раздел «Безопасность»).
============================================================
EOF
