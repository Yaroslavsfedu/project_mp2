#!/usr/bin/env bash
# Установка IDP-платформы на локальной машине (Linux / macOS; Windows — WSL2).
# Создаёт venv, ставит зависимости, готовит демо-данные и .env.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

echo "==> Проверка окружения"
"$PYTHON" scripts/check_env.py || true

echo
echo "==> Backend: виртуальное окружение .venv"
if [ ! -d "backend/.venv" ]; then
  "$PYTHON" -m venv backend/.venv
fi
# shellcheck disable=SC1091
source backend/.venv/bin/activate
python -m pip install --upgrade pip wheel >/dev/null
pip install -r backend/requirements.txt

echo
echo "==> .env"
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "    создан .env из .env.example"
else
  echo "    .env уже существует — не перезаписываю"
fi

echo
echo "==> Демо-документы и эталонная разметка"
(cd backend && PYTHONPATH=. python -m scripts.generate_samples --out ../samples >/dev/null)
echo "    samples/: $(ls samples | wc -l | tr -d ' ') файлов"

echo
echo "==> Frontend: npm install"
if command -v npm >/dev/null 2>&1; then
  (cd frontend && npm install --no-audit --no-fund)
else
  echo "    [!] npm не найден — установите Node.js 20+ и выполните: cd frontend && npm install"
fi

echo
echo "==> Локальные модели (опционально)"
if command -v ollama >/dev/null 2>&1; then
  echo "    Ollama найдена. Загрузить модели:  ./scripts/pull_models.sh"
else
  echo "    Ollama не найдена. Без неё работает эвристический fallback-движок."
  echo "    Установка: https://ollama.com  → затем ./scripts/pull_models.sh"
fi

cat <<'EOF'

Готово. Запуск:

  make dev            # backend :8000 + frontend :5173 одновременно
  # либо в двух терминалах:
  make api            # FastAPI + Swagger на http://localhost:8000/api/docs
  make web            # Vite на http://localhost:5173

Демо-доступ:  demo@idp-project.ru / demo12345
Проверка:     make check | make test | make evaluate
EOF
