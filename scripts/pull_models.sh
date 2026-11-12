#!/usr/bin/env bash
# Загрузка локальных моделей в Ollama (инференс полностью на вашей машине).
# Подбирает LLM под объём VRAM, если OLLAMA_MODEL не задан явно.
set -euo pipefail

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-bge-m3}"
VISION_MODEL="${OLLAMA_VISION_MODEL:-qwen2-vl:7b}"
WITH_VISION="${WITH_VISION:-0}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama не найдена. Установите: https://ollama.com/download" >&2
  exit 1
fi

if ! curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  echo "==> Запускаю ollama serve в фоне"
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  for _ in $(seq 1 30); do
    curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1 && break
    sleep 1
  done
fi

MODEL="${OLLAMA_MODEL:-}"
if [ -z "$MODEL" ]; then
  VRAM=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 0)
    VRAM=${VRAM%%.*}
  fi
  echo "==> Обнаружено VRAM: ${VRAM} МБ"
  if   [ "${VRAM:-0}" -ge 20000 ]; then MODEL="qwen2.5:14b-instruct"
  elif [ "${VRAM:-0}" -ge 7000  ]; then MODEL="qwen2.5:7b-instruct"
  elif [ "${VRAM:-0}" -ge 4500  ]; then MODEL="qwen2.5:3b-instruct"
  else MODEL="qwen2.5:1.5b-instruct"
  fi
fi

echo "==> LLM:        $MODEL"
ollama pull "$MODEL"

echo "==> Эмбеддинги: $EMBEDDING_MODEL"
ollama pull "$EMBEDDING_MODEL"

if [ "$WITH_VISION" = "1" ]; then
  echo "==> VLM (OCR сканов): $VISION_MODEL"
  ollama pull "$VISION_MODEL"
else
  echo "==> VLM пропущена (WITH_VISION=1, чтобы скачать $VISION_MODEL для OCR сканов)"
fi

cat <<EOF

Готово. Пропишите в .env:
  OLLAMA_URL=$OLLAMA_URL
  OLLAMA_MODEL=$MODEL
  EMBEDDING_MODEL=$EMBEDDING_MODEL
$( [ "$WITH_VISION" = "1" ] && echo "  OLLAMA_VISION_MODEL=$VISION_MODEL" )

Проверка: python scripts/check_env.py
EOF
