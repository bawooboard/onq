#!/usr/bin/env bash
# Ollama 설치 + 경량 모델 다운로드 (Ubuntu/Debian 기준)
set -euo pipefail

echo "[1/3] Ollama 설치"
# 공식 스크립트: https://ollama.com/install.sh
curl -fsSL https://ollama.com/install.sh | sh

echo "[2/3] 모델 다운로드 (qwen2.5:7b — 4bit ~4.7GB)"
ollama pull qwen2.5:7b

echo "[3/3] 서버 기동 확인"
ollama list
echo "완료! 서버 실행: ollama serve"
