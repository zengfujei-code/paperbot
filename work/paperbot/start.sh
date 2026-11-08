#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

secret=./.env.deepseek
if [[ ! -f "$secret" ]] || ! grep -q '^DEEPSEEK_API_KEY=.' "$secret"; then
  echo "DeepSeek API key file is missing or empty: $secret" >&2
  exit 1
fi
chmod 600 "$secret"
install -d -o 1000 -g 1000 -m 0770 runtime/shared runtime/logs runtime/db runtime/user_data runtime/user_data/strategies

if [[ ! -f runtime/experiment.json ]]; then
  now=$(date +%s)
  end=$((now + 2592000))
  printf '{"started_at":"%s","ends_at":"%s","virtual_wallet_usdt":100,"mode":"dry_run"}\n' \
    "$(date -u -d "@$now" +%FT%TZ)" "$(date -u -d "@$end" +%FT%TZ)" > runtime/experiment.json
  chown 1000:1000 runtime/experiment.json
fi

docker compose config --quiet
docker compose up -d
docker compose ps
