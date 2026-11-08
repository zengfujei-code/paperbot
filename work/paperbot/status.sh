#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "=== 30-day paper experiment ==="
cat runtime/experiment.json 2>/dev/null || echo "not started"
echo
echo "=== containers ==="
docker compose ps
echo
echo "=== latest AI signal (API key is never shown) ==="
if [[ -f runtime/shared/signal.json ]]; then
  docker run --rm --network none -v "$PWD/runtime/shared:/shared:ro" --entrypoint python \
    ghcr.io/freqtrade/freqtrade@sha256:2ac8ba805e0ea164fd3612ed1ca623cda1fd2f7aab3d6e1b61132e49b35e7e29 \
    -c 'import json; s=json.load(open("/shared/signal.json")); print(json.dumps({k:s.get(k) for k in ("action","confidence","reason","status","generated_at_utc","expires_at")}, ensure_ascii=False, indent=2))'
else
  echo "no signal yet"
fi
echo
echo "=== recent service logs ==="
docker compose logs --tail 12 advisor freqtrade
