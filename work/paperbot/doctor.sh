#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
image='ghcr.io/freqtrade/freqtrade@sha256:2ac8ba805e0ea164fd3612ed1ca623cda1fd2f7aab3d6e1b61132e49b35e7e29'
test -s ./.env.deepseek
test "$(stat -c '%a' ./.env.deepseek)" = 600
docker image inspect "$image" >/dev/null
docker compose config --quiet
docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges \
  --tmpfs /tmp -v "$PWD:/app:ro" -w /app --entrypoint python "$image" -m unittest discover -s /app/tests -v
echo "All offline safety and unit checks passed."
