#!/usr/bin/env bash
# Render motion-QA clips from the real composition (CPU, concurrency 4).
# usage: tools/render_clips.sh name:start-end [name:start-end ...]
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/home/win10ubuntu/.nvm/versions/node/v22.23.1/bin:$PATH
mkdir -p out/qa
for spec in "$@"; do
  name=${spec%%:*}; range=${spec#*:}
  nice -n 10 npx remotion render src/index.ts CifarJourney "out/qa/$name.mp4" --frames="$range" \
    --codec=h264 --concurrency=4 --gl=swiftshader --log=error
  echo "rendered out/qa/$name.mp4"
done
