#!/bin/sh
# Serve the installed DiffusionGemma-26B-A4B through the launcher.
set -eu
cd "$(dirname -- "$0")"
exec ./richengine serve --model google/diffusiongemma-26B-A4B-it "$@"
