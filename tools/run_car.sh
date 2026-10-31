#!/usr/bin/env bash
# Same preparation implementation is embedded in both native entrypoints.
set -euo pipefail
if [ "$#" -gt 3 ];then printf '%s\n' '用法: bash run_car.sh [模式] [场景] [white|black]' >&2;exit 2;fi
cd -- "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec ./trackview_race --launch "${1:-preview}" "${2:-lane}" "${3:-config}"
