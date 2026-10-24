#!/usr/bin/env bash
# 一次性落地低速短测；默认 preview，不驱动车轮。drive 也必须按 a 才启动。
set -euo pipefail
if [ "$#" -gt 1 ]; then printf '%s\n' '用法: bash run_ground_trial_20261022.sh [preview|drive]' >&2; exit 2; fi
MODE="${1:-preview}"
case "$MODE" in preview|drive) ;; *) printf '%s\n' '仅支持 preview 或 drive。' >&2; exit 2 ;; esac
cd -- "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROGRAM=./trackview_race_timedtrial
CONFIG=./trackview_low_speed_20261022.json
for file in "$PROGRAM" "$CONFIG" ./race.json; do
    if [ ! -f "$file" ]; then printf '缺少 %s\n' "$file" >&2; exit 1; fi
done
if [ ! -x "$PROGRAM" ]; then printf '%s\n' '请先 chmod +x ./trackview_race_timedtrial' >&2; exit 1; fi
mkdir -p ./low-speed-trial-records
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
META="./low-speed-trial-records/${STAMP}-baseline-launch.json"
COPY="./low-speed-trial-records/${STAMP}-baseline.json"
python3 - "$MODE" "$CONFIG" "$COPY" "$META" <<'PY'
import hashlib,json,sys
from pathlib import Path
mode,source,copy,meta=sys.argv[1:]
config=json.loads(Path(source).read_text(encoding='utf-8-sig'))
for name in ['cruiseSpeed','maxSpeed','curveSpeed','degradedSpeed']:
    value=config.get(name)
    if isinstance(value,bool) or not isinstance(value,(float,int)) or not 0<value<=.100001:
        raise SystemExit('短测配置的所有速度必须不超过 0.10 m/s：'+name)
if config.get('diagnostics') is not True:raise SystemExit('短测需要 diagnostics=true。')
Path(copy).write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
command=['./trackview_race_timedtrial','--config',copy,'--race-config','./race.json','--trial-seconds','2']
if mode=='preview':command.append('--no-control')
record={'mode':'baseline','trial_mode':mode,'experiment':'one-shot low-speed ground trial','limit_s':2,
    'binary':'trackview_race_timedtrial',
    'no_control':mode=='preview','binary_sha256':sha('./trackview_race_timedtrial'),
    'trackview_config_sha256':sha(copy),'source_race_config_sha256':sha('./race.json'),'command':command}
Path(meta).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
PY
if [ "$MODE" = drive ]; then
    printf '%s\n' '真实车轮控制：默认 MANUAL。准备好直道和车旁断电手段后，按一次 a 开始。约两秒后请求零速并锁定停车。'
    exec "$PROGRAM" --config "$COPY" --race-config ./race.json --trial-seconds 2
else
    printf '%s\n' '预览 NO CONTROL：不驱动车轮；按 a 后也会检查两秒定时退出。'
    exec "$PROGRAM" --config "$COPY" --race-config ./race.json --trial-seconds 2 --no-control
fi
