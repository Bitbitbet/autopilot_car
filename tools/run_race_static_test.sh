#!/usr/bin/env bash
# 静态决策测试专用。所有模式强制 --no-control，不接受附加运行参数。
set -euo pipefail
if [ "$#" -gt 1 ]; then
    printf '%s\n' '用法: bash run_race_static_test.sh [baseline|signal|limit]' >&2
    exit 2
fi
MODE="${1:-baseline}"
case "$MODE" in baseline|signal|limit) ;; *) printf '%s\n' '仅支持 baseline、signal、limit。' >&2; exit 2 ;; esac
cd -- "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROGRAM=./trackview_race_topology
if [ ! -x "$PROGRAM" ]; then
    printf '%s\n' '缺少可执行的 trackview_race_topology，请先上传并 chmod +x ./trackview_race_topology。' >&2
    exit 1
fi
for file in trackview.json race.json; do
    if [ ! -f "$file" ]; then printf '缺少 %s\n' "$file" >&2; exit 1; fi
done
if ! command -v python3 >/dev/null 2>&1; then
    printf '%s\n' '需要车上现有的 python3 读取配置；脚本不会自动安装软件。' >&2
    exit 1
fi
mkdir -p ./static-test-records
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
CONFIG="./static-test-records/${STAMP}-${MODE}.json"
META="./static-test-records/${STAMP}-${MODE}-launch.json"
python3 - "$MODE" "$CONFIG" "$META" <<'PY'
import hashlib,json,sys
from pathlib import Path
mode,config_path,meta_path=sys.argv[1:]
track=Path('trackview.json')
lane=json.loads(track.read_text(encoding='utf-8-sig'))
if lane.get('diagnostics') is not True:
    raise SystemExit('trackview.json 的 diagnostics 未开启，请核查配置后再测试；原配置没有改写。')
race=Path('race.json')
config=json.loads(race.read_text(encoding='utf-8-sig'))
if mode in ('signal','limit'):
    config['stopAfterCross']=False
if mode=='limit':
    config['limitedSpeed']=0.15
Path(config_path).write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
meta={'mode':mode,'no_control':True,'binary':'trackview_race_topology',
      'binary_sha256':sha('trackview_race_topology'),'trackview_config_sha256':sha(track),
      'source_race_config_sha256':sha(race),'test_race_config':config_path,
      'test_race_config_sha256':sha(config_path),
      'command':['./trackview_race_topology','--config','./trackview.json',
                 '--race-config',config_path,'--no-control']}
Path(meta_path).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
PY
printf '\n%s\n' '静态测试：强制 NO CONTROL。点击图像窗口后按一次 a 才开始决策；Esc 退出。'
printf '测试模式: %s\n配置副本: %s\n启动记录: %s\n' "$MODE" "$CONFIG" "$META"
# 不通过 tee 重定向程序输出，保留终端实时显示和原程序的信号处理。
exec "$PROGRAM" --config ./trackview.json --race-config "$CONFIG" --no-control
