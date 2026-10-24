#!/usr/bin/env bash
# 行驶中停车/放行试验；仅更改独立配置副本，默认 preview。
set -euo pipefail
if [ "$#" -gt 1 ]; then printf '%s\n' '用法: bash run_signal_motion_20261023.sh [preview|drive]' >&2; exit 2; fi
MODE="${1:-preview}"
case "$MODE" in preview|drive) ;; *) printf '%s\n' '仅支持 preview 或 drive。' >&2; exit 2 ;; esac
cd -- "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
command -v python3 >/dev/null || { printf '%s\n' '需要车上现有的 python3。' >&2; exit 1; }
mkdir -p ./signal-motion-records
PREFIX="./signal-motion-records/$(date +%Y%m%d-%H%M%S)-$$-signal"
python3 - "$MODE" "$PREFIX" <<'PY'
import hashlib,json,math,os,sys
from pathlib import Path
mode,prefix=sys.argv[1:]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
program=Path('trackview_race_continuous_20261023')
if not program.is_file() or sha(program)!='2b22de3116e9fb323bdd89111e953034722c7d6bc640b6ae49af250dc70d8494':
    raise SystemExit('需要刚完成连续巡线的 trackview_race_continuous_20261023，文件缺失或版本不符。')
if not os.access(program,os.X_OK):raise SystemExit('请先 chmod +x ./trackview_race_continuous_20261023')
if not Path('librace_npu_backend.so').is_file():raise SystemExit('缺少车上原有 AI 适配库 librace_npu_backend.so。')
source=Path('trackview.json');config=json.loads(source.read_text(encoding='utf-8-sig'))
if config.get('diagnostics') is not True:raise SystemExit('需要 diagnostics=true；原配置未改写。')
minimum=config.get('minSpeed',.05)
if isinstance(minimum,bool) or not isinstance(minimum,(float,int)) or not math.isfinite(minimum) or not 0<minimum<=.1:
    raise SystemExit('minSpeed 必须大于零且不超过 0.10。')
for key in ['cruiseSpeed','maxSpeed','curveSpeed','degradedSpeed']:config[key]=.1
race_source=Path('race.json');race=json.loads(race_source.read_text(encoding='utf-8-sig'))
# 本轮不依赖斑马线：红牌确认后直接等待通行牌；其余比赛保护仍然有效。
race['stopAfterCross']=False
lane_path=prefix+'.json';race_path=prefix+'-race.json';meta_path=prefix+'-launch.json'
Path(lane_path).write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
Path(race_path).write_text(json.dumps(race,ensure_ascii=False,indent=2),encoding='utf-8')
command=['./'+str(program),'--config',lane_path,'--race-config',race_path]
if mode=='preview':command+=['--no-control']
record={'mode':'signal','no_control':mode=='preview','experiment':'moving stop and flow signs',
        'binary':str(program),'binary_sha256':sha(program),'source_trackview_config_sha256':sha(source),
        'trackview_config_sha256':sha(lane_path),'source_race_config_sha256':sha(race_source),
        'test_race_config':race_path,'test_race_config_sha256':sha(race_path),
        'speed_overrides_mps':.1,'stop_after_cross_override':False,'trial_timer_enabled':False,'command':command}
Path(meta_path).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print('启动记录: '+meta_path)
PY
mapfile -t COMMAND < <(python3 - "$PREFIX-launch.json" <<'PY'
import json,sys
for item in json.load(open(sys.argv[1],encoding='utf-8'))['command']:print(item)
PY
)
if [ "${#COMMAND[@]}" -eq 0 ]; then printf '%s\n' '无法读取启动记录。' >&2; exit 1; fi
if [ "$MODE" = drive ]; then
    printf '%s\n' '真实车轮控制；四项速度上限 0.10 m/s。默认 MANUAL，图像窗口按一次 a 开始。'
    printf '%s\n' '确认红牌后等待，拿走红牌不会自行放行；确认绿牌后会自动恢复行驶，无需再按 a。'
    printf '%s\n' '仅使用平整直道，车旁保留断电手段；空格停车，终端 Ctrl+C 退出。远程按键可能延迟。'
else
    printf '%s\n' 'NO CONTROL 预览，不驱动车轮；红牌直接进入等待，不要求斑马线。'
fi
exec "${COMMAND[@]}"
