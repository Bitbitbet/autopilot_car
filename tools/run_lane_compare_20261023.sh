#!/usr/bin/env bash
# 普通赛道连续巡线对照；默认只预览，必须明确写 drive 才允许车轮控制。
set -euo pipefail
if [ "$#" -gt 2 ]; then
    printf '%s\n' '用法: bash run_lane_compare_20261023.sh [race|v6] [preview|drive]' >&2; exit 2
fi
VERSION="${1:-race}"
MODE="${2:-preview}"
case "$VERSION" in race|v6) ;; *) printf '%s\n' '版本仅支持 race 或 v6。' >&2; exit 2 ;; esac
case "$MODE" in preview|drive) ;; *) printf '%s\n' '模式仅支持 preview 或 drive。' >&2; exit 2 ;; esac
cd -- "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
command -v python3 >/dev/null || { printf '%s\n' '需要车上现有的 python3。' >&2; exit 1; }
mkdir -p ./lane-compare-records
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
PREFIX="./lane-compare-records/${STAMP}-${VERSION}"
python3 - "$VERSION" "$MODE" "$PREFIX" <<'PY'
import hashlib,json,math,os,sys
from pathlib import Path
version,mode,prefix=sys.argv[1:]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
if version=='race':
    program=Path('trackview_race_continuous_20261023')
    if not program.is_file() or sha(program)!='2b22de3116e9fb323bdd89111e953034722c7d6bc640b6ae49af250dc70d8494':
        raise SystemExit('新版程序缺失或版本不符，请上传本次包中的 trackview_race_continuous_20261023。')
    if not Path('librace_npu_backend.so').is_file():raise SystemExit('缺少现有 AI 适配库 librace_npu_backend.so。')
else:
    candidates=[Path(n) for n in ['trackview_fixed_v6_max10_defaults','trackview_fixed_v6_max10_default'] if Path(n).is_file()]
    if not candidates:raise SystemExit('未找到原 v6；请保留原文件名，不需要替换它。')
    if len({sha(p) for p in candidates})>1:raise SystemExit('发现两个不同的 v6，请先核对版本。')
    program=candidates[0]
if not os.access(program,os.X_OK):raise SystemExit('程序没有执行权限，请先 chmod +x ./'+str(program))
source=Path('trackview.json')
config=json.loads(source.read_text(encoding='utf-8-sig'))
if config.get('diagnostics') is not True:raise SystemExit('原配置需要 diagnostics=true；脚本没有改写原配置。')
minimum=config.get('minSpeed',.05)
if isinstance(minimum,bool) or not isinstance(minimum,(float,int)) or not math.isfinite(minimum) or not 0<minimum<=.1:
    raise SystemExit('minSpeed 必须大于零且不超过 0.10，请先核对配置。')
# 两版使用相同的现场巡线参数，仅覆盖四个速度项，测试中请勿按调速键。
for key in ['cruiseSpeed','maxSpeed','curveSpeed','degradedSpeed']:config[key]=.1
lane_path=prefix+'.json'
Path(lane_path).write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
command=['./'+str(program),'--config',lane_path]
record={'mode':version,'no_control':mode=='preview','experiment':'ordinary lane continuous comparison',
        'binary':str(program),'binary_sha256':sha(program),'source_trackview_config_sha256':sha(source),
        'trackview_config_sha256':sha(lane_path),'speed_overrides_mps':.1,'trial_timer_enabled':False}
if version=='race':
    race=Path('race.json')
    content=race.read_text(encoding='utf-8-sig'); json.loads(content)
    race_path=prefix+'-race.json';Path(race_path).write_text(content,encoding='utf-8')
    record.update(source_race_config_sha256=sha(race),test_race_config=race_path,test_race_config_sha256=sha(race_path))
    command+=['--race-config',race_path]
if mode=='preview':command+=['--no-control']
record['command']=command
Path(prefix+'-launch.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print('启动记录: '+prefix+'-launch.json')
PY
mapfile -t COMMAND < <(python3 - "$PREFIX-launch.json" <<'PY'
import json,sys
for item in json.load(open(sys.argv[1],encoding='utf-8'))['command']:print(item)
PY
)
if [ "${#COMMAND[@]}" -eq 0 ]; then printf '%s\n' '无法读取启动记录。' >&2; exit 1; fi
printf '版本: %s；模式: %s；四项速度上限均为 0.10 m/s；连续运行，没有两秒定时退出。\n' "$VERSION" "$MODE"
if [ "$MODE" = drive ]; then
    printf '%s\n' '真实车轮控制，默认 MANUAL。点击图像窗口后按一次 a 启动。空格停车；终端 Ctrl+C 请求退出。'
    printf '%s\n' '先清空赛道路牌和杂物；车旁保留断电手段，远程按键可能延迟。不要在本轮按调速键。'
else
    printf '%s\n' 'NO CONTROL，仅预览和计算决策，不驱动车轮。'
fi
exec "${COMMAND[@]}"
