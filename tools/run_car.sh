#!/usr/bin/env bash
# 固定入口：相同程序通过参数切换预览/实跑及测试场景。
set -euo pipefail
if [ "$#" -gt 2 ]; then printf '%s\n' '用法: bash run_car.sh [preview|drive] [lane|signal|limit|competition]' >&2;exit 2;fi
MODE="${1:-preview}"
PROFILE="${2:-lane}"
case "$MODE" in preview|drive) ;; *) printf '%s\n' '模式仅支持 preview 或 drive。' >&2;exit 2 ;; esac
case "$PROFILE" in lane|signal|limit|competition) ;; *) printf '%s\n' '未知测试场景。' >&2;exit 2 ;; esac
cd -- "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
command -v python3 >/dev/null || { printf '%s\n' '需要车上现有的 python3。' >&2;exit 1; }
mkdir -p ./run-records
PREFIX="./run-records/$(date +%Y%m%d-%H%M%S)-$$-${PROFILE}"
python3 - "$MODE" "$PROFILE" "$PREFIX" <<'PY'
import hashlib,json,math,os,sys
from pathlib import Path
mode,profile,prefix=sys.argv[1:]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads(Path('release-manifest.json').read_text(encoding='utf-8-sig'))
for name in ['trackview_race','librace_npu_backend.so']:
    if not Path(name).is_file() or sha(name)!=manifest['files'][name]:
        raise SystemExit('主程序/AI适配库与交付清单不匹配，请上传同一交付包中的两个文件：'+name)
if not os.access('trackview_race',os.X_OK):raise SystemExit('请先 chmod +x ./trackview_race')
source=Path('trackview.json');source_race=Path('race.json')
track=json.loads(source.read_text(encoding='utf-8-sig'));race=json.loads(source_race.read_text(encoding='utf-8-sig'))
minimum=track.get('minSpeed',.05)
if isinstance(minimum,bool) or not isinstance(minimum,(int,float)) or not math.isfinite(minimum) or not 0<minimum<=.1:
    raise SystemExit('当前低速测试要求 minSpeed 大于0且不超过0.10。')
if track.get('diagnostics') is not True:raise SystemExit('需要 diagnostics=true；原配置没有改写。')
for key in ['cruiseSpeed','curveSpeed','degradedSpeed']:track[key]=.1
track['maxSpeed']=10.0
# 正式比赛的规则仍保留；signal/limit 为当前实验室的独立测试副本。
if profile in ('signal','limit'):race['stopAfterCross']=False
if profile=='limit':race['limitedSpeed']=max(minimum,.08)
race.setdefault('verifyStopCandidates',True)
race.setdefault('stopCandidateConfidence',.35)
lane_path=prefix+'.json';race_path=prefix+'-race.json'
Path(lane_path).write_text(json.dumps(track,ensure_ascii=False,indent=2),encoding='utf-8')
Path(race_path).write_text(json.dumps(race,ensure_ascii=False,indent=2),encoding='utf-8')
command=['./trackview_race','--config',lane_path,'--race-config',race_path]
if mode=='preview':command+=['--no-control']
record={'mode':profile,'run_mode':mode,'no_control':mode=='preview','binary':'trackview_race',
        'binary_sha256':sha('trackview_race'),'plugin_sha256':sha('librace_npu_backend.so'),
        'source_trackview_config_sha256':sha(source),'source_race_config_sha256':sha(source_race),
        'trackview_config_sha256':sha(lane_path),'test_race_config':race_path,'test_race_config_sha256':sha(race_path),
        'stop_candidate_threshold':race['stopCandidateConfidence'],'confidence':race.get('confidence',.65),
        'speed_overrides_mps':{key:track[key] for key in ['cruiseSpeed','maxSpeed','curveSpeed','degradedSpeed']},
        'trial_timer_enabled':False,'command':command}
Path(prefix+'-launch.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print('启动记录: '+prefix+'-launch.json')
PY
mapfile -t COMMAND < <(python3 - "$PREFIX-launch.json" <<'PY'
import json,sys
for item in json.load(open(sys.argv[1],encoding='utf-8'))['command']:print(item)
PY
)
if [ "${#COMMAND[@]}" -eq 0 ];then printf '%s\n' '无法读取启动记录。' >&2;exit 1;fi
printf '模式: %s；场景: %s；初始巡航/弯道/降级速度0.10 m/s；最高可调上限10 m/s；默认MANUAL。\n' "$MODE" "$PROFILE"
if [ "$MODE" = preview ]; then
    printf '%s\n' 'NO CONTROL。识别无需按a；按a只开始计算比赛状态，不驱动车轮。s保存样本，q/Esc或终端Ctrl+C退出。'
else
    printf '%s\n' '真实车轮控制。图像窗口按a启动、空格停车；终端Ctrl+C退出。远程按键可能延迟，车旁保留断电手段。'
fi
exec "${COMMAND[@]}"
