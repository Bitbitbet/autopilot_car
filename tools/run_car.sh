#!/usr/bin/env bash
# 固定入口：相同程序通过参数切换预览/实跑及测试场景。
set -euo pipefail
if [ "$#" -gt 3 ]; then printf '%s\n' '用法: bash run_car.sh [preview|drive|calibrate|bypass] [lane|signal|limit|competition] [white|black]' >&2;exit 2;fi
MODE="${1:-preview}"
PROFILE="${2:-lane}"
SURFACE="${3:-config}"
case "$MODE" in preview|drive|calibrate|bypass) ;; *) printf '%s\n' '模式支持 preview、drive、calibrate、bypass。' >&2;exit 2 ;; esac
case "$PROFILE" in lane|signal|limit|competition) ;; *) printf '%s\n' '未知测试场景。' >&2;exit 2 ;; esac
case "$SURFACE" in config|white|black) ;; *) printf '%s\n' '路面模式支持 white、black。' >&2;exit 2 ;; esac
cd -- "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
command -v python3 >/dev/null || { printf '%s\n' '需要车上现有的 python3。' >&2;exit 1; }
mkdir -p ./run-records
PREFIX="./run-records/$(date +%Y%m%d-%H%M%S)-$$-${PROFILE}"
python3 - "$MODE" "$PROFILE" "$PREFIX" "$SURFACE" <<'PY'
import hashlib,json,math,os,sys
from pathlib import Path
mode,profile,prefix,surface=sys.argv[1:]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads(Path('release-manifest.json').read_text(encoding='utf-8-sig'))
for name in ['trackview_race','librace_npu_backend.so']:
    if not Path(name).is_file() or sha(name)!=manifest['files'][name]:
        raise SystemExit('主程序/AI适配库与交付清单不匹配，请上传race_release中配套的文件：'+name)
if not os.access('trackview_race',os.X_OK):raise SystemExit('请先 chmod +x ./trackview_race')
source=Path('trackview.json');source_race=Path('race.json')
track=json.loads(source.read_text(encoding='utf-8-sig'));race=json.loads(source_race.read_text(encoding='utf-8-sig'))
if surface!='config':track['roadSurface']=surface
if track.get('roadSurface','white') not in ('white','black'):raise SystemExit('roadSurface 必须是 white 或 black。')
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
race.setdefault('verifyCrosswalkGeometry',True)
race.setdefault('crossCandidateConfidence',.10)
lane_path=prefix+'.json';race_path=prefix+'-race.json'
Path(lane_path).write_text(json.dumps(track,ensure_ascii=False,indent=2),encoding='utf-8')
Path(race_path).write_text(json.dumps(race,ensure_ascii=False,indent=2),encoding='utf-8')
command=['./trackview_race','--config',lane_path,'--race-config',race_path]
navigation=Path('navigation.json')
if navigation.is_file():command+=['--navigation-config',str(navigation)]
if mode=='preview':command+=['--no-control']
if mode=='calibrate':command+=['--calibrate-ground','--no-control']
ground=Path('ground-calibration.json')
if mode=='bypass':
    if profile!='competition':raise SystemExit('绕行实验请使用 bypass competition，保留完整比赛任务优先级。')
    if not ground.is_file():raise SystemExit('请先执行 bash ./run_car.sh calibrate competition，生成当前车辆的地面标定。')
    command+=['--obstacle-bypass','--ground-calibration',str(ground)]
record={'mode':profile,'run_mode':mode,'no_control':mode in ('preview','calibrate'),'binary':'trackview_race',
        'road_surface':track.get('roadSurface','white'),
        'binary_sha256':sha('trackview_race'),'plugin_sha256':sha('librace_npu_backend.so'),
        'source_trackview_config_sha256':sha(source),'source_race_config_sha256':sha(source_race),
        'trackview_config_sha256':sha(lane_path),'test_race_config':race_path,'test_race_config_sha256':sha(race_path),
        'stop_candidate_threshold':race['stopCandidateConfidence'],'confidence':race.get('confidence',.65),
        'cross_candidate_threshold':race['crossCandidateConfidence'],'verify_crosswalk_geometry':race['verifyCrosswalkGeometry'],
        'speed_overrides_mps':{key:track[key] for key in ['cruiseSpeed','maxSpeed','curveSpeed','degradedSpeed']},
        'trial_timer_enabled':False,'command':command}
record['navigation_config_sha256']=sha(navigation) if navigation.is_file() else None
record['obstacle_planning_preview_only']=mode!='bypass'
record['obstacle_route_controls_vehicle']=mode=='bypass'
record['bypass_hard_speed_cap_mps']=track['maxSpeed'] if mode=='bypass' else None
record['ground_calibration_sha256']=sha(ground) if mode=='bypass' else None
record['sample_write_mode']='bounded_background_queue'
Path(prefix+'-launch.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print('启动记录: '+prefix+'-launch.json')
print('路面模式: '+track.get('roadSurface','white')+'；寻路二值图统一为白色道路。')
PY
mapfile -t COMMAND < <(python3 - "$PREFIX-launch.json" <<'PY'
import json,sys
for item in json.load(open(sys.argv[1],encoding='utf-8'))['command']:print(item)
PY
)
if [ "${#COMMAND[@]}" -eq 0 ];then printf '%s\n' '无法读取启动记录。' >&2;exit 1;fi
printf '模式: %s；场景: %s；初始巡航/弯道/降级速度0.10 m/s；最高可调上限10 m/s；默认MANUAL。\n' "$MODE" "$PROFILE"
if [ "$MODE" = calibrate ]; then
    printf '%s\n' '地面四点标定：仅摄像头，不打开电机或AI后端。终端会显示摆点和点击顺序。'
elif [ "$MODE" = preview ]; then
    printf '%s\n' 'NO CONTROL。识别无需按a；按a只开始计算比赛状态，不驱动车轮。s保存样本，q/Esc或终端Ctrl+C退出。'
else
    printf '%s\n' '真实车轮控制。图像窗口按a启动、空格停车；终端Ctrl+C退出。远程按键可能延迟，车旁保留断电手段。'
    if [ "$MODE" = bypass ];then printf '%s\n' '标定后的直道锥桶绕行实验：初始速度0.10 m/s，最高可调10 m/s，MANUAL禁用油门。视觉里程/路线中断锁停，检查后退出重启。';fi
fi
exec "${COMMAND[@]}"
