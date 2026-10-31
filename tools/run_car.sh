#!/usr/bin/env bash
# 固定入口：相同程序通过参数切换预览/实跑及测试场景。
set -euo pipefail
if [ "$#" -gt 3 ]; then printf '%s\n' '用法: bash run_car.sh [preview|drive|calibrate|bypass|parking-preview|parking-drive|reverse-check|bypass-virtual|bypass-virtual-preview|mission|mission-preview|mission-lab|mission-lab-preview] [lane|signal|limit|competition] [white|black]' >&2;exit 2;fi
MODE="${1:-preview}"
PROFILE="${2:-lane}"
SURFACE="${3:-config}"
case "$MODE" in preview|drive|calibrate|bypass|parking-preview|parking-drive|reverse-check|bypass-virtual|bypass-virtual-preview|mission|mission-preview|mission-lab|mission-lab-preview) ;; *) printf '%s\n' '模式支持 preview、drive、calibrate、bypass、parking-preview、parking-drive、reverse-check。' >&2;exit 2 ;; esac
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
if mode.startswith('mission'):race['forkMaxTime']=max(12.,race.get('forkMaxTime',4.))
if mode.startswith('mission'):race['stopAfterCross']=False
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
if mode in ('bypass-virtual','bypass-virtual-preview'):
    if profile!='competition':raise SystemExit('虚拟边界绕障使用 competition 配置。')
    command+=['--virtual-boundary-bypass']
    if mode.endswith('-preview'):command+=['--no-control']
if mode.startswith('mission'):
    if profile!='competition':raise SystemExit('总控使用 competition 配置。')
    if 'lab' not in mode and not Path('ground-calibration.json').is_file():raise SystemExit('总控停车需要当前车的 ground-calibration.json。')
    if mode.endswith('-preview'):command+=['--mission-lab' if 'lab' in mode else '--mission','--no-control','--ground-calibration','ground-calibration.json']
if mode=='calibrate':command+=['--calibrate-ground','--no-control']
if mode=='reverse-check':command+=['--reverse-check']
if mode=='parking-preview':
    if not Path('ground-calibration.json').is_file():raise SystemExit('停车路径预览需要当前车辆 ground-calibration.json，请先做地面标定。')
    command+=['--parking-preview','--ground-calibration','ground-calibration.json','--no-control']
ground=Path('ground-calibration.json')
reverse_source=None
if mode in ('parking-drive','mission'):
    if profile!='competition':raise SystemExit('停车实跑请使用 parking-drive competition，保留比赛停车任务。')
    if not ground.is_file():raise SystemExit('停车实跑需要当前车辆 ground-calibration.json。')
    capability=Path('parking-reverse-check.json')
    if capability.is_file():
        confirmation=json.loads(capability.read_text(encoding='utf-8-sig'))
        if confirmation.get('reverse_verified') is not True or confirmation.get('pending',False):
            report=manifest.get('parking_control',{})
            # The current human report can complete an older unfinished pulse.
            # A newer failed/pending check still invalidates that earlier report.
            if report.get('reverse_verified_by_user') is True and capability.stat().st_mtime<report.get('reverse_report_epoch_s',0):
                reverse_source='current_project_user_report_supersedes_older_check'
            else:raise SystemExit('最新倒车记录未确认成功；请先核实该车实际后退并保存y确认。')
        else:reverse_source='saved_current_vehicle_operator_observation'
    elif manifest.get('parking_control',{}).get('reverse_verified_by_user') is True:
        reverse_source='current_project_user_report_2026_10_29'
    else:raise SystemExit('缺少当前车辆实际倒车成功的确认。')
    command+=['--parking-drive' if mode=='parking-drive' else '--mission-lab' if mode=='mission-lab' else '--mission','--reverse-verified','--ground-calibration',str(ground)]
if mode=='mission-lab':command+=['--mission-lab']
if mode=='bypass':
    if profile!='competition':raise SystemExit('绕行实验请使用 bypass competition，保留完整比赛任务优先级。')
    if not ground.is_file():raise SystemExit('请先执行 bash ./run_car.sh calibrate competition，生成当前车辆的地面标定。')
    command+=['--obstacle-bypass','--ground-calibration',str(ground)]
record={'mode':profile,'run_mode':mode,'no_control':mode in ('preview','calibrate','parking-preview','bypass-virtual-preview','mission-preview','mission-lab-preview'),'binary':'trackview_race',
        'road_surface':track.get('roadSurface','white'),
        'binary_sha256':sha('trackview_race'),'plugin_sha256':sha('librace_npu_backend.so'),
        'source_trackview_config_sha256':sha(source),'source_race_config_sha256':sha(source_race),
        'trackview_config_sha256':sha(lane_path),'test_race_config':race_path,'test_race_config_sha256':sha(race_path),
        'stop_candidate_threshold':race['stopCandidateConfidence'],'confidence':race.get('confidence',.65),
        'cross_candidate_threshold':race['crossCandidateConfidence'],'verify_crosswalk_geometry':race['verifyCrosswalkGeometry'],
        'speed_overrides_mps':{key:track[key] for key in ['cruiseSpeed','maxSpeed','curveSpeed','degradedSpeed']},
        'trial_timer_enabled':False,'command':command}
record['navigation_config_sha256']=sha(navigation) if navigation.is_file() else None
record['obstacle_planning_preview_only']=mode not in ('bypass','bypass-virtual','mission','mission-lab')
record['obstacle_route_controls_vehicle']=mode in ('bypass','bypass-virtual','mission','mission-lab')
record['bypass_hard_speed_cap_mps']=track['maxSpeed'] if mode=='bypass' else None
record['ground_calibration_sha256']=sha(ground) if ground.is_file() and mode in ('bypass','parking-drive','mission','mission-lab','mission-preview','mission-lab-preview') else None
record['sample_write_mode']='bounded_background_queue'
record['parking_geometry_preview_only']=mode=='parking-preview'
record['parking_control']={'enabled':mode=='parking-drive' or mode.startswith('mission') and 'lab' not in mode,'reverse_source':reverse_source,'bay':None,'selection_policy':'fully visible left bay, all assumed empty',
    'live_teaching_required':False,'automatic_live_mouth_geometry':True,'dwell_seconds':3,'entrance_exit_turns_integrated':mode.startswith('mission') and 'lab' not in mode}
record['reverse_check']={'operator_trigger_required':True,'speed_command_mps':-.08,'duration_s':.5,'command_is_not_measured_motion':True} if mode=='reverse-check' else None
record['mission']={'enabled':mode.startswith('mission'),'lab_single_loop':'lab' in mode,'lap_finish_source':'operator n at physical start','ground_arrow_assisted':True}
record['bypass_algorithm']='virtual_boundary_v6' if 'virtual' in mode or mode.startswith('mission') else 'metric_ground' if mode=='bypass' else 'shadow'
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
if [[ "$MODE" == mission* ]]; then
    printf '%s\n' '总控：a开始，n在真实起点确认计圈/结束，空格停，g复位，s保存当前图；preview不打开电机串口。实验室单圈用mission-lab。'
elif [[ "$MODE" == bypass-virtual* ]]; then
    printf '%s\n' '虚拟边界绕障：a开始，空格停车，preview不驱动车轮；绕过后的回线是视觉/指令距离启发式，不是实测车尾过桶证明。'
elif [ "$MODE" = reverse-check ]; then
    printf '%s\n' '倒车短测：按r才发送一次0.5秒倒车指令，之后自动停车；y确认实际后退，n记录未后退；空格/q退出。不执行自动停车任务。'
elif [ "$MODE" = parking-drive ]; then
    printf '%s\n' '停车实跑：自动找左侧完整可见车位并规划，出现黄线和绿线后按a执行，不需每次点选。g重新识别；f手动诊断；空格停。车尾不越过起始后沿，入口/出口左转尚未接入。'
elif [ "$MODE" = parking-preview ]; then
    printf '%s\n' 'NO MOTOR 停车几何预览：2号车位，41cm边长/38cm进深/45cm通道。f冻结，点近/远两个口角，p规划，s保存，q退出。'
elif [ "$MODE" = calibrate ]; then
    printf '%s\n' '地面四点标定：仅摄像头，不打开电机或AI后端。终端会显示摆点和点击顺序。'
elif [ "$MODE" = preview ]; then
    printf '%s\n' 'NO CONTROL。识别无需按a；按a只开始计算比赛状态，不驱动车轮。s保存样本，q/Esc或终端Ctrl+C退出。'
else
    printf '%s\n' '真实车轮控制。图像窗口按a启动、空格停车；终端Ctrl+C退出。远程按键可能延迟，车旁保留断电手段。'
    if [ "$MODE" = bypass ];then printf '%s\n' '标定后的直道锥桶绕行实验：初始速度0.10 m/s，最高可调10 m/s，MANUAL禁用油门。视觉里程/路线中断锁停，检查后退出重启。';fi
fi
exec "${COMMAND[@]}"
