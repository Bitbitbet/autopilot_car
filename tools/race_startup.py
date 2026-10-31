import hashlib,json,math,os,sys
from pathlib import Path
mode,profile,prefix,surface,binary_name=sys.argv[1:]
allowed=('preview','drive','calibrate','bypass','parking-preview','parking-drive','reverse-check','bypass-virtual','bypass-virtual-preview','mission','mission-preview','mission-lab','mission-lab-preview')
if mode not in allowed or profile not in ('lane','signal','limit','competition') or surface not in ('config','white','black'):
    print('未知模式、场景或路面参数。',file=sys.stderr)
    raise SystemExit(2)
if binary_name not in ('racing','trackview_race'):raise SystemExit('请使用固定程序名 racing 或 trackview_race。')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads(Path('release-manifest.json').read_text(encoding='utf-8-sig'))
for name in [binary_name,'librace_npu_backend.so']:
    if not Path(name).is_file() or sha(name)!=manifest['files'].get(name,manifest['files'].get('trackview_race') if name=='racing' else None):
        raise SystemExit('主程序/AI适配库与交付清单不匹配，请上传race_release中配套的文件：'+name)
if not os.access(binary_name,os.X_OK):raise SystemExit('请先 chmod +x ./'+binary_name)
source=Path('trackview.json');source_race=Path('race.json')
track=json.loads(source.read_text(encoding='utf-8-sig'));race=json.loads(source_race.read_text(encoding='utf-8-sig'))
def native_pixels(config, distances, gains=False):
    reference=config.get('imageWidth',320)
    if reference not in (320,640) or config.get('imageHeight',reference*3//4)!=reference*3//4:
        raise SystemExit('配置图像坐标必须为320×240旧口径或640×480。')
    if reference==320:
        for key in distances:
            if key in config:config[key]*=2
        if gains:
            for key in ('runP1','turnD'):
                if key in config:config[key]/=2
            if 'runP2' in config:config['runP2']/=4
    config.update(imageWidth=640,imageHeight=480)
native_pixels(track,('curveLookaheadRow','borderMargin','minSpan','minWidth','maxWidth','rowCutUp','rowCutBottom','maxErrorRate'),True)
native_pixels(race,('minBranchWidth','minBranchGap','triggerRow','obstacleRow','crossPassedRow'))
if surface!='config':track['roadSurface']=surface
if track.get('roadSurface','white') not in ('white','black'):raise SystemExit('roadSurface 必须是 white 或 black。')
minimum=track.get('minSpeed',.05)
if isinstance(minimum,bool) or not isinstance(minimum,(int,float)) or not math.isfinite(minimum) or not 0<minimum<=.1:
    raise SystemExit('当前低速测试要求 minSpeed 大于0且不超过0.10。')
if track.get('diagnostics') is not True:raise SystemExit('需要 diagnostics=true；原配置没有改写。')
for key in ['cruiseSpeed','curveSpeed','degradedSpeed']:track[key]=.1
track['maxSpeed']=10.0
race.setdefault('parkingSpeed',.30)
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
command=['./'+binary_name,'--config',lane_path,'--race-config',race_path]
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
record={'mode':profile,'run_mode':mode,'no_control':mode in ('preview','calibrate','parking-preview','bypass-virtual-preview','mission-preview','mission-lab-preview'),'binary':binary_name,
        'road_surface':track.get('roadSurface','white'),
        'binary_sha256':sha(binary_name),'plugin_sha256':sha('librace_npu_backend.so'),
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
record['parking_control']={'enabled':mode=='parking-drive' or mode.startswith('mission') and 'lab' not in mode,'reverse_source':reverse_source,'bay':None,'selection_policy':'nearby fully visible left or right bay, all assumed empty',
    'parking_area_speed_cap_mps':race['parkingSpeed'],'exit_gate_release_restores_cruise':True,
    'live_teaching_required':False,'automatic_live_mouth_geometry':True,'dwell_seconds':3,'entrance_exit_turns_integrated':mode.startswith('mission') and 'lab' not in mode}
record['reverse_check']={'operator_trigger_required':True,'speed_command_mps':-.08,'duration_s':.5,'command_is_not_measured_motion':True} if mode=='reverse-check' else None
record['mission']={'enabled':mode.startswith('mission'),'lab_single_loop':'lab' in mode,'lap_finish_source':'post-parking FLOW or operator n; finish operator n','ground_arrow_assisted':True,'ground_arrow_turn_trigger':'parking_exit_stage_only'}
record['processing_resolution']=[640,480]
record['backend_abi']=5
record['bypass_algorithm']='virtual_boundary_master640' if 'virtual' in mode or mode.startswith('mission') else 'metric_ground' if mode=='bypass' else 'shadow'
Path(prefix+'-launch.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print('启动记录: '+prefix+'-launch.json')
print('路面模式: '+track.get('roadSurface','white')+'；寻路二值图统一为白色道路。')
print('模式: '+mode+'；场景: '+profile+'；初始巡航/弯道/降级速度0.10 m/s；最高可调上限10 m/s；默认MANUAL。')
if mode.startswith('mission'):
    print('总控：a开始，n在真实起点确认计圈/结束，空格停，g复位，s保存当前图；preview不打开电机串口。实验室单圈用mission-lab。')
elif mode.startswith('bypass-virtual'):
    print('虚拟边界绕障：a开始，空格停车；preview不驱动车轮。')
elif mode=='parking-drive':
    print('停车：自动选择完整可见的左侧或右侧车位；a执行，g重新识别，空格停。')
elif mode in ('preview','calibrate','parking-preview'):
    print('NO MOTOR；按模式进行预览、标定或保存样本。')
elif mode=='reverse-check':
    print('倒车短测：按r才发送一次0.5秒倒车指令，之后自动停车；y确认后退，n记录未后退。')
else:
    print('普通巡线与交通规则：图像窗口a开始，空格停，q退出。')
