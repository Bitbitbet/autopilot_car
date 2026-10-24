"""固定图片 NPU 诊断：复制模型到隔离目录，不打开相机或车辆控制设备。"""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def protected_snapshot(source):
    paths=list((source/'res/models/yolov3_mobilenet_v1').glob('*'))
    paths+=list((source/'res/models').glob('config_ppnc*.json'))
    paths+=list((source/'res/calibration').glob('*'))
    paths+=[source/'bin'/name for name in ['trackview.json','race.json','trackview_race_continuous_20261023','librace_npu_backend.so']]
    return {str(p.relative_to(source)):sha(p) for p in paths if p.is_file()}

def prepare(source,root):
    # 所有新增文件必须落在明确的新诊断目录里，不允许覆盖已有准备记录。
    if root.resolve().parent!=(source/'bin').resolve() or not root.name.startswith('recognition-probe-'):
        raise RuntimeError('Probe directory must be a dedicated child of vehicle bin')
    if (root/'manifest.json').exists():raise RuntimeError('Probe already prepared; preserve the previous result')
    (root/'bin').mkdir(parents=True,exist_ok=True)
    model=root/'res/models/yolov3_mobilenet_v1';model.parent.mkdir(parents=True,exist_ok=True)
    original=source/'res/models/yolov3_mobilenet_v1'
    total=sum(p.stat().st_size for p in original.rglob('*') if p.is_file())
    if total>150*1024*1024:raise RuntimeError('Unexpected model directory size')
    if shutil.disk_usage(root).free<total+30*1024*1024:raise RuntimeError('Insufficient free space for isolated model copy')
    shutil.copytree(str(original),str(model))
    calibration=source/'res/calibration'
    if calibration.is_dir():shutil.copytree(str(calibration),str(root/'res/calibration'))
    for name in ['config_ppncnna.json','config_ppncnms.json']:
        config=json.loads((source/'res/models'/name).read_text())
        config['model_dir']='../res/models/yolov3_mobilenet_v1'
        (model.parent/name).write_text(json.dumps(config,indent=2))
    images=root/'images';images.mkdir()
    rows=[]
    sample=source/'bin/trackview_logs/1792746339640-6299'
    for n in range(1,11):
        path=sample/('ai-sample-'+str(n))/'camera.png'
        if not path.is_file():raise RuntimeError('Missing saved sample '+str(n))
        target=images/('sample-'+str(n)+'.png');shutil.copy2(str(path),str(target))
        rows.append({'id':'sample-'+str(n),'path':str(target),'source_sha256':sha(path)})
    negatives=sorted((source/'bin/trackview_logs/1792742773230-23287/event-1').glob('*-camera.jpg'))
    for n,path in enumerate(negatives[::max(len(negatives)-1,1)][:2],1):
        target=images/('negative-'+str(n)+'.jpg');shutil.copy2(str(path),str(target))
        rows.append({'id':'negative-'+str(n),'path':str(target),'source_sha256':sha(path)})
    (root/'manifest.json').write_text(json.dumps({'images':rows,'repeats':3},indent=2))
    return rows

def main():
    root=Path(__file__).resolve().parent
    if sys.argv[1:]==['--preview']:
        source=Path('/root/track-trial-20261010')
        if root.parent!=source/'bin' or not root.name.startswith('recognition-probe-'):
            raise RuntimeError('Unexpected diagnostic preview location')
        if not (root/'probe-run.json').is_file():raise RuntimeError('Run saved-image diagnostic first')
        os.chdir(str(root/'bin'))
        executable=root/'bin/stop_candidate_preview'
        os.execv(str(executable),[str(executable)])
    if sys.argv[1:]==['--infer']:
        os.chdir(str(root/'bin'))
        # 使用带同一套 C++ 运行库的独立进程，避免 Python/SDK 动态加载的 ABI 混用。
        executable=root/'bin/race_image_probe'
        os.execv(str(executable),[str(executable),str(root/'manifest.json'),str(root/'probe.jsonl')])
    if sys.argv[1:]:raise RuntimeError('No probe options are accepted')
    source=Path('/root/track-trial-20261010')
    if root.parent!=(source/'bin') or not root.name.startswith('recognition-probe-'):
        raise RuntimeError('Unexpected saved-image diagnostic location')
    # 不和任何现有摄像头/驾驶程序争用 NPU。
    active=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:comm=(entry/'comm').read_text().strip()
        except OSError:continue
        if comm.startswith(('trackview','icar','camera','detection_tool')):active.append((entry.name,comm))
    if active:raise RuntimeError('Exit vehicle/camera program first: '+str(active))
    before=protected_snapshot(source)
    rows=prepare(source,root)
    record={'no_serial_or_camera':True,'input_count':len(rows),'original_files_before':before}
    try:
        completed=subprocess.run([sys.executable,str(root/'run_saved_image_probe.py'),'--infer'],
                                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        (root/'probe-console.txt').write_bytes(completed.stdout)
        record['returncode']=completed.returncode
    except subprocess.TimeoutExpired as error:
        (root/'probe-console.txt').write_bytes(error.stdout or b'')
        record.update(returncode=124,error='NPU probe timeout; subprocess terminated')
    finally:
        after=protected_snapshot(source)
        record['original_files_after']=after
        record['original_files_unchanged']=before==after
        (root/'probe-run.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({'root':str(root),'returncode':record['returncode'],
                      'original_files_unchanged':record['original_files_unchanged'],'input_count':len(rows)}))
    return record['returncode'] if record['original_files_unchanged'] else 3

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as error:print(str(error),file=sys.stderr);sys.exit(1)
