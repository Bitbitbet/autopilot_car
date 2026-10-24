#!/usr/bin/env python3
"""只读盘点另一辆同型号车；不加载模型，不打开相机、串口，不安装或改写文件。"""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

MODEL_FILES = ['deploy_paddle.tar', 'deploy_paddle.so', 'deploy_paddle.params',
               'deploy_paddle.ro', 'post.onnx', 'io_paddle.json', 'label_list.txt',
               'nms.tar', 'nms.tar.so']
EXPECTED_LABELS = ['cone', 'person', 'busy', 'limit', 'unlimit', 'stop', 'flow',
                   'park', 'gate', 'cross', 'fork', 'left', 'choice', 'yleft', 'yright']


def inspect(path):
    p = Path(path)
    row = {'path': str(p), 'exists': p.is_file()}
    if not row['exists']:
        return row
    try:
        size = p.stat().st_size
        row.update(size=size, resolved=str(p.resolve()))
        if size > 100 * 1024 * 1024:
            row['hash_skipped'] = '文件大于 100 MB'
        else:
            h = hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(block)
            row['sha256'] = h.hexdigest()
    except (OSError, IOError) as e:
        row['error'] = str(e)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', action='append', help='原车项目目录；可重复指定，不会修改目录')
    args = parser.parse_args()
    roots = args.root or ['/home/bot', '/root/track-trial-20261010',
                          '/root/workspace/icar_autopilot_2026th']
    report = {'read_only': True, 'architecture': platform.machine(),
              'glibc': platform.libc_ver(), 'projects': [],
              'devices_present': {name: sorted(glob.glob(pattern)) for name, pattern in
                                  [('camera', '/dev/video*'), ('serial', '/dev/serial/by-id/*'),
                                   ('ttyUSB', '/dev/ttyUSB*')]},
              'commands_present': {n: bool(shutil.which(n)) for n in ['python3', 'g++', 'readelf']}}
    for root in roots:
        project = Path(root)
        if not project.is_dir():
            continue
        model = project/'res/models/yolov3_mobilenet_v1'
        item = {'root': str(project), 'model_files': [inspect(model/n) for n in MODEL_FILES],
                'calibration': inspect(project/'res/calibration/calibration.xml'), 'ppnc_configs': {}}
        for n in ['config_ppncnna.json', 'config_ppncnms.json']:
            p = project/'res/models'/n
            try:
                item['ppnc_configs'][n] = json.loads(p.read_text(encoding='utf-8-sig'))
            except (OSError, ValueError) as e:
                item['ppnc_configs'][n] = {'error': str(e)}
        try:
            labels = (model/'label_list.txt').read_text(encoding='utf-8-sig').splitlines()
            item.update(labels=labels, label_order_matches=labels == EXPECTED_LABELS)
        except OSError as e:
            item['labels_error'] = str(e)
        report['projects'].append(item)
    if shutil.which('ldconfig'):
        try:
            text = subprocess.check_output(['ldconfig', '-p'], stderr=subprocess.STDOUT).decode('utf-8', 'replace')
            report['runtime_library_entries'] = [s.strip() for s in text.splitlines()
                if any(n in s for n in ['libppnc', 'libonnxruntime', 'libopencv_', 'libserial', 'libevdev', 'libudev'])]
        except (OSError, subprocess.CalledProcessError) as e:
            report['library_error'] = str(e)
    report['not_verified'] = ['相机安装角度及标定是否适配', '舵机中值、方向和限位',
        '下位机固件与速度单位', '主控设备认证及实际 NPU 推理', '真实赛道通过能力']
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
