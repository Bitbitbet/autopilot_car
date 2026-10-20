#!/usr/bin/env python3
"""离线分析 trackview_race 日志；只读原日志，不连接设备。仅需 Python 标准库。"""
import argparse
import csv
import html
import json
import math
from collections import Counter
from pathlib import Path

LABELS = ['cone', 'person', 'busy', 'limit', 'unlimit', 'stop', 'flow',
          'park', 'gate', 'cross', 'fork', 'left', 'choice', 'yleft', 'yright']


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def integer(value):
    value = number(value)
    return int(value) if value is not None and value == int(value) else None


def stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {'samples': 0, 'p50': None, 'p95': None, 'max': None}
    # 最近秩百分位；保留样本数，避免把缺失值误记成零。
    return {'samples': len(values), 'p50': values[math.ceil(len(values)*.5)-1],
            'p95': values[math.ceil(len(values)*.95)-1], 'max': values[-1]}


def csv_rows(path, warnings, required=False):
    if not path.is_file():
        if required:
            raise ValueError('缺少文件：' + str(path))
        warnings.append('缺少 ' + path.name + '，对应指标不可用。')
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or 'frame' not in reader.fieldnames:
            raise ValueError(path.name + ' 缺少 frame 列')
        rows = []
        for line, row in enumerate(reader, 2):
            if integer(row.get('frame')) is None or None in row or any(v is None for v in row.values()):
                warnings.append('{} 第 {} 行不完整，已跳过。'.format(path.name, line))
                continue
            row['frame'] = integer(row['frame'])
            rows.append(row)
        return rows


def read_annotations(path):
    if path is None:
        return []
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        annotations = list(csv.DictReader(stream))
    previous = 0
    for row in annotations:
        start, end = integer(row.get('start_frame')), integer(row.get('end_frame'))
        if start is None or end is None or start < 1 or end < start:
            raise ValueError('标注的起止帧不合法')
        row.update(start_frame=start, end_frame=end)
    annotations.sort(key=lambda r: r['start_frame'])
    for row in annotations:
        if row['start_frame'] <= previous:
            raise ValueError('标注范围不能重叠')
        previous = row['end_frame']
        if row.get('expected_label', '') not in LABELS + ['none', '']:
            raise ValueError('未知标注类别：' + row['expected_label'])
        if row.get('road_layout', '') not in ['', 'unknown', 'straight', 'fork']:
            raise ValueError('road_layout 只能为 straight、fork、unknown 或空')
    return annotations


def annotation_at(annotations, frame):
    return next((r for r in annotations if r['start_frame'] <= frame <= r['end_frame']), None)


def valid_detection(d, confidence):
    if not isinstance(d, dict):
        return False
    kind, score = integer(d.get('type')), number(d.get('score'))
    x, y, w, h = (integer(d.get(key)) for key in ['x', 'y', 'width', 'height'])
    return (kind is not None and 0 <= kind < len(LABELS) and score is not None and confidence <= score <= 1
            and None not in (x, y, w, h) and x >= 0 and y >= 0 and w > 0 and h > 0
            and x+w <= 320 and y+h <= 240)


def analyze(directory, annotations=None):
    directory = Path(directory)
    warnings = []
    race = csv_rows(directory/'race.csv', warnings, required=True)
    required = {'state', 'reason', 'ai_sequence', 'ai_age_s', 'fresh', 'stop', 'cap', 'command_speed'}
    if not race or not required <= race[0].keys():
        raise ValueError('race.csv 为空或缺少状态/AI/速度列')
    control = csv_rows(directory/'control.csv', warnings)
    timing = csv_rows(directory/'timing.csv', warnings)
    controls = {r['frame']: r for r in control}
    max_age, confidence = .4, .65
    config_file = directory/'race-config.json'
    if config_file.is_file():
        try:
            config = json.loads(config_file.read_text(encoding='utf-8-sig'))
            max_age, confidence = number(config.get('maxDetectionAge')), number(config.get('confidence'))
            if max_age is None or confidence is None or not 0 < confidence <= 1 or max_age <= 0:
                raise ValueError('阈值无效')
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError('race-config.json 无效：' + str(exc))
    else:
        warnings.append('缺少 race-config.json；统计暂按置信度 0.65、年龄上限 0.4 秒解释。')
    observations = {}
    detections_file = directory/'detections.jsonl'
    if detections_file.is_file():
        with detections_file.open(encoding='utf-8-sig') as stream:
            for line, text in enumerate(stream, 1):
                try:
                    item = json.loads(text)
                    sequence = integer(item.get('frame'))
                    if sequence is None or sequence < 1 or not isinstance(item.get('detections'), list):
                        raise ValueError('帧号或 detections 无效')
                    if sequence in observations:
                        warnings.append('AI 帧 {} 重复，采用最后一条记录。'.format(sequence))
                    observations[sequence] = item
                except (ValueError, TypeError, AttributeError):
                    warnings.append('detections.jsonl 第 {} 行无效，已跳过。'.format(line))
    else:
        warnings.append('缺少 detections.jsonl；不能判断 AI 结果是否为空。')
    labels = {frame: [LABELS[integer(d['type'])] for d in obs['detections']
                      if valid_detection(d, confidence)] for frame, obs in observations.items()}
    segments = []
    frame_gaps = 0
    for index, row in enumerate(race):
        previous = race[index-1] if index else None
        contiguous = previous is not None and row['frame'] == previous['frame']+1
        if previous is not None and row['frame'] > previous['frame']+1:
            frame_gaps += row['frame']-previous['frame']-1
        if previous is not None and row['frame'] <= previous['frame']:
            warnings.append('race.csv 帧号重复或倒退：{}'.format(row['frame']))
        if not segments or not contiguous or (row['state'], row['reason']) != (segments[-1]['state'], segments[-1]['reason']):
            segments.append({'state': row['state'], 'reason': row['reason'], 'start_frame': row['frame'],
                             'end_frame': row['frame'], 'frames': 0, 'stop_frames': 0,
                             'empty_ai_consumptions': 0, 'unavailable_ai_consumptions': 0,
                             'annotated_false_fork_frames': 0})
        segment = segments[-1]
        segment['end_frame'] = row['frame']; segment['frames'] += 1
        segment['stop_frames'] += row['stop'] == '1'
        seq = integer(row['ai_sequence'])
        if seq is not None and seq > 0:
            segment['empty_ai_consumptions'] += seq in labels and not labels[seq]
            segment['unavailable_ai_consumptions'] += seq not in labels
        truth = annotation_at(annotations or [], row['frame'])
        segment['annotated_false_fork_frames'] += bool(truth and truth.get('road_layout') == 'straight' and row['state'] == 'FORK_WAIT')
    for segment in segments:
        start = number(controls.get(segment['start_frame'], {}).get('time_s'))
        end = number(controls.get(segment['end_frame'], {}).get('time_s'))
        segment['observed_span_s'] = round(end-start, 6) if start is not None and end is not None and end >= start else None
    age = [number(r['ai_age_s']) for r in race]
    age = [a for a in age if a is not None and a >= 0]
    auto = [r for r in race if r['state'] != 'MANUAL']
    nonzero_stop, over_cap, invalid_speeds = [], [], []
    for row in race:
        speed, cap = number(row['command_speed']), number(row['cap'])
        if speed is None or cap is None:
            invalid_speeds.append(row['frame']); continue
        if row['stop'] == '1' and abs(speed) > 1e-5:
            nonzero_stop.append(row['frame'])
        if row['state'] != 'MANUAL' and speed > cap+1e-5:
            over_cap.append(row['frame'])
    times = [number(r.get('time_s')) for r in control]
    times = [t for t in times if t is not None]
    duration = times[-1]-times[0] if len(times) > 1 and times[-1] > times[0] else None
    observed_fps = (len(times)-1)/duration if duration else None
    annotation_summary = {'provided': annotations is not None,
        'confirmed_false_fork_frames': sum(s['annotated_false_fork_frames'] for s in segments) if annotations is not None else None,
        'annotated_road_layout_frames': sum(bool(annotation_at(annotations or [], r['frame']) and
            annotation_at(annotations or [], r['frame']).get('road_layout') in ('straight', 'fork')) for r in race),
        'unannotated_fork_wait_frames': sum(r['state'] == 'FORK_WAIT' and not
            (annotation_at(annotations or [], r['frame']) and
             annotation_at(annotations or [], r['frame']).get('road_layout') in ('straight', 'fork')) for r in race),
        'positive_expected_observations': 0, 'positive_detected': 0, 'positive_label_missing': 0,
        'negative_expected_observations': 0, 'unexpected_detections': 0}
    for frame, found in labels.items():
        truth = annotation_at(annotations or [], frame)  # 按 AI 自己的图像帧号，不使用滞后一帧的控制帧号。
        if not truth or not truth.get('expected_label'):
            continue
        if truth['expected_label'] == 'none':
            annotation_summary['negative_expected_observations'] += 1
            annotation_summary['unexpected_detections'] += bool(found)
        else:
            annotation_summary['positive_expected_observations'] += 1
            hit = truth['expected_label'] in found
            annotation_summary['positive_detected'] += hit
            annotation_summary['positive_label_missing'] += not hit
    stage_names = ['capture_ms', 'preprocess_ms', 'track_ms', 'control_ms', 'record_publish_ms']
    traces = []
    for trace_file in sorted(directory.glob('fork-trace-*/trace.json')):
        try:
            trace = json.loads(trace_file.read_text(encoding='utf-8-sig'))
            frame = next(f for f in trace['frames'] if f['frame'] == trace['trigger_frame'])
            traces.append({'directory': trace_file.parent.name, 'trigger_frame': frame['frame'],
                           'ai_frame': frame['ai']['frame'], 'ai_objects': len(frame['ai']['detections']),
                           'raw_split_rows': frame.get('raw_split_rows'), 'masked_split_rows': frame.get('masked_split_rows')})
        except (ValueError, KeyError, TypeError, StopIteration):
            warnings.append('取证元数据无效：' + trace_file.parent.name)
    report = {'run': directory.name, 'race_frames': len(race), 'missing_control_frame_numbers': frame_gaps,
        'state_frames': dict(Counter(r['state'] for r in race)), 'state_segments': segments,
        'fork_wait_episodes': sum(r['state'] == 'FORK_WAIT' and
            (i == 0 or race[i-1]['state'] != 'FORK_WAIT' or race[i-1]['frame']+1 != r['frame'])
            for i, r in enumerate(race)),
        'race_stop_reasons': dict(Counter(r['reason'] for r in race if r['stop'] == '1')),
        'lane_path_issues': dict(Counter(r.get('path_issue', 'UNKNOWN') for r in control if r.get('valid') == '0')),
        'control_stop_reasons': dict(Counter(r.get('speed_reason', 'UNKNOWN') for r in control
                                          if number(r.get('speed')) == 0 and r.get('mode') == 'AUTO')),
        'ai': {'unique_logged_results': len(observations), 'empty_accepted_results': sum(not v for v in labels.values()),
               'label_observation_counts': dict(Counter(label for found in labels.values() for label in set(found))),
               'observation_age_s': stats(age), 'active_stale_frames': sum(r['fresh'] == '0' for r in auto),
               'active_age_over_limit_frames': sum(number(r['ai_age_s']) is not None and number(r['ai_age_s']) > max_age for r in auto)},
        'timing': {'observed_control_fps': observed_fps, 'observed_control_span_s': duration,
                   'dt_s': stats([number(r.get('dt_s')) for r in control]),
                   'stages_ms': {key: stats([number(r.get(key)) for r in timing]) for key in stage_names}},
        'command_checks': {'stop_with_nonzero_speed_frames': nonzero_stop, 'speed_over_race_cap_frames': over_cap,
                           'invalid_numeric_frames': invalid_speeds},
        'ground_truth': annotation_summary, 'fork_traces': traces,
        'thresholds': {'confidence': confidence, 'max_detection_age_s': max_age},
        'notes': ['空 AI 结果不等于漏检；没有人工标注时不能计算真实漏检或误判次数。',
                  '观测年龄包含采集、排队、推理和调度，不等于纯 NPU 推理耗时。',
                  'command_speed 是程序计算的命令值；日志不能单独证明串口已发送或小车实际速度。',
                  '观测帧率来自 control.csv，不是摄像头硬件帧率；静态与动态测试需分别解释。'],
        'warnings': warnings}
    return report


def write_report(report, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (output/'state-segments.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        if report['state_segments']:
            writer = csv.DictWriter(stream, fieldnames=list(report['state_segments'][0]))
            writer.writeheader(); writer.writerows(report['state_segments'])
    esc = lambda value: html.escape(str(value))
    def table(headers, rows):
        return '<table><tr>'+''.join('<th>'+esc(v)+'</th>' for v in headers)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
    states = table(['状态', '帧数'], report['state_frames'].items())
    segments = table(['状态', '原因', '起始帧', '结束帧', '帧数', 'AI 空结果使用帧数'],
                     [(s['state'], s['reason'], s['start_frame'], s['end_frame'], s['frames'], s['empty_ai_consumptions'])
                      for s in report['state_segments']])
    ai = report['ai']; age = ai['observation_age_s']; truth = report['ground_truth']
    metrics = [('日志帧数', report['race_frames']), ('FORK_WAIT 进入次数', report['fork_wait_episodes']),
               ('人工确认误判帧数', truth['confirmed_false_fork_frames'] if truth['provided'] else '未提供人工标注'),
               ('不同 AI 结果数', ai['unique_logged_results']), ('其中空结果数', ai['empty_accepted_results']),
               ('AI 观测年龄中位数 / P95（秒）', str(age['p50'])+' / '+str(age['p95'])),
               ('自动模式结果过期帧数', ai['active_stale_frames']), ('观测控制帧率', report['timing']['observed_control_fps']),
               ('停车标志为 1 但计算速度非零', len(report['command_checks']['stop_with_nonzero_speed_frames'])),
               ('自动模式计算速度超过比赛策略上限', len(report['command_checks']['speed_over_race_cap_frames']))]
    notes = ''.join('<li>'+esc(n)+'</li>' for n in report['notes']+report['warnings'])
    stop = table(['停车原因', '帧数'], report['race_stop_reasons'].items())
    truth_table = table(['标注指标', '数量'], truth.items())
    timing = table(['阶段（毫秒）', '样本数', '中位数', 'P95', '最大'],
                   [(k, v['samples'], v['p50'], v['p95'], v['max']) for k, v in report['timing']['stages_ms'].items()])
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>小车日志分析</title>
<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:1150px;margin:36px auto;padding:0 24px;color:#192e40;background:#f7f9fb}h1,h2{color:#173c56}table{border-collapse:collapse;background:white;width:100%;margin:16px 0 28px}td,th{padding:9px 12px;text-align:left;border-bottom:1px solid #d8e2e9}th{background:#e7f0f5}li{margin:9px 0}.notice{padding:16px;background:#fff1cf;border-left:5px solid #d49821;line-height:1.7}code{word-break:break-all}</style>
<h1>小车离线日志分析</h1><p>实验：{{RUN}}</p><div class="notice"><ul>{{NOTES}}</ul></div>
<h2>主要指标</h2>{{METRICS}}<h2>状态分布</h2>{{STATES}}<h2>停车原因</h2>{{STOP}}
<h2>阶段耗时</h2>{{TIMING}}<h2>人工标注核对</h2>{{TRUTH}}<h2>状态变化明细</h2>{{SEGMENTS}}</html>'''
    for token, content in [('RUN', esc(report['run'])), ('NOTES', notes), ('METRICS', table(['指标', '结果'], metrics)),
                           ('STATES', states), ('STOP', stop), ('TIMING', timing), ('TRUTH', truth_table), ('SEGMENTS', segments)]:
        document = document.replace('{{'+token+'}}', content)
    (output/'report.html').write_text(document, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='包含 race.csv 的实验日志目录')
    parser.add_argument('--output', type=Path, help='报告输出目录；默认在当前目录 race_reports/实验名')
    parser.add_argument('--annotations', type=Path, help='人工标注 CSV；按图像帧号标注，不按显示截图时间')
    args = parser.parse_args()
    output = args.output or Path.cwd()/'race_reports'/args.run.name
    try:
        if output.resolve() == args.run.resolve():
            raise ValueError('输出目录不能直接使用原日志目录，请使用单独子目录')
        report = analyze(args.run, read_annotations(args.annotations) if args.annotations else None)
        write_report(report, output)
    except (ValueError, OSError, csv.Error) as exc:
        parser.error(str(exc))
    print('报告已生成：' + str(output/'report.html'))
    print('状态帧数：' + json.dumps(report['state_frames'], ensure_ascii=False))
    print('FORK_WAIT 进入次数：' + str(report['fork_wait_episodes']))
    print('空 AI 结果数：{}；不自动等同于漏检。'.format(report['ai']['empty_accepted_results']))


if __name__ == '__main__':
    main()
