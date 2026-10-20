import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('race_log_report', Path(__file__).parents[1]/'tools/race_log_report.py')
reporter = importlib.util.module_from_spec(spec); spec.loader.exec_module(reporter)


class LogReportTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).parents[1]/'.cache'
        base.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.run = Path(self.temporary.name)/'sample'; self.run.mkdir()
        self.headers = ['frame','state','reason','ai_sequence','ai_age_s','fresh','stop','cap','branch_selected','command_speed']
        self.rows = [[1,'MANUAL','MANUAL',0,-1,0,0,10,0,0],
                     [2,'LANE','CRUISE',1,.1,1,0,10,0,.2],
                     [3,'FORK_WAIT','NO_CONFIRMED_DIRECTION',2,0,1,1,10,0,0],
                     [4,'FORK_WAIT','NO_CONFIRMED_DIRECTION',2,.1,1,1,10,0,0],
                     [5,'AI_WAIT','AI_STALE',2,.5,0,1,10,0,0],
                     [6,'FORK_WAIT','NO_CONFIRMED_DIRECTION',5,.1,1,1,10,0,.1]]
        self.write_race()
        ds = [{'frame':1,'captured':1,'detections':[]}, {'frame':2,'captured':2,'detections':[]},
              {'frame':5,'captured':3,'detections':[{'type':3,'score':.99,'x':20,'y':20,'width':40,'height':40}]}]
        (self.run/'detections.jsonl').write_text('\n'.join(json.dumps(d) for d in ds), encoding='utf-8')
        (self.run/'race-config.json').write_text('{"confidence":0.65,"maxDetectionAge":0.4}', encoding='utf-8')

    def tearDown(self):
        self.temporary.cleanup()

    def write_race(self):
        with (self.run/'race.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f); writer.writerow(self.headers); writer.writerows(self.rows)

    def test_counts_do_not_call_empty_results_misses(self):
        r = reporter.analyze(self.run)
        self.assertEqual(r['fork_wait_episodes'], 2)
        self.assertEqual(r['ai']['unique_logged_results'], 3)
        self.assertEqual(r['ai']['empty_accepted_results'], 2)
        self.assertIsNone(r['ground_truth']['confirmed_false_fork_frames'])
        self.assertEqual(r['ground_truth']['unannotated_fork_wait_frames'], 3)
        self.assertEqual(r['ai']['active_stale_frames'], 1)  # 不把启动时的 MANUAL 算过期驾驶。
        self.assertEqual(r['ai']['observation_age_s']['samples'], 5)
        self.assertEqual(r['command_checks']['stop_with_nonzero_speed_frames'], [6])
        self.assertIsNone(r['timing']['observed_control_fps'])

    def test_annotations_match_ai_frame_not_consuming_control_frame(self):
        a = [{'start_frame':2,'end_frame':2,'expected_label':'limit','road_layout':'straight'},
             {'start_frame':3,'end_frame':4,'expected_label':'none','road_layout':'straight'}]
        r = reporter.analyze(self.run, a)
        self.assertEqual(r['ground_truth']['confirmed_false_fork_frames'], 2)
        self.assertEqual(r['ground_truth']['positive_label_missing'], 1)
        self.assertEqual(r['ground_truth']['negative_expected_observations'], 0)  # AI 没保存 3、4 帧，不能补成空结果。
        self.assertEqual(r['ground_truth']['unannotated_fork_wait_frames'], 1)

    def test_missing_ai_log_is_unavailable(self):
        (self.run/'detections.jsonl').unlink()
        r = reporter.analyze(self.run)
        self.assertEqual(r['ai']['empty_accepted_results'], 0)
        self.assertEqual(r['state_segments'][2]['unavailable_ai_consumptions'], 2)

    def test_partial_files_and_nonfinite_values(self):
        with (self.run/'race.csv').open('a') as f: f.write('7,LANE')
        with (self.run/'detections.jsonl').open('a') as f: f.write('\n{"frame":')
        r = reporter.analyze(self.run)
        self.assertEqual(r['race_frames'], 6)
        self.assertTrue(any('不完整' in w for w in r['warnings']))
        self.assertTrue(any('无效' in w for w in r['warnings']))
        self.assertIsNone(reporter.number('nan'))
        self.assertIsNone(reporter.number('inf'))
        self.assertFalse(reporter.valid_detection({'type':3,'score':float('nan')},.65))

    def test_html_escapes_user_text_and_keeps_state_names(self):
        self.rows[4][2] = '<script>alert(1)</script> REVIEW_STOP'
        self.write_race(); r = reporter.analyze(self.run)
        out = self.run/'analysis'; reporter.write_report(r, out)
        html = (out/'report.html').read_text(encoding='utf-8')
        self.assertNotIn('<script>', html)
        self.assertIn('&lt;script&gt;', html)
        self.assertIn('REVIEW_STOP', html)
        self.assertNotIn('{{', html)
        self.assertEqual(json.loads((out/'summary.json').read_text(encoding='utf-8'))['race_frames'], 6)

    def test_reason_change_does_not_count_an_extra_fork_episode(self):
        self.rows[3][2] = 'ANOTHER_REASON'; self.write_race()
        self.assertEqual(reporter.analyze(self.run)['fork_wait_episodes'], 2)

    def test_overlapping_annotations_rejected(self):
        path = self.run/'truth.csv'
        path.write_text('start_frame,end_frame,expected_label,road_layout\n1,3,none,straight\n3,5,limit,fork\n', encoding='utf-8')
        with self.assertRaises(ValueError): reporter.read_annotations(path)

    def test_command_cap_check_only_applies_to_auto(self):
        self.rows[0][-1] = 20
        self.rows[1][-1] = 11
        self.write_race()
        self.assertEqual(reporter.analyze(self.run)['command_checks']['speed_over_race_cap_frames'], [2])


if __name__ == '__main__':
    unittest.main()
