"""验证连续实车入口不会意外开启控制或丢失现场标定配置。"""
import hashlib,json,os
from pathlib import Path
import shutil,subprocess,tempfile,unittest

@unittest.skipUnless(os.name=='posix','需要 Linux bash')
class LaneCompareLauncher(unittest.TestCase):
    def setUp(self):
        self.repo=Path(__file__).parents[1]
        (self.repo/'.cache').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=self.repo/'.cache')
        self.root=Path(self.temp.name)
        # 测试替身只记录参数；替换测试副本中的版本指纹，不改正式脚本。
        binary=b'#!/usr/bin/env python3\nimport json,sys\nopen("invoked.json","w").write(json.dumps(sys.argv[1:]))\n'
        script=(self.repo/'tools/run_lane_compare_20261023.sh').read_text().replace(
            '2b22de3116e9fb323bdd89111e953034722c7d6bc640b6ae49af250dc70d8494',hashlib.sha256(binary).hexdigest())
        (self.root/'start.sh').write_text(script)
        for name in ['trackview_race_continuous_20261023','trackview_fixed_v6_max10_defaults']:
            p=self.root/name;p.write_bytes(binary);p.chmod(0o755)
        (self.root/'librace_npu_backend.so').write_bytes(b'mock')
        self.config=json.loads((self.repo/'config/trackview.json').read_text())
        self.config.update(runP1=2.37,rowCutBottom=23)
        (self.root/'trackview.json').write_text(json.dumps(self.config))
        shutil.copy2(self.repo/'config/race.json',self.root/'race.json')
    def tearDown(self):self.temp.cleanup()
    def invoke(self,*args):
        return subprocess.run(['bash',str(self.root/'start.sh'),*args],capture_output=True,text=True)
    def test_modes_and_identical_site_parameters_without_timer(self):
        originals={p.name:p.read_bytes() for p in self.root.glob('*.json')}
        configs=[]
        for args in [(),('race','preview'),('race','drive'),('v6','preview'),('v6','drive')]:
            result=self.invoke(*args);self.assertEqual(result.returncode,0,result.stderr)
            argv=json.loads((self.root/'invoked.json').read_text())
            self.assertEqual('--no-control' in argv,not args or args[-1]!='drive')
            self.assertNotIn('--trial-seconds',argv)
            config=json.loads((self.root/argv[argv.index('--config')+1]).read_text());configs.append(config)
            expected=dict(self.config)
            for key in ['cruiseSpeed','maxSpeed','curveSpeed','degradedSpeed']:expected[key]=.1
            self.assertEqual(config,expected)
        self.assertTrue(all(c==configs[0] for c in configs))
        for name,data in originals.items():self.assertEqual((self.root/name).read_bytes(),data)
        records=list((self.root/'lane-compare-records').glob('*-launch.json'))
        self.assertEqual(len(records),5)
        for p in records:
            record=json.loads(p.read_text());self.assertFalse(record['trial_timer_enabled'])
            self.assertEqual(record['command'][0],'./'+record['binary'])
    def test_reject_unknown_arguments_and_unsafe_minimum(self):
        for args in [('unknown',),('race','unknown'),('race','drive','--headless')]:
            self.assertEqual(self.invoke(*args).returncode,2)
        for minimum in [.15,True,float('nan')]:
            config=dict(self.config,minSpeed=minimum)
            (self.root/'trackview.json').write_text(json.dumps(config))
            self.assertNotEqual(self.invoke('race','drive').returncode,0)
        self.assertFalse((self.root/'invoked.json').exists())
    def test_reject_wrong_binary_and_conflicting_v6(self):
        (self.root/'trackview_race_continuous_20261023').write_bytes(b'wrong version')
        self.assertNotEqual(self.invoke('race','drive').returncode,0)
        (self.root/'trackview_fixed_v6_max10_default').write_bytes(b'other v6')
        self.assertNotEqual(self.invoke('v6','drive').returncode,0)
        self.assertFalse((self.root/'invoked.json').exists())

if __name__=='__main__':unittest.main()
