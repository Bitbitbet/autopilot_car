"""检查实跑信号牌入口的模式、速度和原配置隔离。"""
import hashlib,json,os
from pathlib import Path
import shutil,subprocess,tempfile,unittest

@unittest.skipUnless(os.name=='posix','需要 Linux bash')
class SignalMotionLauncher(unittest.TestCase):
    def setUp(self):
        repo=Path(__file__).parents[1];(repo/'.cache').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=repo/'.cache');self.root=Path(self.temp.name)
        binary=b'#!/usr/bin/env python3\nimport json,sys\nopen("invoked.json","w").write(json.dumps(sys.argv[1:]))\n'
        script=(repo/'tools/run_signal_motion_20261023.sh').read_text().replace(
            '2b22de3116e9fb323bdd89111e953034722c7d6bc640b6ae49af250dc70d8494',hashlib.sha256(binary).hexdigest())
        (self.root/'start.sh').write_text(script)
        p=self.root/'trackview_race_continuous_20261023';p.write_bytes(binary);p.chmod(0o755)
        (self.root/'librace_npu_backend.so').write_bytes(b'mock')
        for name in ['trackview.json','race.json']:shutil.copy2(repo/'config'/name,self.root/name)
    def tearDown(self):self.temp.cleanup()
    def invoke(self,*args):return subprocess.run(['bash',str(self.root/'start.sh'),*args],capture_output=True,text=True)
    def test_explicit_drive_and_independent_config(self):
        before={name:(self.root/name).read_bytes() for name in ['trackview.json','race.json']}
        for mode in [None,'preview','drive']:
            result=self.invoke(*([] if mode is None else [mode]));self.assertEqual(result.returncode,0,result.stderr)
            argv=json.loads((self.root/'invoked.json').read_text())
            self.assertEqual('--no-control' in argv,mode!='drive');self.assertNotIn('--trial-seconds',argv)
            config=json.loads((self.root/argv[argv.index('--config')+1]).read_text())
            expected=json.loads(before['trackview.json'])
            for k in ['cruiseSpeed','maxSpeed','curveSpeed','degradedSpeed']:expected[k]=.1
            self.assertEqual(config,expected)
            expected_race=json.loads(before['race.json']);expected_race['stopAfterCross']=False
            actual=json.loads((self.root/argv[argv.index('--race-config')+1]).read_text())
            self.assertEqual(actual,expected_race)
        for name,data in before.items():self.assertEqual((self.root/name).read_bytes(),data)
        records=list((self.root/'signal-motion-records').glob('*-launch.json'));self.assertEqual(len(records),3)
        for p in records:
            record=json.loads(p.read_text());self.assertFalse(record['trial_timer_enabled'])
            self.assertFalse(record['stop_after_cross_override'])
    def test_refuses_extra_options_and_wrong_binary(self):
        for args in [('unknown',),('drive','--config','x')]:self.assertEqual(self.invoke(*args).returncode,2)
        (self.root/'trackview_race_continuous_20261023').write_bytes(b'wrong version')
        self.assertNotEqual(self.invoke('drive').returncode,0)
        self.assertFalse((self.root/'invoked.json').exists())
    def test_refuses_unsafe_minimum_or_disabled_logs(self):
        p=self.root/'trackview.json';source=json.loads(p.read_text())
        for extra in [{'minSpeed':.2},{'diagnostics':False}]:
            config=dict(source);config.update(extra);p.write_text(json.dumps(config))
            self.assertNotEqual(self.invoke('drive').returncode,0)
        self.assertFalse((self.root/'invoked.json').exists())

if __name__=='__main__':unittest.main()
