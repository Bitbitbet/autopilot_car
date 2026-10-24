import hashlib,json,os
from pathlib import Path
import shutil,subprocess,tempfile,unittest

@unittest.skipUnless(os.name=='posix','需要本地 Linux bash')
class GroundTrialLauncher(unittest.TestCase):
    def setUp(self):
        self.repo=Path(__file__).parents[1]
        (self.repo/'.cache').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=self.repo/'.cache')
        self.root=Path(self.temp.name)
        shutil.copy2(self.repo/'tools/run_ground_trial_20261022.sh',self.root/'start.sh')
        c=json.loads((self.repo/'config/trackview.json').read_text())
        for key in ['cruiseSpeed','maxSpeed','curveSpeed','degradedSpeed']:c[key]=.1
        (self.root/'trackview_low_speed_20261022.json').write_text(json.dumps(c))
        shutil.copy2(self.repo/'config/race.json',self.root/'race.json')
        p=self.root/'trackview_race_timedtrial'
        p.write_text('#!/usr/bin/env python3\nimport json,sys\nopen("invoked.json","w").write(json.dumps(sys.argv[1:]))\n')
        p.chmod(0o755)
    def tearDown(self):
        self.assertEqual(self.root.resolve().parent,(self.repo/'.cache').resolve())
        self.temp.cleanup()
    def invoke(self,*args):
        return subprocess.run(['bash',str(self.root/'start.sh'),*args],capture_output=True,text=True)
    def test_explicit_modes_and_immutable_original_files(self):
        files=[self.root/'race.json',self.root/'trackview_low_speed_20261022.json']
        before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
        for mode in [None,'preview','drive']:
            result=self.invoke(*([] if mode is None else [mode]))
            self.assertEqual(result.returncode,0,result.stderr)
            argv=json.loads((self.root/'invoked.json').read_text())
            self.assertEqual(argv[argv.index('--trial-seconds')+1],'2')
            self.assertEqual('--no-control' in argv,mode!='drive')
        self.assertEqual(before,[hashlib.sha256(p.read_bytes()).hexdigest() for p in files])
        records=[json.loads(p.read_text()) for p in (self.root/'low-speed-trial-records').glob('*-launch.json')]
        self.assertEqual(len(records),3)
        self.assertEqual(sum(not r['no_control'] for r in records),1)
    def test_reject_unknown_or_extra_arguments(self):
        for args in [('drive','--trial-seconds','9'),('drive --no-control',),('unknown',)]:
            self.assertEqual(self.invoke(*args).returncode,2)
        self.assertFalse((self.root/'invoked.json').exists())
    def test_high_speed_or_missing_logs_cannot_start(self):
        p=self.root/'trackview_low_speed_20261022.json'; c=json.loads(p.read_text())
        for update in [{'maxSpeed':10},{'curveSpeed':.15},{'cruiseSpeed':.25},{'diagnostics':False}]:
            original=dict(c);original.update(update);p.write_text(json.dumps(original))
            self.assertNotEqual(self.invoke('drive').returncode,0)
        self.assertFalse((self.root/'invoked.json').exists())

if __name__=='__main__':unittest.main()
