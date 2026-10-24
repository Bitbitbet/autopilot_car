import hashlib,json,os,shutil,subprocess,tempfile,unittest
from pathlib import Path

@unittest.skipUnless(os.name=='posix','需要 Linux bash')
class UnifiedCarLauncher(unittest.TestCase):
    def setUp(self):
        self.repo=Path(__file__).parents[1];(self.repo/'.cache').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=self.repo/'.cache');self.root=Path(self.temp.name)
        shutil.copy2(self.repo/'tools/run_car.sh',self.root/'run_car.sh')
        binary=self.root/'trackview_race';binary.write_text('#!/usr/bin/env python3\nimport json,sys\nopen("invoked.json","w").write(json.dumps(sys.argv[1:]))\n');binary.chmod(0o755)
        (self.root/'librace_npu_backend.so').write_bytes(b'mock paired backend')
        files={n:hashlib.sha256((self.root/n).read_bytes()).hexdigest() for n in ['trackview_race','librace_npu_backend.so']}
        (self.root/'release-manifest.json').write_text(json.dumps({'files':files}))
        for name in ['trackview.json','race.json']:shutil.copy2(self.repo/'config'/name,self.root/name)
    def tearDown(self):self.temp.cleanup()
    def invoke(self,*args):return subprocess.run(['bash',str(self.root/'run_car.sh'),*args],capture_output=True,text=True)
    def test_default_preview_and_profiles_preserve_originals(self):
        before={n:(self.root/n).read_bytes() for n in ['trackview.json','race.json']}
        for args in [()]+[(mode,profile) for mode in ['preview','drive'] for profile in ['lane','signal','limit','competition']]:
            result=self.invoke(*args);self.assertEqual(result.returncode,0,result.stderr)
            argv=json.loads((self.root/'invoked.json').read_text())
            self.assertEqual('--no-control' in argv,not args or args[0]=='preview')
            self.assertNotIn('--trial-seconds',argv)
            track=json.loads((self.root/argv[argv.index('--config')+1]).read_text())
            for key in ['cruiseSpeed','curveSpeed','degradedSpeed']:self.assertEqual(track[key],.1)
            self.assertEqual(track['maxSpeed'],10.0)
            launch=max((self.root/'run-records').glob('*-launch.json'),key=lambda p:p.stat().st_mtime_ns)
            overrides=json.loads(launch.read_text())['speed_overrides_mps']
            self.assertEqual(overrides,{key:track[key] for key in ['cruiseSpeed','maxSpeed','curveSpeed','degradedSpeed']})
            race=json.loads((self.root/argv[argv.index('--race-config')+1]).read_text())
            self.assertEqual(race['confidence'],.65);self.assertEqual(race['stopCandidateConfidence'],.35)
            self.assertTrue(race['verifyStopCandidates'])
            if args and args[-1] in ('signal','limit'):self.assertFalse(race['stopAfterCross'])
            if args and args[-1]=='limit':self.assertEqual(race['limitedSpeed'],.08)
        for name,data in before.items():self.assertEqual((self.root/name).read_bytes(),data)
    def test_mixed_plugin_or_unexpected_arguments_cannot_drive(self):
        for args in [('bad',),('drive','bad'),('drive','signal','--headless')]:self.assertEqual(self.invoke(*args).returncode,2)
        (self.root/'librace_npu_backend.so').write_bytes(b'wrong backend')
        self.assertNotEqual(self.invoke('drive','signal').returncode,0)
        self.assertFalse((self.root/'invoked.json').exists())
    def test_explicit_stop_review_configuration_is_preserved(self):
        p=self.root/'race.json';cfg=json.loads(p.read_text())
        cfg.update(verifyStopCandidates=False,stopCandidateConfidence=.4);p.write_text(json.dumps(cfg))
        result=self.invoke('preview','signal');self.assertEqual(result.returncode,0,result.stderr)
        argv=json.loads((self.root/'invoked.json').read_text())
        actual=json.loads((self.root/argv[argv.index('--race-config')+1]).read_text())
        self.assertFalse(actual['verifyStopCandidates']);self.assertEqual(actual['stopCandidateConfidence'],.4)
    def test_invalid_minimum_or_logs_cannot_drive(self):
        p=self.root/'trackview.json';c=json.loads(p.read_text())
        for change in [{'minSpeed':.15},{'minSpeed':True},{'diagnostics':False}]:
            new=dict(c);new.update(change);p.write_text(json.dumps(new))
            self.assertNotEqual(self.invoke('drive','signal').returncode,0)
        self.assertFalse((self.root/'invoked.json').exists())

if __name__=='__main__':unittest.main()
