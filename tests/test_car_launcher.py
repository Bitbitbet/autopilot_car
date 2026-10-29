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
            self.assertTrue(race['verifyCrosswalkGeometry']);self.assertEqual(race['crossCandidateConfidence'],.10)
            if args and args[-1] in ('signal','limit'):self.assertFalse(race['stopAfterCross'])
            if args and args[-1]=='limit':self.assertEqual(race['limitedSpeed'],.08)
        for name,data in before.items():self.assertEqual((self.root/name).read_bytes(),data)
    def test_mixed_plugin_or_unexpected_arguments_cannot_drive(self):
        for args in [('bad',),('drive','bad'),('drive','signal','--headless')]:self.assertEqual(self.invoke(*args).returncode,2)
        (self.root/'librace_npu_backend.so').write_bytes(b'wrong backend')
        self.assertNotEqual(self.invoke('drive','signal').returncode,0)
        self.assertFalse((self.root/'invoked.json').exists())
    def test_black_surface_is_explicit_and_preserves_vehicle_configuration(self):
        original=(self.root/'trackview.json').read_bytes()
        for mode in ('preview','drive'):
            result=self.invoke(mode,'competition','black');self.assertEqual(result.returncode,0,result.stderr)
            argv=json.loads((self.root/'invoked.json').read_text())
            actual=json.loads((self.root/argv[argv.index('--config')+1]).read_text())
            self.assertEqual(actual['roadSurface'],'black')
            self.assertEqual((self.root/'trackview.json').read_bytes(),original)
            self.assertEqual('--no-control' in argv,mode=='preview')
    def test_explicit_stop_review_configuration_is_preserved(self):
        p=self.root/'race.json';cfg=json.loads(p.read_text())
        cfg.update(verifyStopCandidates=False,stopCandidateConfidence=.4,
                   verifyCrosswalkGeometry=False,crossCandidateConfidence=.2);p.write_text(json.dumps(cfg))
        result=self.invoke('preview','signal');self.assertEqual(result.returncode,0,result.stderr)
        argv=json.loads((self.root/'invoked.json').read_text())
        actual=json.loads((self.root/argv[argv.index('--race-config')+1]).read_text())
        self.assertFalse(actual['verifyStopCandidates']);self.assertEqual(actual['stopCandidateConfidence'],.4)
        self.assertFalse(actual['verifyCrosswalkGeometry']);self.assertEqual(actual['crossCandidateConfidence'],.2)
    def test_optional_navigation_configuration_is_recorded_without_rewriting(self):
        nav=self.root/'navigation.json';shutil.copy2(self.repo/'config/navigation.json',nav);before=nav.read_bytes()
        for mode in ['preview','drive']:
            result=self.invoke(mode,'competition');self.assertEqual(result.returncode,0,result.stderr)
            argv=json.loads((self.root/'invoked.json').read_text())
            self.assertEqual(argv[argv.index('--navigation-config')+1],'navigation.json')
            launch=max((self.root/'run-records').glob('*-launch.json'),key=lambda p:p.stat().st_mtime_ns)
            record=json.loads(launch.read_text())
            self.assertEqual(record['navigation_config_sha256'],hashlib.sha256(before).hexdigest())
            self.assertTrue(record['obstacle_planning_preview_only'])
            self.assertFalse(record['obstacle_route_controls_vehicle'])
            self.assertEqual(record['sample_write_mode'],'bounded_background_queue')
            self.assertEqual(nav.read_bytes(),before)
    def test_invalid_minimum_or_logs_cannot_drive(self):
        p=self.root/'trackview.json';c=json.loads(p.read_text())
        for change in [{'minSpeed':.15},{'minSpeed':True},{'diagnostics':False}]:
            new=dict(c);new.update(change);p.write_text(json.dumps(new))
            self.assertNotEqual(self.invoke('drive','signal').returncode,0)
        self.assertFalse((self.root/'invoked.json').exists())
    def test_calibration_and_bypass_are_explicit_and_preserve_ground_file(self):
        result=self.invoke('calibrate','competition');self.assertEqual(result.returncode,0,result.stderr)
        argv=json.loads((self.root/'invoked.json').read_text())
        self.assertIn('--calibrate-ground',argv);self.assertIn('--no-control',argv)
        (self.root/'invoked.json').unlink()
        self.assertNotEqual(self.invoke('bypass','competition').returncode,0)
        self.assertFalse((self.root/'invoked.json').exists())
        ground=self.root/'ground-calibration.json';ground.write_text('{"test_only":true}')
        before=ground.read_bytes()
        result=self.invoke('bypass','competition');self.assertEqual(result.returncode,0,result.stderr)
        argv=json.loads((self.root/'invoked.json').read_text())
        self.assertIn('--obstacle-bypass',argv);self.assertNotIn('--no-control',argv)
        self.assertEqual(argv[argv.index('--ground-calibration')+1],'ground-calibration.json')
        launch=max((self.root/'run-records').glob('*-launch.json'),key=lambda p:p.stat().st_mtime_ns)
        record=json.loads(launch.read_text());self.assertTrue(record['obstacle_route_controls_vehicle'])
        self.assertEqual(record['bypass_hard_speed_cap_mps'],10.0)
        self.assertEqual(ground.read_bytes(),before)
        self.assertNotEqual(self.invoke('bypass','signal').returncode,0)

    def test_parking_preview_has_no_control_and_reverse_check_is_explicit(self):
        self.assertNotEqual(self.invoke('parking-preview','competition','black').returncode,0)
        ground=self.root/'ground-calibration.json';ground.write_text('{"test_only":true}')
        result=self.invoke('parking-preview','competition','black');self.assertEqual(result.returncode,0,result.stderr)
        argv=json.loads((self.root/'invoked.json').read_text())
        self.assertIn('--parking-preview',argv);self.assertIn('--no-control',argv)
        self.assertNotIn('--obstacle-bypass',argv)
        result=self.invoke('reverse-check','competition','black');self.assertEqual(result.returncode,0,result.stderr)
        argv=json.loads((self.root/'invoked.json').read_text())
        self.assertIn('--reverse-check',argv);self.assertNotIn('--no-control',argv)
        launch=max((self.root/'run-records').glob('*-launch.json'),key=lambda p:p.stat().st_mtime_ns)
        record=json.loads(launch.read_text());self.assertEqual(record['reverse_check']['duration_s'],.5)
        self.assertTrue(record['reverse_check']['operator_trigger_required'])

    def test_parking_drive_requires_real_reverse_and_explicit_competition_mode(self):
        self.assertNotEqual(self.invoke('parking-drive','competition','black').returncode,0)
        (self.root/'ground-calibration.json').write_text('{"test_only":true}')
        self.assertNotEqual(self.invoke('parking-drive','competition','black').returncode,0)
        p=self.root/'release-manifest.json';manifest=json.loads(p.read_text())
        manifest['parking_control']={'reverse_verified_by_user':True};p.write_text(json.dumps(manifest))
        result=self.invoke('parking-drive','competition','black');self.assertEqual(result.returncode,0,result.stderr)
        argv=json.loads((self.root/'invoked.json').read_text())
        self.assertIn('--parking-drive',argv);self.assertIn('--reverse-verified',argv);self.assertNotIn('--no-control',argv)
        capability=self.root/'parking-reverse-check.json';capability.write_text('{"reverse_verified":false,"pending":true}')
        self.assertNotEqual(self.invoke('parking-drive','competition','black').returncode,0)
        manifest['parking_control']['reverse_report_epoch_s']=capability.stat().st_mtime+1
        p.write_text(json.dumps(manifest))
        self.assertEqual(self.invoke('parking-drive','competition','black').returncode,0)
        manifest['parking_control']['reverse_report_epoch_s']=capability.stat().st_mtime-1
        p.write_text(json.dumps(manifest))
        self.assertNotEqual(self.invoke('parking-drive','competition','black').returncode,0)
        capability.write_text('{"reverse_verified":true}')
        self.assertEqual(self.invoke('parking-drive','competition','black').returncode,0)
        self.assertNotEqual(self.invoke('parking-drive','signal','black').returncode,0)

if __name__=='__main__':unittest.main()
