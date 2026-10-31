import hashlib,json,os,shutil,subprocess,tempfile,unittest
from pathlib import Path

@unittest.skipUnless(os.environ.get('RACE_NATIVE_TEST_BINARY'),'Requires cross-built native entry and QEMU')
class NativeStartup(unittest.TestCase):
    def setUp(self):
        self.repo=Path(__file__).parents[1]
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        shutil.copy2(os.environ['RACE_NATIVE_TEST_BINARY'],self.root/'racing')
        (self.root/'racing').chmod(0o755)
        (self.root/'librace_npu_backend.so').write_bytes(b'No real backend: preparation must never load it')
        for name in ['trackview.json','race.json']:
            shutil.copy2(self.repo/'config'/name,self.root/name)
        self.manifest={'files':{name:self.sha(name) for name in ['racing','librace_npu_backend.so']}}
        self.save_manifest()
    def tearDown(self):self.temp.cleanup()
    def sha(self,name):return hashlib.sha256((self.root/name).read_bytes()).hexdigest()
    def save_manifest(self):(self.root/'release-manifest.json').write_text(json.dumps(self.manifest))
    def run_native(self,*args):
        return subprocess.run(['qemu-aarch64','-cpu','max','-L',os.environ['RACE_NATIVE_SYSROOT'],str(self.root/'racing'),*args],cwd='/',capture_output=True,text=True)
    def record(self):
        return json.loads(max((self.root/'run-records').glob('*-launch.json'),key=lambda p:p.stat().st_mtime_ns).read_text())
    def test_prepare_only_uses_executable_directory_and_never_opens_hardware(self):
        result=self.run_native('--prepare-only','mission-lab','competition','white')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('PREPARED ONLY',result.stdout)
        record=self.record();self.assertEqual(record['binary'],'racing')
        self.assertIn('--mission-lab',record['command'])
        self.assertEqual(record['processing_resolution'],[640,480])
        config=json.loads((self.root/record['command'][record['command'].index('--config')+1]).read_text())
        self.assertEqual(config['cruiseSpeed'],.1);self.assertEqual(config['maxSpeed'],10)
        self.assertEqual(config['roadSurface'],'white')
    def test_manifest_mismatch_is_rejected_before_vehicle_initialization(self):
        (self.root/'librace_npu_backend.so').write_bytes(b'mismatched')
        result=self.run_native('--prepare-only','drive','competition','black')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('交付清单不匹配',result.stderr)
        self.assertNotIn('Device authentication',result.stdout)
    def test_parking_checks_then_records_automatic_reverse_parameter(self):
        first=self.run_native('--prepare-only','mission','competition','black')
        self.assertNotEqual(first.returncode,0)
        (self.root/'ground-calibration.json').write_text('{}')
        (self.root/'parking-reverse-check.json').write_text('{"reverse_verified":true}')
        result=self.run_native('--prepare-only','mission','competition','black')
        self.assertEqual(result.returncode,0,result.stderr)
        record=self.record();self.assertIn('--reverse-verified',record['command'])
        race_path=record['command'][record['command'].index('--race-config')+1]
        race=json.loads((self.root/race_path).read_text())
        self.assertFalse(race['stopAfterCross']);self.assertGreaterEqual(race['forkMaxTime'],12)
        self.assertEqual(race['parkingSpeed'],.30)
        self.assertEqual(record['backend_abi'],5)
        self.assertIn('left or right',record['parking_control']['selection_policy'])
    def test_parking_area_speed_preserves_vehicle_override(self):
        path=self.root/'race.json';config=json.loads(path.read_text());config['parkingSpeed']=.4
        path.write_text(json.dumps(config))
        result=self.run_native('--prepare-only','mission-lab','competition','black')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(self.record()['parking_control']['parking_area_speed_cap_mps'],.4)
    def test_bare_racing_prepares_drive_before_any_hardware_access(self):
        self.manifest['files']['racing']='invalid';self.save_manifest()
        result=self.run_native()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('交付清单不匹配',result.stderr)
        self.assertNotIn('Cannot read race config',result.stderr)

if __name__=='__main__':unittest.main()
