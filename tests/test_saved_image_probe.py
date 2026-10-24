import importlib.util,json
from pathlib import Path
import tempfile,unittest

repo=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('image_probe',repo/'tools/run_saved_image_probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

class SavedImageIsolation(unittest.TestCase):
    def setUp(self):
        (repo/'.cache').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=repo/'.cache');self.source=Path(self.temp.name)
        model=self.source/'res/models/yolov3_mobilenet_v1';model.mkdir(parents=True)
        (model/'deploy_paddle.params').write_bytes(b'unchanged weights')
        (model/'nms.tar.so').write_bytes(b'original generated model')
        for name in ['config_ppncnna.json','config_ppncnms.json']:
            (model.parent/name).write_text(json.dumps({'model_dir':'../res/models/yolov3_mobilenet_v1','mode':'professional'}))
        for n in range(1,11):
            p=self.source/('bin/trackview_logs/1792746339640-6299/ai-sample-'+str(n));p.mkdir(parents=True)
            (p/'camera.png').write_bytes(('saved image '+str(n)).encode())
        self.root=self.source/'bin/recognition-probe-test';self.root.mkdir()
    def tearDown(self):self.temp.cleanup()
    def test_isolated_copy_and_immutable_originals(self):
        before=probe.protected_snapshot(self.source)
        rows=probe.prepare(self.source,self.root)
        self.assertEqual(len(rows),10)
        copied=self.root/'res/models/yolov3_mobilenet_v1/nms.tar.so'
        copied.write_bytes(b'diagnostic model compilation changes only the copy')
        self.assertEqual(probe.protected_snapshot(self.source),before)
        for row in rows:self.assertTrue(Path(row['path']).is_relative_to(self.root))
        with self.assertRaisesRegex(RuntimeError,'already prepared'):probe.prepare(self.source,self.root)
    def test_refuses_other_locations(self):
        with self.assertRaisesRegex(RuntimeError,'dedicated child'):
            probe.prepare(self.source,self.source/'res/recognition-probe-test')

if __name__=='__main__':unittest.main()
