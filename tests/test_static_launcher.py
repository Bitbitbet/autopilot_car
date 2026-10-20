import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.name == 'posix', '需要 Linux bash；使用离线 WSL 环境测试')
class StaticLauncherTests(unittest.TestCase):
    def setUp(self):
        self.repo = Path(__file__).parents[1]
        cache = self.repo/'.cache'; cache.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.bin = Path(self.temp.name)
        self.launcher = self.bin/'run_race_static_test.sh'
        shutil.copy2(self.repo/'tools/run_race_static_test.sh', self.launcher)
        for name in ['race.json', 'trackview.json']:
            shutil.copy2(self.repo/'config'/name, self.bin/name)
        (self.bin/'trackview_race_cropui').write_text(
            '#!/usr/bin/env python3\nimport json,sys\n'
            'open("invocation.json","w").write(json.dumps(sys.argv[1:]))\n', encoding='utf-8')
        (self.bin/'trackview_race_cropui').chmod(0o755)

    def tearDown(self):
        self.assertEqual(self.bin.resolve().parent, (self.repo/'.cache').resolve())
        self.temp.cleanup()

    def invoke(self, *args):
        return subprocess.run(['bash', str(self.launcher), *args], capture_output=True, text=True)

    def test_all_modes_force_no_control_and_preserve_original_config(self):
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        before = {name: sha(self.bin/name) for name in ['race.json', 'trackview.json']}
        for mode in ['baseline', 'signal', 'limit']:
            result = self.invoke(mode)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads((self.bin/'invocation.json').read_text())
            self.assertEqual(args[-1], '--no-control')
            self.assertEqual(len(args), 5)
            config = json.loads((self.bin/args[3]).read_text())
            if mode in ['signal', 'limit']: self.assertFalse(config['stopAfterCross'])
            if mode == 'limit': self.assertEqual(config['limitedSpeed'], .15)
        for name, expected in before.items(): self.assertEqual(sha(self.bin/name), expected)
        meta = list((self.bin/'static-test-records').glob('*-launch.json'))
        self.assertEqual(len(meta), 3)
        self.assertTrue(all(json.loads(p.read_text())['no_control'] for p in meta))

    def test_cannot_pass_extra_or_unknown_arguments(self):
        for args in [('limit','--headless'), ('drive',), ('limit --no-control',)]:
            self.assertEqual(self.invoke(*args).returncode, 2)
        self.assertFalse((self.bin/'invocation.json').exists())

    def test_disabled_logs_or_missing_binary_do_not_start(self):
        path = self.bin/'trackview.json'; config = json.loads(path.read_text()); config['diagnostics'] = False
        path.write_text(json.dumps(config))
        self.assertNotEqual(self.invoke('limit').returncode, 0)
        self.assertFalse((self.bin/'invocation.json').exists())
        (self.bin/'trackview_race_cropui').unlink()
        self.assertNotEqual(self.invoke('limit').returncode, 0)


if __name__ == '__main__':
    unittest.main()
