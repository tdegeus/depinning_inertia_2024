import os
import shutil
import sys
import unittest

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_front", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

from mycode_line import slurm  # noqa: E402


class MyTests(unittest.TestCase):
    def test_exec_cmd(self):

        cmd = 'echo "hello world"'
        script = slurm.script_exec(cmd)
        self.assertEqual(script.split("\n")[-2], "stdbuf -o0 -e0 " + cmd)

        _ = slurm.script_exec(cmd, conda=dict(condabase="my"))
        _ = slurm.script_exec(cmd, conda=None)
        _ = slurm.script_exec(cmd, conda=False)


if __name__ == "__main__":

    unittest.main()
