import os
import shutil
import sys
import unittest

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_line", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

from mycode_line import QuasiStatic  # noqa: E402
from mycode_line import Trigger  # noqa: E402

dirname = os.path.join(os.path.dirname(__file__), "output")
workdir = os.path.join(dirname, "trigger")
idname = "id=0000.h5"
filename = os.path.join(dirname, idname)
infoname = os.path.join(dirname, "EnsembleInfo.h5")

tfile = os.path.join(workdir, idname)
tinfo = os.path.join(workdir, "EnsembleInfo.h5")


class MyTests(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(self):

        for file in [filename, infoname]:
            if os.path.isfile(file):
                os.remove(file)

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        QuasiStatic.cli_generate(["--dev", "--eta", 1e0, "-N", 50, "-n", 1, dirname])
        QuasiStatic.cli_run(["--dev", "-n", 1000, filename])
        QuasiStatic.cli_ensembleinfo(["--dev", "-o", infoname, filename])

    @classmethod
    def tearDownClass(self):

        shutil.rmtree(dirname)

    def test_branch_trigger(self):
        """
        Branch and trigger
        """

        Trigger.cli_generate(["--dev", "-o", workdir, infoname])
        Trigger.cli_run(["--dev", tfile])

        shutil.rmtree(workdir)

        Trigger.cli_generate(["--dev", "-o", workdir, "--delta-f", 0.1, infoname])
        Trigger.cli_run(["--dev", tfile])
        Trigger.cli_ensembleinfo(["--dev", "-o", tinfo, tfile])


if __name__ == "__main__":

    unittest.main(verbosity=2)