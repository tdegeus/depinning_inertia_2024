import os
import shutil
import sys
import unittest

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_line", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

from mycode_line import QuasiStatic  # noqa: E402

dirname = os.path.join(os.path.dirname(__file__), "output")
idname = "id=0000.h5"
filename = os.path.join(dirname, idname)
infoname = os.path.join(dirname, "EnsembleInfo.h5")


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

    @classmethod
    def tearDownClass(self):

        shutil.rmtree(dirname)

    def test_read(self):
        """
        Read output.
        """

        ss = os.path.join(dirname, "AfterSystemSpanning.h5")

        QuasiStatic.cli_ensembleinfo(["--dev", "-o", infoname, filename])
        QuasiStatic.cli_stateaftersystemspanning(["--dev", "-o", ss, infoname])


if __name__ == "__main__":

    unittest.main(verbosity=2)
