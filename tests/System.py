import os
import shutil
import sys
import unittest

import h5py
import numpy as np
import shelephant

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_line", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

import mycode_line as my  # noqa: E402

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

        my.System.cli_generate(["--dev", "-N", 50, "-n", 1, dirname])
        my.System.cli_run(["--dev", filename])

    @classmethod
    def tearDownClass(self):

        shutil.rmtree(dirname)

    def test_foo(self):
        """
        No tests yet
        """

        self.assertTrue(True)


if __name__ == "__main__":

    unittest.main(verbosity=2)
