import os
import pathlib
import shutil
import sys
import unittest
from functools import partialmethod

import h5py
import numpy as np
from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_line", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

from mycode_line import QuasiStatic  # noqa: E402

dirname = pathlib.Path(__file__).parent / "output"


class MyTests(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(self):

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

    @classmethod
    def tearDownClass(self):
        """
        Remove the temporary directory.
        """

        shutil.rmtree(dirname)

    def test_simple(self):
        """
        Distributions: weibull, delta, random
        """

        for distro in ["weibull", "delta", "random"]:

            dname = dirname / distro
            filename = dname / "id=0000.h5"
            infoname = dname / "EnsembleInfo.h5"

            QuasiStatic.cli_generate(
                ["--dev", "--eta", 1e0, "-N", 50, "-n", 1, dname, "--distribution", distro]
            )

            with h5py.File(filename, "a") as file:
                file["param"]["xyield"]["nchunk"][...] = 100

            QuasiStatic.cli_run(["--dev", "-n", 1000, filename])
            QuasiStatic.cli_ensembleinfo(["--dev", "-o", infoname, filename])

            with h5py.File(infoname) as file:
                A = file["/loading/A"][...]

            self.assertTrue(np.all(A == 0))


if __name__ == "__main__":

    unittest.main(verbosity=2)
