import os
import pathlib
import shutil
import sys
import tempfile
import unittest

from shelephant.path import cwd

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "depinning_inertia_2024", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

from depinning_inertia_2024 import Flow  # noqa: E402

dirname = os.path.join(os.path.dirname(__file__), "output")
idname = "id=0000.h5"
filename = os.path.join(dirname, idname)
infoname = os.path.join(dirname, "EnsembleInfo.h5")


class MyTests(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(self):
        self.origin = pathlib.Path().absolute()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)

    @classmethod
    def tearDownClass(self):
        os.chdir(self.origin)
        shutil.rmtree(self.tempdir)

    def test_basic(self):
        outdir = "athermal"
        Flow.cli_generate(
            [
                "--dev",
                "--eta",
                1e0,
                "--size",
                50,
                "-n",
                1,
                "--v-frame",
                1,
                outdir,
                "--kframe",
                1 / 50,
                "--nstep",
                100,
            ]
        )

        with cwd(outdir):
            Flow.cli_run(["--dev", "id=0000.h5"])
            Flow.cli_ensemblepack(["--dev", "-o", "info.h5", "id=0000.h5"])
            Flow.cli_ensemblepack(["--dev", "-o", "info.h5", "-i", "id=0000.h5"])

    def test_thermal(self):
        outdir = "thermal"
        Flow.cli_generate(
            [
                "--dev",
                "--eta",
                1e0,
                "--size",
                50,
                "-n",
                1,
                "--v-frame",
                1,
                outdir,
                "--kframe",
                1 / 50,
                "--nstep",
                100,
                "--temperature",
                0.1,
            ]
        )

        with cwd(outdir):
            Flow.cli_run(["--dev", "id=0000.h5"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
