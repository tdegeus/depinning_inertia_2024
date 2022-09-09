import os
import shutil
import sys
import unittest

import GooseHDF5 as g5
import h5py
import numpy as np

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_line", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

from mycode_line import QuasiStatic  # noqa: E402
from mycode_line import Trigger  # noqa: E402

dirname = os.path.join(os.path.dirname(__file__), "output")
workdir = os.path.join(dirname, "trigger")
bakdir = os.path.join(dirname, "trigger_bak")
idname = "id=0000.h5"
filename = os.path.join(dirname, idname)
infoname = os.path.join(dirname, "EnsembleInfo.h5")
fastname = os.path.join(dirname, "EnsembleFastLoad.h5")

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

        with h5py.File(filename, "a") as file:
            file["param"]["xyield"]["nchunk"][...] = 150
            file["param"]["xyield"]["nbuffer"][...] = 20

        QuasiStatic.cli_run(["--dev", "-n", 1000, filename])
        QuasiStatic.cli_ensembleinfo(["--dev", "-o", infoname, filename])
        QuasiStatic.cli_fastload(["--dev", "-s", "-o", fastname, infoname])

    @classmethod
    def tearDownClass(self):

        shutil.rmtree(dirname)

    def test_branch_trigger(self):
        """
        Branch and trigger at system spanning
        """

        Trigger.cli_generate(["--dev", "-o", workdir, infoname])
        Trigger.cli_run(["--dev", tfile])
        shutil.rmtree(workdir)

    def test_branch_trigger_deltaf(self):
        """
        Branch and trigger at --delta-f > 0
        """

        # generate, store as reference
        Trigger.cli_generate(["--dev", "--delta-f", 0.1, "-o", workdir, infoname])
        os.makedirs(bakdir)
        shutil.copyfile(os.path.join(workdir, idname), os.path.join(bakdir, idname))
        shutil.rmtree(workdir)

        # generate using "--fastload" and check
        Trigger.cli_generate(["--dev", "--delta-f", 0.1, "-o", workdir, "-q", fastname, infoname])
        cmp = g5.compare(os.path.join(workdir, idname), os.path.join(bakdir, idname))
        cmp["!="].remove("/meta/Trigger_Generate")
        self.assertEqual(len(cmp["<-"]), 0)
        self.assertEqual(len(cmp["->"]), 0)
        self.assertEqual(len(cmp["!="]), 0)

        # run
        Trigger.cli_run(["--dev", tfile])
        Trigger.cli_ensembleinfo(["--dev", "-o", tinfo, tfile])

        with h5py.File(tfile, "a") as file:
            file["param"]["xyield"]["nchunk"][...] = 10000
            file["param"]["xyield"]["nbuffer"][...] = 300
            branch = np.arange(file["/Trigger/step"].size)

        Trigger.cli_run(["--dev", tfile, "--check", branch[0]])
        shutil.rmtree(workdir)


if __name__ == "__main__":

    unittest.main(verbosity=2)
