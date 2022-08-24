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

from mycode_line import EventMap  # noqa: E402
from mycode_line import MeasureDynamics  # noqa: E402
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

        with h5py.File(filename, "a") as file:
            file["param"]["xyield"]["nchunk"][...] = 100
            file["param"]["xyield"]["nbuffer"][...] = 20

        QuasiStatic.cli_run(["--dev", "-n", 1000, filename])
        QuasiStatic.cli_ensembleinfo(["--dev", "-o", infoname, filename])

    @classmethod
    def tearDownClass(self):
        """
        Remove the temporary directory.
        """

        shutil.rmtree(dirname)

    def test_fastload(self):
        """
        Store state of the random sequence at system spanning events for fast reloading.
        """

        fastname = os.path.join(dirname, "EnsembleFastLoad.h5")
        QuasiStatic.cli_fastload(["--dev", "-f", "-o", fastname, infoname, "-n", 10])

        with h5py.File(filename) as file, h5py.File(fastname) as fastload:

            system = QuasiStatic.System(file)
            step = fastload[idname]["step"][...][-1]

            system.restore_quasistatic_step(file=file, step=step)

            i = np.copy(system.i + system.istart)
            all = np.arange(system.N)
            yll = system.y[all, system.i - 2]
            yl = system.y[all, system.i - 1]
            yr = system.y[all, system.i]
            yrr = system.y[all, system.i + 1]

            system.restore_quasistatic_step(
                file=file,
                step=step,
                state=fastload[f"/{idname}/data/{step:d}/state"][...],
                istate=fastload[f"/{idname}/data/{step:d}/istate"][...],
                y0=fastload[f"/{idname}/data/{step:d}/y0"][...],
            )

            self.assertTrue(np.all(np.equal(system.i + system.istart, i)))
            self.assertTrue(np.allclose(system.y[all, system.i - 2], yll))
            self.assertTrue(np.allclose(system.y[all, system.i - 1], yl))
            self.assertTrue(np.allclose(system.y[all, system.i], yr))
            self.assertTrue(np.allclose(system.y[all, system.i + 1], yrr))

    def test_chunk(self):

        with h5py.File(filename, "a") as file:
            file["param"]["xyield"]["nchunk"][...] = 10000
            file["param"]["xyield"]["nbuffer"][...] = 300

        QuasiStatic.cli_run(["--dev", "--check", 950, filename])
        QuasiStatic.cli_run(["--dev", "--check", 951, filename])
        QuasiStatic.cli_run(["--dev", "--check", 952, filename])
        QuasiStatic.cli_run(["--dev", "--check", 953, filename])

        tmp = os.path.join(dirname, "EnsembleInfo_duplicate.h5")
        QuasiStatic.cli_ensembleinfo(["--dev", "-f", "-o", tmp, filename])

        with h5py.File(infoname) as src, h5py.File(tmp) as dest:
            dset = list(g5.getdatasets(src))
            diff = g5.compare(src, dest, dset)
            for key in diff:
                if key == "==":
                    self.assertGreater(len(diff[key]), 0)
                else:
                    self.assertEqual(len(diff[key]), 0)

    def test_eventmap(self):

        with h5py.File(infoname) as file:
            for fname in file["full"]:
                path = os.path.join(os.path.dirname(infoname), fname)
                step = file["full"][fname]["step"][...]
                A = file["full"][fname]["A"][...]
                N = file["normalisation"]["N"][...]
                i = np.argwhere(A == N).ravel()
                s = step[i[-2]]
                t = step[i[-1]]
                break

        out_s = os.path.join(dirname, "EventMap_s.h5")
        out_t = os.path.join(dirname, "EventMap_t.h5")
        EventMap.cli_run(["--dev", "-f", "-o", out_s, "-s", str(s), path])
        EventMap.cli_run(["--dev", "-f", "-o", out_t, "-s", str(t), path])

        out = os.path.join(dirname, "EventMapInfo.h5")
        EventMap.cli_basic_output(["--dev", "-f", "-o", out, out_s, out_t])

    def test_measuredynamics(self):

        with h5py.File(infoname) as file:
            for fname in file["full"]:
                path = os.path.join(os.path.dirname(infoname), fname)
                step = file["full"][fname]["step"][...]
                A = file["full"][fname]["A"][...]
                N = file["normalisation"]["N"][...]
                i = np.argwhere(A == N).ravel()
                s = step[i[-2]]
                break

        out = os.path.join(dirname, "MeasureDynamics_s.h5")
        MeasureDynamics.cli_run(["--dev", "-f", "-i", "20", "-o", out, "-s", str(s), path])

    def test_read(self):
        """
        Read output.
        """

        ss = os.path.join(dirname, "AfterSystemSpanning.h5")
        QuasiStatic.cli_stateaftersystemspanning(["--dev", "-f", "-o", ss, infoname])


if __name__ == "__main__":

    unittest.main(verbosity=2)
