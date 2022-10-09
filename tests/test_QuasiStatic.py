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

from mycode_line import Dynamics  # noqa: E402
from mycode_line import EventMap  # noqa: E402
from mycode_line import QuasiStatic  # noqa: E402

dirname = os.path.join(os.path.dirname(__file__), "output")
idname = "id=0000.h5"
filename = os.path.join(dirname, idname)
infoname = os.path.join(dirname, "EnsembleInfo.h5")
fastname = os.path.join(dirname, "EnsembleFastLoad.h5")


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
        QuasiStatic.cli_fastload(["--dev", "-f", "-o", fastname, infoname, "-s"])

    @classmethod
    def tearDownClass(self):
        """
        Remove the temporary directory.
        """

        shutil.rmtree(dirname)

    def test_y(self):
        """
        Jumping in yield history.
        """

        with h5py.File(filename) as file:
            system = QuasiStatic.allocate_system(file)
            jump = QuasiStatic.allocate_system(file)

        dx = system.y[..., -1] * 0.5
        n = 50

        for i in range(1, n):
            system._chunk_goto(dx * i)
            system.x = dx * i

        jump._chunk_goto(system.x)
        jump.x = system.x

        self.assertTrue(np.all(np.equal(system.istart + system.i, jump.istart + jump.i)))
        self.assertTrue(np.allclose(system.y_left(), jump.y_left()))
        self.assertTrue(np.allclose(system.y_right(), jump.y_right()))

    def test_y_delta(self):
        """
        Jumping in yield history: delta distribution.
        """

        deltaname = os.path.join(dirname, "id=delta.h5")

        with h5py.File(filename) as file, h5py.File(deltaname, "w") as delta:
            datasets = list(
                g5.getdatapaths(file, root="/param", fold="/param/xyield/weibull", fold_symbol="")
            )
            datasets.remove("/param/xyield/weibull")
            g5.copy(file, delta, datasets)
            g5.copy(file, delta, g5.getdatapaths(file, root="/realisation"))
            delta["/param/xyield/delta/mean"] = 2.0

        with h5py.File(deltaname) as file:
            system = QuasiStatic.allocate_system(file)
            jump = QuasiStatic.allocate_system(file)

        dx = system.y[..., -1] * 0.5
        n = 50

        for i in range(1, n):
            system._chunk_goto(dx * i)
            system.x = dx * i

        jump._chunk_goto(system.x)
        jump.x = system.x

        self.assertTrue(np.all(np.equal(system.istart + system.i, jump.istart + jump.i)))
        self.assertTrue(np.allclose(system.y_left(), jump.y_left()))
        self.assertTrue(np.allclose(system.y_right(), jump.y_right()))

    def test_fastload(self):
        """
        Store state of the random sequence at system spanning events for fast reloading.
        """

        with h5py.File(filename) as file:

            fastload = QuasiStatic.FastLoad(fastname, idname)
            system = QuasiStatic.System(file)
            step = fastload.file[idname]["step"][...][-1]

            system.restore_quasistatic_step(file["QuasiStatic"], step=step)

            i = np.copy(system.i + system.istart)
            all = np.arange(system.N)
            yll = system.y[all, system.i - 2]
            yl = system.y[all, system.i - 1]
            yr = system.y[all, system.i]
            yrr = system.y[all, system.i + 1]

            system.restore_quasistatic_step(file["QuasiStatic"], step=0, fastload=fastload)
            system.restore_quasistatic_step(file["QuasiStatic"], step=step, fastload=fastload)

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
        ens = os.path.join(dirname, "MeasureDynamics_average.h5")
        Dynamics.cli_run(["--dev", "-f", "--step", s, "-o", out, path])
        Dynamics.cli_average_systemspanning(["-f", "--dev", "-o", ens, out])

    def test_read(self):
        """
        Read output.
        """

        ss = os.path.join(dirname, "AfterSystemSpanning.h5")
        QuasiStatic.cli_stateaftersystemspanning(
            ["--dev", "-f", "-q", fastname, "-o", ss, infoname]
        )


if __name__ == "__main__":

    unittest.main(verbosity=2)
