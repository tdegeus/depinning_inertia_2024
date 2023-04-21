import os
import pathlib
import shutil
import sys
import unittest
from functools import partialmethod

import GooseHDF5 as g5
import h5py
import numpy as np
from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_line", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

from mycode_line import Dynamics  # noqa: E402
from mycode_line import Trigger  # noqa: E402
from mycode_line import EventMap  # noqa: E402
from mycode_line import QuasiStatic  # noqa: E402

dirname = pathlib.Path(__file__).parent / "output"
idname = "id=0000.h5"
idname2 = "id=0001.h5"
filename = dirname / idname
infoname = dirname / "EnsembleInfo.h5"


class MyTests(unittest.TestCase):
    """
    Various detailed tests.
    """

    @classmethod
    def setUpClass(self):
        for file in [filename, infoname]:
            if os.path.isfile(file):
                os.remove(file)

        dirname.mkdir(parents=True, exist_ok=True)

        QuasiStatic.cli_generate(
            ["--dev", "--eta", 1, "--size", 50, "-n", 1, dirname, "--kframe", 1 / 50]
        )
        QuasiStatic.cli_run(["--dev", "-n", 1000, filename])
        QuasiStatic.cli_ensembleinfo(["--dev", "-o", infoname, filename])

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

        du = system.chunk.data[..., -1] * 0.5
        n = 50

        for i in range(1, n):
            system.chunk_goto(du * i)
            system.u = du * i

        jump.chunk_goto(system.u)
        jump.u = system.u

        self.assertTrue(np.all(np.equal(system.chunk.index_at_align, jump.chunk.index_at_align)))
        self.assertTrue(np.allclose(system.chunk.left_of_align, jump.chunk.left_of_align))
        self.assertTrue(np.allclose(system.chunk.right_of_align, jump.chunk.right_of_align))

    def test_y_delta(self):
        """
        Jumping in yield history: delta distribution.
        """

        deltaname = dirname / "id=delta.h5"

        with h5py.File(filename) as file, h5py.File(deltaname, "w") as delta:
            datasets = list(
                g5.getdatapaths(
                    file, root="/param", fold="/param/potentials/weibull", fold_symbol=""
                )
            )
            datasets.remove("/param/potentials/weibull")
            g5.copy(file, delta, datasets)
            g5.copy(file, delta, g5.getdatapaths(file, root="/realisation"))
            delta["/param/potentials/delta/mean"] = 2.0

        with h5py.File(deltaname) as file:
            system = QuasiStatic.allocate_system(file)
            jump = QuasiStatic.allocate_system(file)

        du = system.chunk.data[..., -1] * 0.5
        n = 50

        for i in range(1, n):
            system.chunk_goto(du * i)
            system.u = du * i

        jump.chunk_goto(system.u)
        jump.u = system.u

        self.assertTrue(np.all(np.equal(system.chunk.index_at_align, jump.chunk.index_at_align)))
        self.assertTrue(np.allclose(system.chunk.left_of_align, jump.chunk.left_of_align))
        self.assertTrue(np.allclose(system.chunk.right_of_align, jump.chunk.right_of_align))

    def test_fastload(self):
        """
        Read using fastload.
        """

        with h5py.File(filename) as file:
            system = QuasiStatic.allocate_system(file)
            step = int(0.2 * file["/QuasiStatic/inc"].size)
            system.restore_quasistatic_step(file["QuasiStatic"], step)
            yleft = system.chunk.left_of_align
            yright = system.chunk.right_of_align
            fpot = system.f_potential

        QuasiStatic.cli_generatefastload(["--dev", filename, "--force"])

        with h5py.File(filename) as file:
            system = QuasiStatic.allocate_system(file)
            system.restore_quasistatic_step(file["QuasiStatic"], step)
            self.assertTrue(np.allclose(yleft, system.chunk.left_of_align))
            self.assertTrue(np.allclose(yright, system.chunk.right_of_align))
            self.assertTrue(np.allclose(fpot, system.f_potential))

        QuasiStatic.cli_checkfastload([filename])

        # just a call
        with h5py.File(filename) as file:
            system = QuasiStatic.allocate_system(file)
            step = file["/QuasiStatic/inc"].size - 1
            system.restore_quasistatic_step(file["QuasiStatic"], step)

    def test_chunk(self):
        """
        Rerun using huge chunk.
        """

        QuasiStatic.cli_run(["--dev", "--check", 950, filename])
        QuasiStatic.cli_run(["--dev", "--check", 951, filename])
        QuasiStatic.cli_run(["--dev", "--check", 952, filename])
        QuasiStatic.cli_run(["--dev", "--check", 953, filename])

        tmp = dirname / "EnsembleInfo_duplicate.h5"
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
        """
        Create event map.
        """

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

        out_s = dirname / "EventMap_s.h5"
        out_t = dirname / "EventMap_t.h5"
        EventMap.cli_run(["--dev", "-f", "-s", "-o", out_s, "--step", str(s), path])
        EventMap.cli_run(["--dev", "-f", "-s", "-o", out_t, "--step", str(t), path])

        out = dirname / "EventMapInfo.h5"
        EventMap.cli_basic_output(["--dev", "-f", "-o", out, out_s, out_t])

    def test_measuredynamics(self):
        """
        Rerun dynamics.
        """

        with h5py.File(infoname) as file:
            for fname in file["full"]:
                path = os.path.join(os.path.dirname(infoname), fname)
                step = file["full"][fname]["step"][...]
                A = file["full"][fname]["A"][...]
                N = file["normalisation"]["N"][...]
                i = np.argwhere(A == N).ravel()
                s = step[i[-2]]
                break

        out = dirname / "MeasureDynamics_s.h5"
        ens = dirname / "MeasureDynamics_average.h5"
        Dynamics.cli_run(["--dev", "-f", "--step", s, "-o", out, path])
        Dynamics.cli_average_systemspanning(["-f", "--dev", "-o", ens, out])

    def test_read(self):
        """
        Read output.
        """

        QuasiStatic.cli_stateaftersystemspanning(
            ["--dev", "-f", "-o", dirname / "AfterSystemSpanning.h5", infoname]
        )

        QuasiStatic.cli_structurefactor_aftersystemspanning(
            ["--dev", "-f", "-o", dirname / "StructureFactor.h5", infoname]
        )


class MyGlobalTests(unittest.TestCase):
    """
    Test various system types.
    """

    @classmethod
    def setUpClass(self):
        dirname.mkdir(parents=True, exist_ok=True)

        for file in [filename, infoname]:
            if os.path.isfile(file):
                os.remove(file)

    @classmethod
    def tearDownClass(self):
        """
        Remove the temporary directory.
        """

        shutil.rmtree(dirname)

    def test_2d(self):
        """
        2d system
        """

        workdir = dirname / "2d"
        iname = "EnsembleInfo.h5"
        qdir = workdir / "QuasiStatic"
        opts = ["--dev", "--eta", 1, "-n", 2, qdir]
        QuasiStatic.cli_generate(opts + ["--shape", 10, 10, "--kframe", 1 / 100])
        QuasiStatic.cli_run(["--dev", "-n", 300, qdir / idname])
        QuasiStatic.cli_run(["--dev", "-n", 300, qdir / idname])
        QuasiStatic.cli_run(["--dev", "-n", 600, qdir / idname2])
        QuasiStatic.cli_ensembleinfo(["--dev", "-o", qdir / iname, qdir / idname, qdir / idname2])

        tdir = workdir / "Trigger"
        Trigger.cli_generate(["--dev", "-o", tdir, qdir / iname])
        Trigger.cli_run(["--dev", tdir / idname])
        shutil.rmtree(qdir)

    def test_quarticgradient(self):
        """
        QuarticGradient
        """

        workdir = dirname / "quadratic_gradient"
        iname = "EnsembleInfo.h5"
        qdir = workdir / "QuasiStatic"
        opts = ["--dev", "--eta", 1, "-n", 2, qdir, "--size", 50, "--kframe", 1 / 50]
        QuasiStatic.cli_generate(opts + ["--quarticgradient", 1, 1])
        QuasiStatic.cli_run(["--dev", "-n", 300, qdir / idname])
        QuasiStatic.cli_run(["--dev", "-n", 300, qdir / idname])
        QuasiStatic.cli_run(["--dev", "-n", 600, qdir / idname2])
        QuasiStatic.cli_ensembleinfo(["--dev", "-o", qdir / iname, qdir / idname, qdir / idname2])

        tdir = workdir / "Trigger"
        Trigger.cli_generate(["--dev", "-o", tdir, qdir / iname])
        Trigger.cli_run(["--dev", tdir / idname])
        shutil.rmtree(qdir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
