import os
import shutil
import sys
import unittest

import GooseHDF5 as g5
import h5py

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "depinning_inertia_2024", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

from depinning_inertia_2024 import QuasiStatic  # noqa: E402

dirname = os.path.join(os.path.dirname(__file__), "output")
idname = "id=0000.h5"
filename = os.path.join(dirname, idname)
infoname = os.path.join(dirname, "EnsembleInfo.h5")
historic = os.path.join(os.path.dirname(__file__), "test_historic_EnsembleInfo.h5")


class MyTests(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(self):
        for file in [filename, infoname]:
            if os.path.isfile(file):
                os.remove(file)

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        QuasiStatic.cli_generate(
            ["--dev", "--eta", 1e0, "--size", 50, "-n", 1, dirname, "--kframe", 1 / 50]
        )
        QuasiStatic.cli_run(["--dev", "-n", 1000, filename])
        QuasiStatic.cli_ensembleinfo(["--dev", "-o", infoname, filename])

    @classmethod
    def tearDownClass(self):
        """
        Remove the temporary directory.
        """

        shutil.rmtree(dirname)

    def test_history(self):
        with h5py.File(infoname) as file_a:
            with h5py.File(historic) as file_b:
                ret, a, b = g5.compare_rename(
                    file_a,
                    file_b,
                    rename=[
                        ["/normalisation/k_interactions", "/normalisation/k_neighbours"],
                        ["/normalisation/u", "/normalisation/x"],
                        ["/avalanche/u_frame", "/avalanche/x_frame"],
                        ["/loading/u_frame", "/loading/x_frame"],
                        ["/full/id=0000.h5/u_frame", "/full/id=0000.h5/x_frame"],
                    ],
                )

        for key in [
            "/normalisation/interactions",
            "/param/data_version",
            "/param/dt",
            "/param/eta",
            "/param/interactions/k",
            "/param/interactions/type",
            "/param/k_frame",
            "/param/m",
            "/param/mu",
            "/param/normalisation/u",
            "/param/potentials/du",
            "/param/potentials/type",
            "/param/potentials/weibull/k",
            "/param/potentials/weibull/mean",
            "/param/potentials/weibull/offset",
            "/param/potentials/xoffset",
            "/param/shape",
            "/lookup/dependencies/index",
            "/lookup/dependencies/value",
            "/lookup/dynamics/index",
            "/lookup/dynamics/value",
            "/lookup/uuid",
            "/lookup/version/value",
            "/meta/QuasiStatic_EnsembleInfo",
            "/normalisation/dynamics",
            "/normalisation/f",
            "/normalisation/name",
            "/normalisation/potential",
            "/normalisation/shape",
            "/normalisation/system",
        ]:
            if key in ret["!="]:
                ret["!="].remove(key)
            if key in ret["<-"]:
                ret["<-"].remove(key)
            if key in ret["->"]:
                ret["->"].remove(key)

        self.assertEqual(ret["!="], [])
        self.assertEqual(ret["->"], [])
        self.assertEqual(ret["<-"], [])
        self.assertEqual(a["!="], [])
        self.assertEqual(b["!="], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
