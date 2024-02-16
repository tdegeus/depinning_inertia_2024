import h5py
import numpy as np

from depinning_inertia_2024 import QuasiStatic


def test_simple(tmp_path):
    for distro in ["delta", "random", "weibull"]:
        dname = tmp_path / distro
        filename = dname / "id=0000.h5"
        infoname = dname / "EnsembleInfo.h5"

        QuasiStatic.cli_generate(
            [
                "--dev",
                "--eta",
                1e0,
                "--size",
                50,
                "-n",
                1,
                dname,
                "--distribution",
                distro,
                "--kframe",
                1 / 50,
            ]
        )
        QuasiStatic.cli_run(["--dev", "-n", 1000, filename])
        QuasiStatic.cli_ensembleinfo(["--dev", "-o", infoname, filename])

        with h5py.File(infoname) as file:
            A = file["/loading/A"][...]

        assert np.all(A == 0)
