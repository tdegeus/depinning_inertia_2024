import pathlib
import shutil
import tempfile

import h5py
import numpy as np
import pytest

from depinning_inertia_2024 import Dynamics
from depinning_inertia_2024 import QuasiStatic
from depinning_inertia_2024 import Trigger


@pytest.fixture(scope="module")
def mydata():
    """
    Files used in all, remove temporary directory at the end of all tests.
    """
    tmpDir = tempfile.TemporaryDirectory()
    tmp_dir = pathlib.Path(tmpDir.name)
    paths = {"info": tmp_dir / "info.h5", "filename": tmp_dir / "id=0000.h5", "dirname": tmp_dir}

    QuasiStatic.Generate(
        ["--dev", "--eta", 1, "--size", 50, "-n", 1, paths["dirname"], "--kframe", 1 / 50]
    )
    QuasiStatic.Run(["--dev", "-n", 1000, paths["filename"]])
    QuasiStatic.EnsembleInfo(["--dev", "-o", paths["info"], paths["filename"]])

    yield paths

    tmpDir.cleanup()


def test_branch_trigger(mydata):
    """
    Branch and trigger at system spanning
    """
    tdir = mydata["dirname"] / "Trigger"
    Trigger.Generate(["--dev", "-o", tdir, mydata["info"]])
    Trigger.Run(["--dev", tdir / "id=0000.h5"])
    shutil.rmtree(tdir)


def test_branch_trigger_deltaf(mydata):
    """
    Branch and trigger at --delta-f > 0
    """
    tdir = mydata["dirname"] / "Trigger"
    Trigger.Generate(["--dev", "--delta-f", 0.1, "-o", tdir, mydata["info"]])
    Trigger.Run(["--dev", tdir / "id=0000.h5"])
    Trigger.Run(["--dev", tdir / "id=0000.h5", "--check", 0])
    Trigger.UpdateData(["--dev", tdir / "id=0000.h5"])
    Trigger.EnsembleInfo(["--dev", "-o", tdir / "info.h5", tdir / "id=0000.h5"])

    with h5py.File(tdir / "id=0000.h5", "a") as file:
        branch = np.arange(file["/Trigger/step"].size)

    Trigger.Run(["--dev", tdir / "id=0000.h5", "--check", branch[0]])

    Dynamics.Run(
        ["--dev", "-f", "--step", 1, "--branch", 0, "-o", tdir / "dynamics.h5", tdir / "id=0000.h5"]
    )
