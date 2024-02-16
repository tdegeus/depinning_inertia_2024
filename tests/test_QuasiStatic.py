import os
import pathlib
import tempfile

import GooseHDF5 as g5
import h5py
import numpy as np
import pytest

from depinning_inertia_2024 import Dynamics
from depinning_inertia_2024 import EventMap
from depinning_inertia_2024 import QuasiStatic
from depinning_inertia_2024 import Relaxation
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


def test_y(mydata):
    """
    Jumping in yield history.
    """

    with h5py.File(mydata["filename"]) as file:
        system = QuasiStatic.allocate_system(file)
        jump = QuasiStatic.allocate_system(file)

    du = system.chunk.data[..., -1] * 0.5
    n = 50

    for i in range(1, n):
        system.chunk_goto(du * i)
        system.u = du * i

    jump.chunk_goto(system.u)
    jump.u = system.u

    assert np.all(np.equal(system.chunk.index_at_align, jump.chunk.index_at_align))
    assert np.allclose(system.chunk.left_of_align, jump.chunk.left_of_align)
    assert np.allclose(system.chunk.right_of_align, jump.chunk.right_of_align)


def test_y_delta(mydata):
    """
    Jumping in yield history: delta distribution.
    """

    deltaname = mydata["dirname"] / "id=delta.h5"

    with h5py.File(mydata["filename"]) as file, h5py.File(deltaname, "w") as delta:
        datasets = list(
            g5.getdatapaths(file, root="/param", fold="/param/potentials/weibull", fold_symbol="")
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

    assert np.all(np.equal(system.chunk.index_at_align, jump.chunk.index_at_align))
    assert np.allclose(system.chunk.left_of_align, jump.chunk.left_of_align)
    assert np.allclose(system.chunk.right_of_align, jump.chunk.right_of_align)


def test_fastload(mydata):
    """
    Read using fastload.
    """

    with h5py.File(mydata["filename"]) as file:
        system = QuasiStatic.allocate_system(file)
        step = int(0.2 * file["/QuasiStatic/inc"].size)
        system.restore_quasistatic_step(file["QuasiStatic"], step)
        yleft = system.chunk.left_of_align
        yright = system.chunk.right_of_align
        fpot = system.f_potential

    QuasiStatic.GenerateFastLoad(["--dev", mydata["filename"], "--force"])

    with h5py.File(mydata["filename"]) as file:
        system = QuasiStatic.allocate_system(file)
        system.restore_quasistatic_step(file["QuasiStatic"], step)
        assert np.allclose(yleft, system.chunk.left_of_align)
        assert np.allclose(yright, system.chunk.right_of_align)
        assert np.allclose(fpot, system.f_potential)

    QuasiStatic.CheckFastLoad([mydata["filename"]])

    # just a call
    with h5py.File(mydata["filename"]) as file:
        system = QuasiStatic.allocate_system(file)
        step = file["/QuasiStatic/inc"].size - 1
        system.restore_quasistatic_step(file["QuasiStatic"], step)


def test_chunk(mydata):
    """
    Rerun using huge chunk.
    """
    QuasiStatic.Run(["--dev", "--check", 950, mydata["filename"]])
    QuasiStatic.Run(["--dev", "--check", 951, mydata["filename"]])
    QuasiStatic.Run(["--dev", "--check", 952, mydata["filename"]])
    QuasiStatic.Run(["--dev", "--check", 953, mydata["filename"]])

    tmp = mydata["dirname"] / "info_duplicate.h5"
    QuasiStatic.EnsembleInfo(["--dev", "-f", "-o", tmp, mydata["filename"]])

    with h5py.File(mydata["info"]) as src, h5py.File(tmp) as dest:
        dset = list(g5.getdatasets(src))
        diff = g5.compare(src, dest, dset)
        for key in diff:
            if key == "==":
                assert len(diff[key]) > 0
            else:
                assert len(diff[key]) == 0


def test_eventmap(mydata):
    """
    Create event map.
    """
    with h5py.File(mydata["info"]) as file:
        for fname in file["full"]:
            path = mydata["info"].parent / fname
            assert path == mydata["filename"]
            step = file["full"][fname]["step"][...]
            A = file["full"][fname]["A"][...]
            N = file["normalisation"]["N"][...]
            i = np.argwhere(A == N).ravel()
            s = step[i[-2]]
            t = step[i[-1]]
            break

        out_s = mydata["dirname"] / "EventMap_s.h5"
        out_t = mydata["dirname"] / "EventMap_t.h5"
        EventMap.Run(["--dev", "-f", "-s", "-o", out_s, "--step", str(s), path])
        EventMap.Run(["--dev", "-f", "-s", "-o", out_t, "--step", str(t), path])

        out = mydata["dirname"] / "EventMapInfo.h5"
        EventMap.Info(["--dev", "-f", "-o", out, out_s, out_t])


def test_relaxation(mydata):
    """
    Measure relaxation.
    """

    with h5py.File(mydata["info"]) as file:
        for fname in file["full"]:
            path = mydata["info"].parent / fname
            step = file["full"][fname]["step"][...]
            A = file["full"][fname]["A"][...]
            N = file["normalisation"]["N"][...]
            i = np.argwhere(A == N).ravel()
            s = step[i[-2]]
            t = step[i[-1]]
            break

    out_s = mydata["dirname"] / "Relaxation_s.h5"
    out_t = mydata["dirname"] / "Relaxation_t.h5"
    Relaxation.Run(["--dev", "-f", "-o", out_s, "--step", str(s), path])
    Relaxation.Run(["--dev", "-f", "-o", out_t, "--step", str(t), path])

    out = mydata["dirname"] / "RelaxationInfo.h5"
    Relaxation.EnsembleInfo(["--dev", "-f", "-o", out, out_s, out_t])


def test_measuredynamics(mydata):
    """
    Rerun dynamics.
    """

    with h5py.File(mydata["info"]) as file:
        for fname in file["full"]:
            path = os.path.join(os.path.dirname(mydata["info"]), fname)
            step = file["full"][fname]["step"][...]
            A = file["full"][fname]["A"][...]
            N = file["normalisation"]["N"][...]
            i = np.argwhere(A == N).ravel()
            s = step[i[-2]]
            break

    out = mydata["dirname"] / "MeasureDynamics_s.h5"
    ens = mydata["dirname"] / "MeasureDynamics_average.h5"
    Dynamics.Run(["--dev", "-f", "--step", s, "-o", out, path])
    Dynamics.AverageSystemSpanning(["-f", "--dev", "-o", ens, out])


def test_read(mydata):
    """
    Read output.
    """

    QuasiStatic.StateAfterSystemSpanning(
        ["--dev", "-f", "-o", mydata["dirname"] / "AfterSystemSpanning.h5", mydata["info"]]
    )

    QuasiStatic.StructureAfterSystemSpanning(
        ["--dev", "-f", "-o", mydata["dirname"] / "StructureFactor.h5", mydata["info"]]
    )


@pytest.fixture(scope="module")
def data_2d():
    """
    Files used in all 2d tests.
    At the end of all test in this module, the temporary directory is removed.
    """
    tmpDir = tempfile.TemporaryDirectory()
    tmp_dir = pathlib.Path(tmpDir.name)
    paths = {"info": tmp_dir / "info.h5", "filename": tmp_dir / "id=0000.h5", "dirname": tmp_dir}

    QuasiStatic.Generate(
        ["--dev", "--eta", 1, "--shape", 10, 10, "-n", 1, paths["dirname"], "--kframe", 1 / 100]
    )
    QuasiStatic.Run(["--dev", "-n", 600, paths["filename"]])
    QuasiStatic.EnsembleInfo(["--dev", "-o", paths["info"], paths["filename"]])

    yield paths

    tmpDir.cleanup()


opts = [
    ["--size", 50],
    ["--size", 50, "--quarticgradient", 1, 1],
    ["--shape", 10, 10],
]


@pytest.mark.parametrize("opts", opts)
def test_basic_trigger(opts, tmp_path):
    """
    Basic function calls.
    """
    qdir = tmp_path / "QuasiStatic"
    QuasiStatic.Generate(["--dev", "--eta", 1, "-n", 2, qdir, "--kframe", 1 / 50] + opts)
    QuasiStatic.Run(["--dev", "-n", 300, qdir / "id=0000.h5"])
    QuasiStatic.Run(["--dev", "-n", 300, qdir / "id=0000.h5"])
    QuasiStatic.Run(["--dev", "-n", 600, qdir / "id=0001.h5"])
    QuasiStatic.EnsembleInfo(
        ["--dev", "-o", qdir / "info.h5", qdir / "id=0000.h5", qdir / "id=0001.h5"]
    )

    tdir = tmp_path / "Trigger"
    Trigger.Generate(["--dev", "-o", tdir, qdir / "info.h5"])
    Trigger.Run(["--dev", tdir / "id=0000.h5"])
