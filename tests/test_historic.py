import pathlib
import tempfile

import GooseHDF5 as g5
import h5py
import pytest

from depinning_inertia_2024 import QuasiStatic


@pytest.fixture(scope="module")
def data_1d():
    """
    Files used in all 1d tests.
    At the end of all test in this module, the temporary directory is removed.
    """
    tmpDir = tempfile.TemporaryDirectory()
    tmp_dir = pathlib.Path(tmpDir.name)
    paths = {"info": tmp_dir / "info.h5", "filename": tmp_dir / "id=0000.h5", "dirname": tmp_dir}

    QuasiStatic.Generate(
        ["--dev", "--eta", 1e0, "--size", 50, "-n", 1, paths["dirname"], "--kframe", 1 / 50]
    )
    QuasiStatic.Run(["--dev", "-n", 1000, paths["filename"]])
    QuasiStatic.EnsembleInfo(["--dev", "-o", paths["info"], paths["filename"]])

    yield paths

    tmpDir.cleanup()


def test_history(data_1d):
    with h5py.File(data_1d["info"]) as file_a:
        with h5py.File(pathlib.Path(__file__).parent / "test_historic_EnsembleInfo.h5") as file_b:
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

    assert ret["!="] == []
    assert ret["->"] == []
    assert ret["<-"] == []
    assert a["!="] == []
    assert b["!="] == []
