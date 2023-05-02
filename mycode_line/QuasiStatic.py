"""
QuasiStatic simulations, and tools to run those simulators.
"""
from __future__ import annotations

import argparse
import inspect
import os
import pathlib
import re
import shutil
import tempfile
import textwrap
import uuid

import click
import enstat
import FrictionQPotSpringBlock  # noqa: F401
import FrictionQPotSpringBlock as fsb
import FrictionQPotSpringBlock.Line1d as model
import GooseEYE as eye
import GooseFEM
import GooseHDF5 as g5
import h5py
import numpy as np
import prettytable
import prrng
import shelephant
import tqdm
import XDMFWrite_h5py as xh
from numpy.typing import ArrayLike

from . import storage
from . import tag
from . import tools
from ._version import version

entry_points = dict(
    cli_updatedata="QuasiStatic_UpdateData",
    cli_checkdata="QuasiStatic_CheckData",
    cli_force_current_data_version="QuasiStatic_ForceCurrentDataVersion",
    cli_ensembleinfo="QuasiStatic_EnsembleInfo",
    cli_generatefastload="QuasiStatic_GenerateFastLoad",
    cli_generate="QuasiStatic_Generate",
    cli_plot="QuasiStatic_Plot",
    cli_paraview="QuasiStatic_Paraview",
    cli_run="QuasiStatic_Run",
    cli_checkdynamics="QuasiStatic_CheckDynamics",
    cli_checkfastload="QuasiStatic_CheckFastLoad",
    cli_job_rerun="QuasiStatic_JobRerun",
    cli_plotstateaftersystemspanning="QuasiStatic_PlotStateAfterSystemSpanning",
    cli_stateaftersystemspanning="QuasiStatic_StateAfterSystemSpanning",
    cli_structurefactor_aftersystemspanning="QuasiStatic_StructureAfterSystemSpanning",
)

file_defaults = dict(
    cli_ensembleinfo="QuasiStatic_EnsembleInfo.h5",
    cli_stateaftersystemspanning="QuasiStatic_StateAfterSystemSpanning.h5",
    cli_structurefactor_aftersystemspanning="QuasiStatic_StructureAfterSystemSpanning.h5",
)

data_version = "2.0"


def _updatedata_fastload(src: h5py.File, dst: h5py.File, shape: list[int], uid: str):
    """
    Update fastload files written by any version.

    This may reduce the stored data as the current rule of thumb is to store only steps that
    completely renew the stored chunk.
    """

    dst["/param/data_version"] = data_version

    metapath = "/meta/" + entry_points["cli_run"]
    if metapath not in src:
        create_check_meta(dst, metapath, dev=True)
        dst[metapath].attrs["uuid"] = uid
    else:
        assert src[metapath].attrs["uuid"] == uid

    root = "/QuasiStatic"
    start = np.zeros(shape, dtype=np.int64)
    for s in sorted([int(i) for i in src["/QuasiStatic"]]):
        if np.all(start == 0) or np.all(src[f"{root}/{s:d}/index"][...] > start):
            start = src[f"{root}/{s:d}/index"][...]
            for key in ["state", "value", "index"]:
                n = f"{root}/{s:d}/{key}"
                dst[n] = src[f"{root}/{s:d}/{key}"][...].reshape(shape)
        else:
            dst[f"{root}/{s:d}"] = dst[f"{root}/{s - 1:d}"]


def _updatedata_1_0(src: h5py.File, dst: h5py.File):
    """
    Update from data_version == 1.0

    Remove initstate, initseq, dynamics (now distinguished by removing m).
    """

    paths = g5.getdatapaths(src)
    rename = {path: path for path in paths}
    rename["/param/normalisation/shape"] = "/param/shape"
    remove = [
        "/param/normalisation/N",
        "/param/dynamics",
        "/param/data_version",
    ]

    for path in paths:
        # renaming
        for r in [
            re.sub(r"(/QuasiStatic/)(x/)([0-9]*)", r"\1u/\3", path),
        ]:
            if path != r:
                rename[path] = r
        # remove
        if re.match(r"(.*)(/)(initstate)(.*)", path):
            s = src[path][...].ravel()
            assert np.all(s == np.arange(s.size))
            remove.append(path)
        if re.match(r"(.*)(/)(initseq)(.*)", path):
            remove.append(path)
            s = src[path][...].ravel()
            assert np.all(s == np.zeros_like(s))

    if "/param/dynamics" in src:
        if src["/param/dynamics"].asstr()[...] == "nopassing":
            remove.append("/param/m")

    for path in remove:
        paths.remove(path)

    new_paths = [rename[path] for path in paths]
    g5.copy(src, dst, paths, new_paths, shallow=True)
    dst["/param/data_version"] = data_version


def _updatedata_pre_1_0(src: h5py.File, dst: h5py.File):
    """
    Update from data_version < 1.0 (at which no data version was stored).

    -   Rename x* -> u*, xyield -> potentials, *neighbours -> *interactions.
    -   Rename parameters for interactions.
    -   Reshape u to 2d if necessary.
    -   Remove initstate, initseq, nchunk, dynamics (now distinguished by removing m).
    """

    paths = g5.getdatapaths(src)

    size = src["param"]["xyield"]["initstate"].size
    is2d = "width" in src["param"]
    width = None if not is2d else src["param"]["width"][...]
    shape = [size] if not is2d else [int(size / width), width]

    dst["/param/shape"] = shape
    dst["/param/data_version"] = data_version

    rename = {path: path for path in paths}
    reshape = {}
    remove = []

    for path in paths:
        # renaming
        for r in [
            re.sub(r"(.*)(/)(xyield)(/)(.*)", r"\1\2potentials\4\5", path),
            re.sub(r"(.*)(/)(x)([/|_|$])(.*)", r"\1\2u\4\5", path),
            re.sub(r"(.*)(/[[fk]_]?)(neighbours)(.*)", r"\1\2interactions\4", path),
            re.sub(r"(/QuasiStatic/)(x/)([0-9]*)", r"\1u/\3", path),
            re.sub(r"(.*/)(x)($)", r"\1u\3", path),
        ]:
            if path != r:
                rename[path] = r
        # reshaping to 2d
        if is2d:
            for r in [
                re.sub(r"(.*)(/)(x)([/])(.*)", r"\1\2u\4\5", path),
                re.sub(r"(.*)(/)(f[_]?\w*)([/])(.*)", r"\1\2\3\4\5", path),
            ]:
                if path != r:
                    reshape[path] = shape
        # remove
        if re.match(r"(.*)(/)(nchunk)(.*)", path):
            remove.append(path)
        if re.match(r"(.*)(/)(initstate)(.*)", path):
            s = src[path][...].ravel()
            assert np.all(s == np.arange(s.size))
            remove.append(path)
        if re.match(r"(.*)(/)(initseq)(.*)", path):
            remove.append(path)
            s = src[path][...].ravel()
            assert np.all(s == np.zeros_like(s))

    if is2d:
        remove.append("/param/width")

    rename["/param/xyield/dx"] = "/param/potentials/du"
    remove.append("/param/normalisation/N")

    if "k4" in src["param"]:
        rename["/param/k4"] = "/param/interactions/k2"
        rename["/param/k4"] = "/param/interactions/k4"
        dst["/param/interactions/type"] = "QuarticGradient"
    elif "a1" in src["param"]:
        rename["/param/a1"] = "/param/interactions/a1"
        rename["/param/a2"] = "/param/interactions/a2"
        dst["/param/interactions/type"] = "Quartic"
    elif "alpha" in src["param"]:
        rename["/param/k_neighbours"] = "/param/interactions/k"
        rename["/param/alpha"] = "/param/interactions/alpha"
        dst["/param/interactions/type"] = "LongRange"
    else:
        rename["/param/k_neighbours"] = "/param/interactions/k"
        dst["/param/interactions/type"] = "Laplace"

    if "/param/potential/name" in src:
        rename["/param/potential/name"] = "/param/potentials/type"
    elif "potential" in src["param"]:
        rename["/param/potential"] = "/param/potentials/type"
    else:
        dst["/param/potentials/type"] = "Cuspy"

    if "kappa" in src["param"]:
        rename["/param/kappa"] = "/param/potential/kappa"

    m = f"/meta/{entry_points['cli_run']}"
    if m in src:
        if "dynamics" in src[m].attrs:
            if src[m].attrs["dynamics"] == "nopassing":
                remove.append("/param/m")

    for path in remove:
        paths.remove(path)

    for path in reshape:
        paths.remove(path)

    new_paths = [rename[path] for path in paths]
    g5.copy(src, dst, paths, new_paths, shallow=True)
    if f"/meta/{entry_points['cli_run']}" in dst:
        del dst[f"/meta/{entry_points['cli_run']}"].attrs["dynamics"]

    if dst["/param/potentials/type"].asstr()[...] == "Cusp":
        dst["/param/potentials/type"][...] = "Cuspy"

    for path in reshape:
        dst[rename[path]] = src[path][...].reshape(shape)


def _get_data_version(file: h5py.File) -> str:
    """
    Get data version from file
    """
    if "/param/data_version" in file:
        return str(file["/param/data_version"].asstr()[...])
    return "0.0"


def cli_force_current_data_version(cli_args=None):
    """
    Add/overwrite "/param/data_version" to the current version.
    Warning: use with caution.
    There are no checks that the data is compatible with the current version.
    """

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))

    parser.add_argument("--no-bak", action="store_true", help="Do not backup before modifying")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Simulation files")

    args = tools._parse(parser, cli_args)

    assert all(os.path.isfile(f) for f in args.files)

    if not args.no_bak:
        assert not any(os.path.isfile(f + ".bak") for f in args.files)

    for filename in tqdm.tqdm(args.files):
        with h5py.File(filename) as file:
            if _get_data_version(file) == data_version:
                continue

        if not args.no_bak:
            shutil.copy2(filename, filename + ".bak")

        with h5py.File(filename, "a") as file:
            if "/param/data_version" in file:
                file["/param/data_version"][...] = data_version
            else:
                file["/param/data_version"] = data_version


def cli_updatedata(cli_args=None):
    """
    Update the data from any version to the current version.
    """

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("--no-bak", action="store_true", help="Do not backup before modifying")
    parser.add_argument("--fastload", action="store_true", help="Update fastload file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Simulation files")
    args = tools._parse(parser, cli_args)

    assert all([os.path.isfile(f) for f in args.files])
    files = []
    fastload = {}
    for filename in tqdm.tqdm(args.files, desc="Reading version"):
        with h5py.File(filename) as file:
            if _get_data_version(file) != data_version:
                files.append(filename)
        path = pathlib.Path(filename2fastload(filename))
        if path.exists() and not path.is_symlink():
            with h5py.File(path) as file:
                if _get_data_version(file) != data_version:
                    fastload[filename] = str(path)
                    if files[-1] != filename:
                        files.append(filename)

    if len(files) == 0 and len(fastload) == 0:
        return

    if len(fastload) > 0 and not args.fastload:
        if not click.confirm("Proceed without updating fastload files?"):
            raise OSError("Cancelled")

    if not args.no_bak:
        assert not any([os.path.isfile(f + ".bak") for f in files])
        assert not any([os.path.isfile(f + ".bak") for f in fastload.values()])

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for filename in tqdm.tqdm(files, desc="Updating data"):
            # run

            changed = True

            with h5py.File(filename) as src, h5py.File(tmp / "my.h5", "w") as dst:
                if "data_version" not in src["param"]:
                    _updatedata_pre_1_0(src, dst)
                elif src["/param/data_version"].asstr()[...] == "1.0":
                    _updatedata_1_0(src, dst)
                else:
                    assert src["/param/data_version"].asstr()[...] == data_version
                    changed = False

            if changed:
                if not args.no_bak:
                    shutil.copy2(filename, filename + ".bak")
                shutil.copy2(tmp / "my.h5", filename)

            # fastload

            if filename in fastload:
                with h5py.File(filename) as src:
                    shape = src["/param/shape"][...]
                    uid = src[f"/meta/{entry_points['cli_run']}"].attrs["uuid"]

                with h5py.File(fastload[filename]) as src, h5py.File(tmp / "my.h5", "w") as dst:
                    _updatedata_fastload(src, dst, shape, uid)

                if not args.no_bak:
                    shutil.copy2(fastload[filename], fastload[filename] + ".bak")
                shutil.copy2(tmp / "my.h5", fastload[filename])


def cli_checkdata(cli_args=None, my_data_version=data_version):
    """
    Check the data file for data version.
    Prints the files that have failed. No output is written if all files are ok.
    """

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-o", "--output", type=str, help="List files that failed the check (yaml).")
    parser.add_argument("files", nargs="*", type=str, help="Files (read only)")
    args = tools._parse(parser, cli_args)

    assert all([os.path.isfile(f) for f in args.files])

    failed = []

    for f in tqdm.tqdm(args.files):
        with h5py.File(f) as file:
            if _get_data_version(file) != my_data_version:
                failed.append(f)

    failed = sorted(failed)

    if args.output is not None:
        shelephant.yaml.dump(args.output, failed)

    if len(failed) > 0:
        print("\n".join(failed))


def dependencies() -> list[str]:
    """
    Return list with version strings.
    Compared to model.System.version_dependencies() this adds the version of prrng.
    """
    return sorted(list(model.version_dependencies()) + ["prrng=" + prrng.version()])


def replace_ep(doc: str) -> str:
    """
    Replace ``:py:func:`...``` with the relevant entry_point name
    """
    for ep in entry_points:
        doc = doc.replace(rf":py:func:`{ep:s}`", entry_points[ep])
    return doc


def interpret_filename(filename: str) -> dict:
    """
    Split filename in useful information.
    """

    part = re.split("_|/", os.path.splitext(filename)[0])
    info = {}

    for i in part:
        key, value = i.split("=")
        info[key] = value

    for key in info:
        info[key] = int(info[key])

    return info


def filename2fastload(filepath):
    """
    Convert a filepath to the corresponding fastload filepath.
    """
    if not re.match(r"(.*)(id=[0-9]*\.h5)", str(filepath)):
        return None
    return re.sub(r"(.*)(id=[0-9]*\.h5)", r"\1fastload_\2", str(filepath))


class Normalisation:
    def __init__(self, file: h5py.File):
        """
        Read normalisation from file.

        :param file: Open simulation HDF5 archive (read-only).
        :return: Basic information as follows::
            mu: Elastic stiffness (float).
            potential: Type of potential (str)
            k_frame: Stiffness of the load-frame (float).
            k_interactions: Stiffness of the interactions (float).
            interactions: Type of interactions (str).
            eta: Damping (float).
            m: Mass (float).
            dynamics: Type of dynamics (str).
            N: Number of blocks (int).
            shape: Shape of the system (tuple of int).
            dt: Time step of time discretisation.
            system: Name of the system, see below (str).
        """

        self.mu = file["param"]["mu"][...]
        self.k_frame = file["param"]["k_frame"][...]
        self.eta = file["param"]["eta"][...]
        if "m" in file["param"]:
            self.m = file["param"]["m"][...]
        else:
            self.m = 0
        self.shape = file["param"]["shape"][...]
        self.N = np.prod(self.shape)
        self.dt = file["param"]["dt"][...]
        self.u = 1
        self.potential = str(file["/param/potentials/type"].asstr()[...])
        self.interactions = str(file["/param/interactions/type"].asstr()[...])
        self.dynamics = "normal" if "m" in file["param"] else "overdamped"

        if self.interactions == "Laplace":
            self.k_interactions = file["param"]["interactions"]["k"][...]
        elif self.interactions == "Quartic":
            self.a1 = file["param"]["interactions"]["a1"][...]
            self.a2 = file["param"]["interactions"]["a2"][...]
        elif self.interactions == "QuarticGradient":
            self.k2 = file["param"]["interactions"]["k2"][...]
            self.k4 = file["param"]["interactions"]["k4"][...]
        elif self.interactions == "LongRange":
            self.k_interactions = file["param"]["interactions"]["k"][...]
            self.alpha = file["param"]["interactions"]["alpha"][...]
        else:
            raise ValueError(f"Unknown interactions: {self.interactions:s}")

        if "u" in file["param"]["normalisation"]:
            self.u = file["param"]["normalisation"]["u"][...]

        if self.potential == "Cuspy":
            self.f = self.mu * self.u
        elif self.potential == "SemiSmooth":
            self.kappa = file["param"]["kappa"][...]
            self.f = self.mu * self.u * (1 - self.mu / (self.mu + self.kappa))
        elif self.potential == "Smooth":
            self.f = self.mu * self.u / np.pi
        else:
            raise ValueError(f"Unknown potential: {self.potential:s}")

        if len(self.shape) == 1:
            self.system = "Line1d"
        elif len(self.shape) == 2:
            self.system = "Line2d"
        else:
            raise ValueError("Unknown shape")

        extra = []
        if self.dynamics == "overdamped":
            extra.append("Nopassing")
        self.system = "_".join([self.system, "System", self.potential, self.interactions] + extra)

    def asdict(self):
        """
        Return relevant parameters as dictionary.
        """
        ret = dict(
            dt=self.dt,
            dynamics=self.dynamics,
            eta=self.eta,
            f=self.f,
            interactions=self.interactions,
            k_frame=self.k_frame,
            m=self.m,
            mu=self.mu,
            N=self.N,
            potential=self.potential,
            shape=self.shape,
            system=self.system,
            u=self.u,
        )

        if self.interactions == "Laplace":
            ret["k_interactions"] = self.k_interactions
        elif self.interactions == "Quartic":
            ret["a1"] = self.a1
            ret["a2"] = self.a2
        elif self.interactions == "QuarticGradient":
            ret["k2"] = self.k2
            ret["k4"] = self.k4
        elif self.interactions == "LongRange":
            ret["k_interactions"] = self.k_interactions
            ret["alpha"] = self.alpha

        if self.potential == "SemiSmooth":
            ret["kappa"] = self.kappa

        return ret


def _common_param(file: h5py.File) -> dict:
    """
    Get common parameters from file.
    """

    file_yield = file["param"]["potentials"]

    ret = {
        "shape": file["param"]["shape"][...],
        "offset": file_yield["xoffset"][...],
        "seed": file["realisation"]["seed"][...],
    }

    if "weibull" in file_yield:
        ret["distribution"] = "weibull"
        ret["parameters"] = [
            file_yield["weibull"]["k"][...],
            2 * file_yield["weibull"]["mean"][...],
            file_yield["weibull"]["offset"][...],
        ]
    elif "delta" in file_yield:
        ret["distribution"] = "delta"
        ret["parameters"] = [2 * file_yield["delta"]["mean"][...], 0]
    elif "random" in file_yield:
        ret["distribution"] = "random"
        ret["parameters"] = [
            2 * file_yield["random"]["mean"][...],
            file_yield["random"]["offset"][...],
        ]
    else:
        raise OSError("Distribution not supported")

    return ret


class SystemExtra:
    """
    Base class for extra system methods and parameters.
    """

    def __init__(self, file: h5py.File):
        assert tag.greater_equal(str(file["/param/data_version"].asstr()[...]), data_version)
        self.normalisation = Normalisation(file)

    def chunk_goto(
        self,
        u: ArrayLike,
        fastload: tuple(str, str) = None,
    ):
        """
        Update the yield positions to be able to accommodate a target position ``u``.
        Note: the position is not updated!

        :param u: Target position.
        :param fastload: If available ``(filename, groupname)`` of the closest fastload info.
        """

        if self.chunk.contains(u):
            return self.chunk.align(u)

        front = self.chunk.data[..., 0]
        back = self.chunk.data[..., -1]

        if np.all(u < 2 * back - front):
            return self.chunk.align(u)

        if fastload is None:
            return self.chunk.align(u)

        if len(fastload) != 2:
            return self.chunk.align(u)

        if fastload[0] is None:
            return self.chunk.align(u)

        if not os.path.exists(fastload[0]):
            return self.chunk.align(u)

        with h5py.File(fastload[0]) as loadfile:
            if fastload[1] in loadfile:
                root = loadfile[fastload[1]]
                self.chunk.restore(
                    state=root["state"][...], value=root["value"][...], index=root["index"][...]
                )

        return self.chunk.align(u)

    def restore_quasistatic_step(
        self,
        root: h5py.Group,
        step: int,
        fastload: bool = True,
    ):
        """
        Quench and restore an a quasi-static step for the relevant root.
        The ``root`` group should contain::

            root["u"][str(step)]   # Positions
            root["inc"][step]      # Increment (-> time)
            root["u_frame"][step]  # Loading frame position

        :param root: HDF5 archive opened in the right root (read-only).
        :param step: Step number.
        :param fastload: Use fastload file (if detected), see :py:func:`cli_generatefastload`.
        """

        u = root["u"][str(step)][...]

        if type(fastload) == tuple or type(fastload) == list:
            pass
        elif fastload:
            fastload = (filename2fastload(root.file.filename), f"/QuasiStatic/{step:d}")
        else:
            fastload = None

        self.chunk_goto(u, fastload)
        self.quench()
        self.inc = root["inc"][step]
        self.u_frame = root["u_frame"][step]
        self.u = u


class Line1d_System_Cuspy_Laplace(model.System_Cuspy_Laplace, SystemExtra):
    def __init__(self, file: h5py.File):
        SystemExtra.__init__(self, file)
        model.System_Cuspy_Laplace.__init__(
            self,
            m=file["param"]["m"][...],
            eta=file["param"]["eta"][...],
            mu=file["param"]["mu"][...],
            k_interactions=file["param"]["interactions"]["k"][...],
            k_frame=file["param"]["k_frame"][...],
            dt=file["param"]["dt"][...],
            **_common_param(file),
        )


class Line1d_System_Cuspy_Laplace_Nopassing(model.System_Cuspy_Laplace_Nopassing, SystemExtra):
    def __init__(self, file: h5py.File):
        SystemExtra.__init__(self, file)
        model.System_Cuspy_Laplace_Nopassing.__init__(
            self,
            m=file["param"]["m"][...],
            eta=file["param"]["eta"][...],
            mu=file["param"]["mu"][...],
            k_interactions=file["param"]["interactions"]["k"][...],
            k_frame=file["param"]["k_frame"][...],
            dt=file["param"]["dt"][...],
            **_common_param(file),
        )


class Line1d_System_SemiSmooth_Laplace(model.System_SemiSmooth_Laplace, SystemExtra):
    def __init__(self, file: h5py.File):
        SystemExtra.__init__(self, file)
        model.System_SemiSmooth_Laplace.__init__(
            self,
            m=file["param"]["m"][...],
            eta=file["param"]["eta"][...],
            mu=file["param"]["mu"][...],
            kappa=file["param"]["kappa"][...],
            k_interactions=file["param"]["interactions"]["k"][...],
            k_frame=file["param"]["k_frame"][...],
            dt=file["param"]["dt"][...],
            **_common_param(file),
        )


class Line1d_System_Smooth_Laplace(model.System_Smooth_Laplace, SystemExtra):
    def __init__(self, file: h5py.File):
        SystemExtra.__init__(self, file)
        model.System_Smooth_Laplace.__init__(
            self,
            m=file["param"]["m"][...],
            eta=file["param"]["eta"][...],
            mu=file["param"]["mu"][...],
            k_interactions=file["param"]["interactions"]["k"][...],
            k_frame=file["param"]["k_frame"][...],
            dt=file["param"]["dt"][...],
            **_common_param(file),
        )


class Line1d_System_Cuspy_LongRange(model.System_Cuspy_LongRange, SystemExtra):
    def __init__(self, file: h5py.File):
        SystemExtra.__init__(self, file)
        model.System_Cuspy_LongRange.__init__(
            self,
            m=file["param"]["m"][...],
            eta=file["param"]["eta"][...],
            mu=file["param"]["mu"][...],
            k_interactions=file["param"]["interactions"]["k"][...],
            alpha=file["param"]["interactions"]["alpha"][...],
            k_frame=file["param"]["k_frame"][...],
            dt=file["param"]["dt"][...],
            **_common_param(file),
        )


class Line1d_System_Cuspy_QuarticGradient(model.System_Cuspy_QuarticGradient, SystemExtra):
    def __init__(self, file: h5py.File):
        SystemExtra.__init__(self, file)
        model.System_Cuspy_QuarticGradient.__init__(
            self,
            m=file["param"]["m"][...],
            eta=file["param"]["eta"][...],
            mu=file["param"]["mu"][...],
            k2=file["param"]["interactions"]["k2"][...],
            k4=file["param"]["interactions"]["k4"][...],
            k_frame=file["param"]["k_frame"][...],
            dt=file["param"]["dt"][...],
            **_common_param(file),
        )


class Line1d_System_Cuspy_Quartic(model.System_Cuspy_Quartic, SystemExtra):
    def __init__(self, file: h5py.File):
        SystemExtra.__init__(self, file)
        model.System_Cuspy_Quartic.__init__(
            self,
            m=file["param"]["m"][...],
            eta=file["param"]["eta"][...],
            mu=file["param"]["mu"][...],
            a1=file["param"]["interactions"]["a1"][...],
            a2=file["param"]["interactions"]["a2"][...],
            k_frame=file["param"]["k_frame"][...],
            dt=file["param"]["dt"][...],
            **_common_param(file),
        )


class Line1d_System_Cuspy_Laplace_RandomForcing(
    model.System_Cuspy_Laplace_RandomForcing, SystemExtra
):
    def __init__(self, file: h5py.File):
        SystemExtra.__init__(self, file)
        model.System_Cuspy_Laplace_RandomForcing.__init__(
            self,
            m=file["param"]["m"][...],
            eta=file["param"]["eta"][...],
            mu=file["param"]["mu"][...],
            k_interactions=file["param"]["interactions"]["k"][...],
            k_frame=file["param"]["k_frame"][...],
            dt=file["param"]["dt"][...],
            mean=file["param"]["thermal"]["mean"][...],
            stddev=file["param"]["thermal"]["stddev"][...],
            seed_forcing=file["param"]["thermal"]["seed"][...],
            dinc_init=file["param"]["thermal"]["dinc_init"][...],
            dinc=file["param"]["thermal"]["dinc"][...],
            **_common_param(file),
        )


class Line2d_System_Cuspy_Laplace(fsb.Line2d.System_Cuspy_Laplace, SystemExtra):
    def __init__(self, file: h5py.File):
        SystemExtra.__init__(self, file)
        fsb.Line2d.System_Cuspy_Laplace.__init__(
            self,
            m=file["param"]["m"][...],
            eta=file["param"]["eta"][...],
            mu=file["param"]["mu"][...],
            k_interactions=file["param"]["interactions"]["k"][...],
            k_frame=file["param"]["k_frame"][...],
            dt=file["param"]["dt"][...],
            **_common_param(file),
        )


def allocate_system(file: h5py.File):
    """
    Allocate the proper system based on the parameters in the file.
    """

    norm = Normalisation(file)

    if norm.system == "Line1d_System_Cuspy_Laplace":
        return Line1d_System_Cuspy_Laplace(file)

    if norm.system == "Line1d_System_SemiSmooth_Laplace":
        return Line1d_System_SemiSmooth_Laplace(file)

    if norm.system == "Line1d_System_Smooth_Laplace":
        return Line1d_System_Smooth_Laplace(file)

    if norm.system == "Line1d_System_Cuspy_Quartic":
        return Line1d_System_Cuspy_Quartic(file)

    if norm.system == "Line1d_System_Cuspy_QuarticGradient":
        return Line1d_System_Cuspy_QuarticGradient(file)

    if norm.system == "Line1d_System_Cuspy_Laplace_RandomForcing":
        return Line1d_System_Cuspy_Laplace_RandomForcing(file)

    if norm.system == "Line1d_System_Cuspy_LongRange":
        return Line1d_System_Cuspy_LongRange(file)

    if norm.system == "Line2d_System_Cuspy_Laplace":
        return Line2d_System_Cuspy_Laplace(file)

    raise ValueError(f"Unknown system: {norm.system}")


def create_check_meta(
    file: h5py.File = None,
    path: str = None,
    dev: bool = False,
    **kwargs,
) -> h5py.Group:
    """
    Create, update, or read/check metadata. This function creates metadata as attributes to a group
    ``path`` as follows::

        "uuid": A unique identifier that can be used to distinguish simulations.
        "version": The current version of this code (updated).
        "dependencies": The current version of all relevant dependencies (updated).
        "compiler": Compiler information (updated).

    :param file: HDF5 archive.
    :param path: Path in ``file`` to store/read metadata.
    :param dev: Allow uncommitted changes.
    :return: Group to metadata.
    """

    deps = sorted(
        list(
            set(
                list(model.version_dependencies())
                + [
                    "prrng=" + prrng.version(),
                    "gooseye=" + eye.version(),
                    "enstat=" + enstat.version,
                ]
            )
        )
    )

    assert dev or not tag.has_uncommitted(version)
    assert dev or not tag.any_has_uncommitted(deps)

    if file is None:
        return None

    if path not in file:
        meta = file.create_group(path)
        meta.attrs["uuid"] = str(uuid.uuid4())
        meta.attrs["version"] = version
        meta.attrs["dependencies"] = deps
        meta.attrs["compiler"] = model.version_compiler()
        for key in kwargs:
            meta.attrs[key] = kwargs[key]
        return meta

    meta = file[path]
    if file.mode in ["r+", "w", "a"]:
        assert dev or tag.greater_equal(version, meta.attrs["version"])
        assert dev or tag.all_greater_equal(deps, meta.attrs["dependencies"])
        meta.attrs["version"] = version
        meta.attrs["dependencies"] = deps
        meta.attrs["compiler"] = model.version_compiler()
    else:
        assert dev or tag.equal(version, meta.attrs["version"])
        assert dev or tag.all_equal(deps, meta.attrs["dependencies"])
    return meta


def generate(
    file: h5py.File,
    shape: list[int],
    seed: int,
    eta: float = None,
    dt: float = None,
    k_frame: float = None,
    potential: dict = {"type": "Cuspy", "mu": 1.0},
    distribution: str = "weibull",
    interactions: dict = {"type": "Laplace", "k": 1.0},
    overdamped: bool = False,
):
    """
    Generate a simulation file.

    :param file: HDF5 file opened for writing.
    :param shape: Shape of the system.
    :param seed: Base seed.
    :param eta: Damping coefficient.
    :param dt: Time step.
    :param k_frame: Frame stiffness. Default: ``1 / L**2``.
    :param potential: Select potential.
    :param distribution: Distribution of potentials.
    :param interactions: Select interactions.
    :param overdamped: Run overdamped dynamics (no passing rule if quasi-static).
    """

    L = min(shape)

    if eta is None and dt is None:
        eta = 1
        dt = 1
    elif dt is None:
        known_eta = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
        known_dt = np.array([5e-3, 5e-3, 5e-2, 5e-2, 5e-3, 5e-4])
        dt = np.interp(eta, known_eta, known_dt)
    else:
        assert eta is not None

    file["/param/interactions/type"] = interactions["type"]
    if interactions["type"] == "Laplace":
        file["/param/interactions/k"] = interactions["k"]
    elif interactions["type"] == "Quartic":
        file["/param/interactions/a1"] = interactions["a1"]
        file["/param/interactions/a2"] = interactions["a2"]
    elif interactions["type"] == "QuarticGradient":
        file["/param/interactions/k2"] = interactions["k2"]
        file["/param/interactions/k4"] = interactions["k4"]
    elif interactions["type"] == "LongRange":
        file["/param/interactions/k"] = interactions["k"]
        file["/param/interactions/alpha"] = interactions["alpha"]
    else:
        raise ValueError(f"Unknown interactions: {interactions['type']}")

    file["/realisation/seed"] = seed
    if not overdamped:
        file["/param/m"] = 1.0
    file["/param/eta"] = eta
    file["/param/mu"] = potential["mu"]
    file["/param/k_frame"] = 1.0 / L**2
    file["/param/dt"] = dt
    file["/param/potentials/type"] = potential["type"]
    file["/param/potentials/xoffset"] = -100
    file["/param/potentials/du"] = 1e-3

    if distribution.lower() == "weibull":
        file["/param/potentials/weibull/offset"] = 1e-5
        file["/param/potentials/weibull/mean"] = 1
        file["/param/potentials/weibull/k"] = 2
    elif distribution.lower() == "delta":
        file["/param/potentials/delta/mean"] = 1
    elif distribution.lower() == "random":
        file["/param/potentials/random/offset"] = 1e-5
        file["/param/potentials/random/mean"] = 1
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    if k_frame is not None:
        file["/param/k_frame"][...] = k_frame

    file["/param/shape"] = shape
    file["/param/normalisation/u"] = 1
    file["/param/data_version"] = data_version


def _generate_cli_options(parser):
    parser.add_argument("-n", "--nsim", type=int, default=1, help="#simulations")
    parser.add_argument("-s", "--start", type=int, default=0, help="Start simulation")
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")

    parser.add_argument("--size", type=int, help="1d system")
    parser.add_argument("--shape", nargs=2, type=int, help="2d system")

    parser.add_argument("--dt", type=float, help="Time-step")
    parser.add_argument("--eta", type=float, help="Damping coefficient")
    parser.add_argument("--overdamped", action="store_true", help="Overdamped dynamics")
    parser.add_argument(
        "--nstep", default=20000, help="#load-steps to run", type=lambda arg: int(float(arg))
    )

    parser.add_argument("--kframe", type=float, help="Overwrite k_frame")

    parser.add_argument("--distribution", type=str, default="Weibull", help="Distribution type")
    parser.add_argument("--cuspy", nargs=1, type=float, help="Smooth potential: mu")
    parser.add_argument("--smooth", nargs=1, type=float, help="Smooth potential: mu")
    parser.add_argument("--semismooth", nargs=2, type=float, help="Smooth potential: mu, kappa")

    parser.add_argument("--laplace", nargs=1, type=float, help="Laplace interactions: k")
    parser.add_argument("--quartic", nargs=2, type=float, help="Quartic interactions: a1, a2")
    parser.add_argument("--quarticgradient", nargs=2, type=float, help="Quartic gradient: k2, k4")
    parser.add_argument("--longrange", nargs=2, type=float, help="LongRange interactions: k, alpha")

    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("outdir", type=str, help="Output directory")


def _generate_parse(args):
    assert args.size or args.shape
    assert sum([args.cuspy is not None, args.smooth is not None, args.semismooth is not None]) <= 1
    assert (
        sum(
            [
                args.laplace is not None,
                args.quartic is not None,
                args.quarticgradient is not None,
                args.longrange is not None,
            ]
        )
        <= 1
    )

    potential = {"type": "Cuspy", "mu": 1.0}
    interactions = {"type": "Laplace", "k": 1.0}

    if args.cuspy is not None:
        potential = {"type": "Cuspy", "mu": args.cuspy[0]}
    if args.smooth is not None:
        potential = {"type": "Smooth", "mu": args.smooth[0]}
    if args.semismooth is not None:
        potential = {"type": "SemiSmooth", "mu": args.semismooth[0], "kappa": args.semismooth[1]}

    if args.laplace is not None:
        interactions = {"type": "Laplace", "k": args.laplace[0]}
    if args.quartic is not None:
        interactions = {"type": "Quartic", "a1": args.quartic[0], "a2": args.quartic[1]}
    if args.quarticgradient is not None:
        interactions = {
            "type": "QuarticGradient",
            "k2": args.quarticgradient[0],
            "k4": args.quarticgradient[1],
        }
    if args.longrange is not None:
        interactions = {
            "type": "LongRange",
            "k": args.longrange[0],
            "alpha": args.longrange[1],
        }

    ret = {}
    ret["shape"] = [args.size] if args.shape is None else args.shape
    ret["eta"] = args.eta
    ret["dt"] = args.dt
    ret["k_frame"] = args.kframe
    ret["potential"] = potential
    ret["distribution"] = args.distribution
    ret["interactions"] = interactions

    return ret


def cli_generate(cli_args=None):
    """
    Generate IO files (including job-scripts) to run simulations.
    """

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    _generate_cli_options(parser)
    args = tools._parse(parser, cli_args)

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    opts = _generate_parse(args)

    n = args.size if args.shape is None else np.prod(args.shape)
    assert not any(
        [(outdir / f"id={i:04d}.h5").exists() for i in range(args.start, args.start + args.nsim)]
    )
    files = []
    for i in range(args.start, args.start + args.nsim):
        files += [f"id={i:04d}.h5"]
        seed = i * n
        with h5py.File(outdir / files[-1], "w") as file:
            generate(
                file=file,
                seed=seed,
                **opts,
            )

    executable = entry_points["cli_run"]
    commands = [f"{executable} --nstep {args.nstep:d} {file}" for file in files]
    info = [f"{entry_points['cli_ensembleinfo']} id=[0-9]*.h5"]
    shelephant.yaml.dump(outdir / "commands_run.yaml", commands, force=True)
    shelephant.yaml.dump(outdir / "commands_info.yaml", info, force=True)


def cli_run(cli_args=None):
    """
    Run simulation.
    """

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    progname = entry_points[funcname]

    parser.add_argument("--check", type=int, help="Rerun step to check old run / new version")
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument(
        "--fixed-step",
        action="store_true",
        help="Use a fixed loading-step instead for the event-driven protocol",
    )
    parser.add_argument("-n", "--nstep", type=int, default=5000, help="Total #load-steps to run")
    parser.add_argument("file", type=str, help="Input/output file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    basename = os.path.basename(args.file)

    with h5py.File(args.file, "a") as file, h5py.File(filename2fastload(args.file), "a") as fload:
        system = allocate_system(file)
        meta = dict(loading="event-driven")

        if args.fixed_step:
            meta["loading"] = "fixed-step"
            dx_particle = 1e-3 * system.normalisation.u
            dx_frame = (
                dx_particle
                * (system.normalisation.k_frame + system.normalisation.mu)
                / system.normalisation.k_frame
            )

        metapath = f"/meta/{progname}"
        create_check_meta(file, metapath, dev=args.develop, **meta)

        if "QuasiStatic" in fload:
            assert fload[metapath].attrs["uuid"] == file[metapath].attrs["uuid"]
        else:
            create_check_meta(fload, metapath, dev=args.develop, **meta)
            fload[metapath].attrs["uuid"] = file[metapath].attrs["uuid"]
            fload["/param/data_version"] = data_version

        if "QuasiStatic" not in file:
            system.minimise()
            system.inc = 0

            file.create_group("/QuasiStatic/u").create_dataset("0", data=system.u)
            root = file["/QuasiStatic"]
            storage.create_extendible(root, "inc", np.uint64, desc="'Time' (increment number).")
            storage.create_extendible(root, "u_frame", np.float64, desc="Position of load frame.")
            storage.create_extendible(root, "kick", bool, desc="Kick used.")
            storage.dset_extend1d(root, "inc", 0, system.inc)
            storage.dset_extend1d(root, "u_frame", 0, system.u_frame)
            storage.dset_extend1d(root, "kick", 0, True)
            file.flush()
        else:
            root = file["/QuasiStatic"]

        if args.check is not None:
            assert args.check < root["inc"].size
            args.nstep = 1
            step = args.check
        else:
            step = root["inc"].size

        kick = root["kick"][step - 1]
        du = file["/param/potentials/du"][...]
        system.restore_quasistatic_step(root, step - 1)

        desc = f"{basename}: step = {step:8d}, niter = {'-':8s}"
        pbar = tqdm.tqdm(range(step, step + args.nstep), desc=desc)
        start = np.zeros_like(system.chunk.start)

        for step in pbar:
            if args.fixed_step:
                kick = True
                system.u_frame += dx_frame
            else:
                kick = not kick
                system.eventDrivenStep(du, kick)

            if kick:
                inc_n = system.inc
                system.minimise()
                niter = system.inc - inc_n
                pbar.set_description(f"{basename}: step = {step:8d}, niter = {niter:8d}")
                pbar.refresh()

            if args.check is not None:
                assert root["inc"][step] == system.inc
                assert root["kick"][step] == kick
                assert np.isclose(root["u_frame"][step], system.u_frame)
                assert np.allclose(root["u"][str(step)][...], system.u)
                break

            storage.dset_extend1d(root, "inc", step, system.inc)
            storage.dset_extend1d(root, "u_frame", step, system.u_frame)
            storage.dset_extend1d(root, "kick", step, kick)
            root["u"][str(step)] = system.u
            file.flush()

            if np.all(system.chunk.start > start) or np.all(start == 0):
                start = np.copy(system.chunk.start)
                fload[f"/QuasiStatic/{step:d}/state"] = system.chunk.state_at(start)
                fload[f"/QuasiStatic/{step:d}/index"] = start
                fload[f"/QuasiStatic/{step:d}/value"] = system.chunk.data[..., 0]
            else:
                fload[f"/QuasiStatic/{step:d}"] = fload[f"/QuasiStatic/{step - 1:d}"]
            fload.flush()


def cli_checkdynamics(cli_args=None):
    """
    Write or check the detailed dynamics of a quasi-static step.
    """

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))

    parser.add_argument("--step", type=int, required=True, help="Step to rerun")
    parser.add_argument("--write", type=str, help="Write details to file")
    parser.add_argument("--read", type=str, help="Read details from file and compare")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file (read-only)")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    assert args.write is not None or args.read is not None
    outpath = args.write if args.write is not None else args.read
    mode = "w" if args.write is not None else "r"

    with h5py.File(args.file) as file, h5py.File(outpath, mode) as output:
        system = allocate_system(file)
        root = file["/QuasiStatic"]
        step = args.step
        assert step < root["inc"].size
        kick = root["kick"][step - 1]
        du = file["/param/potentials/du"][...]
        system.restore_quasistatic_step(root, step - 1)

        # todo: fixed-step loading
        kick = not kick
        system.eventDrivenStep(du, kick)

        for i in tqdm.tqdm(range(int(root["inc"][step] - system.inc))):
            system.timeStep()

            if mode == "w":
                idx = system.chunk.chunk_index_at_align.ravel()
                allp = np.arange(idx.size)
                data = system.chunk.data.reshape(-1, system.chunk.chunk_size)
                output[f"/iter/{i:d}/f_inter"] = system.f_interactions
                output[f"/iter/{i:d}/f_pot"] = system.f_potential
                output[f"/iter/{i:d}/f_frame"] = system.f_frame
                output[f"/iter/{i:d}/f"] = system.f
                output[f"/iter/{i:d}/u"] = system.u
                output[f"/iter/{i:d}/v"] = system.v
                output[f"/iter/{i:d}/y_i-1"] = data[allp, idx - 1]
                output[f"/iter/{i:d}/y_i"] = system.chunk.left_of_align
                output[f"/iter/{i:d}/y_i+1"] = system.chunk.right_of_align
                output[f"/iter/{i:d}/y_i+2"] = data[allp, idx + 2]
                output[f"/iter/{i:d}/index"] = system.chunk.index_at_align
                output.flush()
                continue

            try:
                assert np.allclose(output[f"/iter/{i:d}/f"][...], system.f)
                assert np.allclose(output[f"/iter/{i:d}/u"][...], system.u)
                assert np.allclose(output[f"/iter/{i:d}/v"][...], system.v)
                assert np.allclose(output[f"/iter/{i:d}/y_i"][...], system.chunk.left_of_align)
                assert np.allclose(output[f"/iter/{i:d}/y_i+1"][...], system.chunk.right_of_align)
                assert np.allclose(output[f"/iter/{i:d}/index"][...], system.chunk.index_at_align)
            except AssertionError:
                passed = (
                    np.isclose(output[f"/iter/{i:d}/f_inter"][...], system.f_interactions)
                    & np.isclose(output[f"/iter/{i:d}/f_pot"][...], system.f_potential)
                    & np.isclose(output[f"/iter/{i:d}/f_frame"][...], system.f_frame)
                    & np.isclose(output[f"/iter/{i:d}/f"][...], system.f)
                    & np.isclose(output[f"/iter/{i:d}/u"][...], system.u)
                    & np.isclose(output[f"/iter/{i:d}/v"][...], system.v)
                    & np.isclose(output[f"/iter/{i:d}/y_i"][...], system.chunk.left_of_align)
                    & np.isclose(output[f"/iter/{i:d}/y_i+1"][...], system.chunk.right_of_align)
                    & np.isclose(output[f"/iter/{i:d}/index"][...], system.chunk.index_at_align)
                )

                idx = system.chunk.chunk_index_at_align.ravel()
                allp = np.arange(idx.size)
                data = system.chunk.data.reshape(-1, system.chunk.chunk_size)

                u_n = system.u
                v_n = system.v
                f_n = system.f
                n_n = system.f_interactions
                p_n = system.f_potential
                f_f = system.f_frame
                c_n = system.chunk.chunk_index_at_align
                i_n = system.chunk.index_at_align
                m1_n = data[allp, idx - 1]
                m0_n = data[allp, idx]
                p1_n = data[allp, idx + 1]
                p2_n = data[allp, idx + 2]

                assert np.allclose(m0_n, system.chunk.left_of_align)
                assert np.allclose(p1_n, system.chunk.right_of_align)

                table = prettytable.PrettyTable()
                table.field_names = [
                    "p",
                    "index",
                    "chunk_index",
                    "diff_index",
                    "diff_fn",
                    "diff_fp",
                    "diff_ff",
                    "diff_f",
                    "diff_u",
                    "diff_v",
                    "diff_y(i-1)",
                    "diff_y(i)",
                    "diff_y(i+1)",
                    "diff_y(i+2)",
                    "sorted",
                ]
                for p in np.argwhere(~passed).ravel():
                    table.add_row(
                        [
                            p,
                            i_n[p],
                            c_n[p],
                            int(output[f"/iter/{i:d}/index"][p] - i_n[p]),
                            np.abs((output[f"/iter/{i:d}/f_inter"][p] - n_n[p]) / n_n[p]),
                            np.abs((output[f"/iter/{i:d}/f_pot"][p] - p_n[p]) / p_n[p]),
                            np.abs((output[f"/iter/{i:d}/f_frame"][p] - f_f[p]) / f_f[p]),
                            np.abs((output[f"/iter/{i:d}/f"][p] - f_n[p]) / f_n[p]),
                            np.abs((output[f"/iter/{i:d}/u"][p] - u_n[p]) / u_n[p]),
                            np.abs((output[f"/iter/{i:d}/v"][p] - v_n[p]) / v_n[p]),
                            np.abs((output[f"/iter/{i:d}/y_i-1"][p] - m1_n[p]) / m1_n[p]),
                            np.abs((output[f"/iter/{i:d}/y_i"][p] - m0_n[p]) / m0_n[p]),
                            np.abs((output[f"/iter/{i:d}/y_i+1"][p] - p1_n[p]) / p1_n[p]),
                            np.abs((output[f"/iter/{i:d}/y_i+2"][p] - p2_n[p]) / p2_n[p]),
                            int(~(m1_n[p] <= u_n[p] < m0_n[p] < p1_n[p] < p2_n[p])),
                        ]
                    )
                table.set_style(prettytable.SINGLE_BORDER)
                print("")
                print(table.get_string(sortby="p"))
                raise AssertionError(f"Assertion failed at iteration {i:d}")


def steadystate(
    u_frame: ArrayLike, f_frame: ArrayLike, kick: ArrayLike, A: ArrayLike, N: int, **kwargs
) -> int:
    """
    Estimate the first step of the steady-state. Constraints:

    -   Start with elastic loading.
    -   Sufficiently low tangent modulus.
    -   All blocks yielded at least once.

    .. note::

        Keywords arguments that are not explicitly listed are ignored.

    :param u_frame: Position of the load frame [nstep].
    :param f_frame: Average driving force [nstep].
    :param kick: Whether a kick was applied [nstep].
    :param A: Number of blocks that yielded at least once [nstep].
    :param N: Number of blocks.
    :return: Step number.
    """

    if f_frame.size <= 3:
        return None

    tangent = np.empty_like(f_frame)
    tangent[0] = np.inf
    tangent[1:] = (f_frame[1:] - f_frame[0]) / (u_frame[1:] - u_frame[0])

    i_yield = np.argmax(A == N)
    i_tangent = np.argmax(tangent <= 0.95 * tangent[~np.isnan(tangent)][1])
    steadystate = max(i_yield + 1, i_tangent)

    if i_yield == 0 or i_tangent == 0:
        return None

    if steadystate >= kick.size - 1:
        return None

    if kick[steadystate]:
        steadystate += 1

    return steadystate


def basic_output(file: h5py.File) -> dict:
    """
    Read basic output from simulation.

    :param file: Open simulation HDF5 archive (read-only).

    :return: Basic output as follows::
        u_frame: Position of the load frame [nstep].
        f_frame: Average driving force [nstep].
        f_potential: Average elastic force [nstep].
        S: Number of times a particle yielded [nstep].
        A: Number of particles that yielded at least once [nstep].
        kick: Step started with a kick (True), or contains only elastic loading (False) [nstep].
        step: Step numbers == np.arange(nstep).
        steadystate: Increment number where the steady state starts (int).
    """

    system = allocate_system(file)
    ret = {}

    if "QuasiStatic" not in file:
        return ret

    root = file["QuasiStatic"]
    nstep = root["inc"].size
    steps = np.arange(nstep)
    ret["u_frame"] = root["u_frame"][...]
    ret["f_frame"] = np.empty((nstep), dtype=float)
    ret["f_potential"] = np.empty((nstep), dtype=float)
    ret["S"] = np.empty((nstep), dtype=int)
    ret["A"] = np.empty((nstep), dtype=int)
    ret["kick"] = root["kick"][...].astype(bool)
    ret["step"] = steps

    for j, step in enumerate(tqdm.tqdm(steps)):
        system.restore_quasistatic_step(root, step)

        if j == 0:
            i_n = np.copy(system.chunk.index_at_align)

        i = system.chunk.index_at_align
        assert np.isclose(ret["u_frame"][step], system.u_frame)
        ret["f_frame"][step] = np.mean(system.f_frame)
        ret["f_potential"][step] = -np.mean(system.f_potential)
        ret["S"][step] = np.sum(i - i_n)
        ret["A"][step] = np.sum(i != i_n)
        i_n = np.copy(i)

    ret["steadystate"] = steadystate(N=system.size, **ret)
    ret["u_frame"] /= system.normalisation.u
    ret["f_frame"] /= system.normalisation.f
    ret["f_potential"] /= system.normalisation.f

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    tools.check_docstring(doc, ret, ":return:")

    return ret


def _check_normalisation(norm: dict, test: dict):
    """
    Check if normalisations are the same.

    :param a: First dictionary.
    :param b: Second dictionary.
    :return: True if normalisation is the same.
    """

    for key in norm:
        if key in ["interactions", "potential", "system", "dynamics"]:
            assert str(norm[key]) == str(test[key])
        elif key == "shape":
            assert list(norm[key]) == list(test[key])
        else:
            assert np.isclose(norm[key], test[key])


def cli_ensembleinfo(cli_args=None):
    """
    Read information (avalanche size, force) of an ensemble,
    see :py:func:`basic_output`.
    Store into a single output file.
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    progname = entry_points[funcname]
    output = file_defaults[funcname]

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Files to read")

    args = tools._parse(parser, cli_args)
    assert len(args.files) > 0
    assert all([os.path.isfile(file) for file in args.files])
    assert list(np.unique(args.files)) == args.files
    tools._check_overwrite_file(args.output, args.force)
    info = dict(
        filepath=[os.path.relpath(i, os.path.dirname(args.output)) for i in args.files],
        seed=[],
        uuid=[],
        version=[],
    )

    fmt = "{:" + str(max(len(i) for i in info["filepath"])) + "s}"
    pbar = tqdm.tqdm(info["filepath"], desc=fmt.format(""))

    with h5py.File(args.output, "w") as output:
        create_check_meta(output, f"/meta/{progname}", dev=args.develop)

        for i, (filename, filepath) in enumerate(zip(pbar, args.files)):
            pbar.set_description(fmt.format(filename), refresh=True)

            with h5py.File(filepath) as file:
                if i == 0:
                    g5.copy(file, output, "/param")
                    norm = Normalisation(file).asdict()
                else:
                    _check_normalisation(norm, Normalisation(file).asdict())

                out = basic_output(file)

                if i == 0:
                    fields_full = [key for key in out if key not in ["steadystate"]]
                    combine_load = {key: [] for key in fields_full}
                    combine_kick = {key: [] for key in fields_full}
                    file_load = []
                    file_kick = []

                info["seed"].append(file["/realisation/seed"][...])

                meta = file[f"/meta/{entry_points['cli_run']}"]
                for key in ["uuid", "version"]:
                    info[key].append(meta.attrs[key])

                if "step" not in out:
                    continue

                for key in fields_full:
                    output[f"/full/{filename}/{key}"] = out[key]
                if out["steadystate"] is not None:
                    output[f"/full/{filename}/steadystate"] = out["steadystate"]
                output.flush()

                if out["steadystate"] is None:
                    continue

                if all(out["kick"]):
                    continue

                assert all(out["kick"][::2])
                assert not any(out["kick"][1::2])
                kick = np.copy(out["kick"])
                load = np.logical_not(out["kick"])
                kick[: out["steadystate"]] = False
                load[: out["steadystate"]] = False

                if np.sum(load) > np.sum(kick):
                    load[-1] = False

                assert np.sum(load) == np.sum(kick)

                for key in combine_load:
                    combine_load[key] += list(out[key][load])
                    combine_kick[key] += list(out[key][kick])

                file_load += list(i * np.ones(np.sum(load), dtype=int))
                file_kick += list(i * np.ones(np.sum(kick), dtype=int))

        # store steady-state of full ensemble together

        combine_load["file"] = np.array(file_load, dtype=np.uint64)
        combine_kick["file"] = np.array(file_kick, dtype=np.uint64)

        for key in ["A", "step"]:
            combine_load[key] = np.array(combine_load[key], dtype=np.uint64)
            combine_kick[key] = np.array(combine_kick[key], dtype=np.uint64)

        for key in ["u_frame", "f_frame", "f_potential"]:
            combine_load[key] = np.array(combine_load[key])
            combine_kick[key] = np.array(combine_kick[key])

        for key in combine_load:
            output[f"/loading/{key}"] = combine_load[key]
            output[f"/avalanche/{key}"] = combine_kick[key]

        for key, value in norm.items():
            output[f"/normalisation/{key}"] = value

        # extract ensemble averages

        ss = np.equal(combine_kick["A"], norm["N"])
        assert all(np.equal(combine_kick["step"][ss], combine_load["step"][ss] + 1))
        output["/averages/f_frame_top"] = np.mean(combine_load["f_frame"][ss])
        output["/averages/f_frame_bot"] = np.mean(combine_kick["f_frame"][ss])

        # store metadata at runtime for each input file

        for key in info:
            assert len(info[key]) == len(info["filepath"])

        output["/lookup/filepath"] = info["filepath"]
        output["/lookup/seed"] = info["seed"]
        output["/lookup/uuid"] = info["uuid"]
        tools.h5py_save_unique(info["version"], output, "/lookup/version", asstr=True)
        output["files"] = output["/lookup/filepath"]


def cli_job_rerun(cli_args=None):
    """
    Write list of jobs to rerun a quasi-static step.
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("--systemspanning", action="store_true", help="System spanning events")
    parser.add_argument("--eventmap", action="store_true", help="Produce event map")
    parser.add_argument("--relaxation", action="store_true", help="Measure relaxation")
    parser.add_argument("--nsim", type=int, help="Select #simulations randomly")
    parser.add_argument("-e", "--executable", type=str, help="Executable")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("info", type=str, help="EnsembleInfo")
    parser.add_argument("output", type=str, help="Output file (yaml)")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.info)

    assert sum([args.executable is not None, args.eventmap, args.relaxation]) == 1

    with h5py.File(args.info) as file:
        step = file["/avalanche/step"][...]
        A = file["/avalanche/A"][...]
        S = file["/avalanche/S"][...]
        N = file["/normalisation/N"][...]
        fname = file["/lookup/filepath"].asstr()[...][file["/avalanche/file"][...]]
        is2d = file["/normalisation/shape"].size == 2

    if args.systemspanning:
        keep = A == N
        step = step[keep]
        S = S[keep]
        fname = list(fname[keep])

    ret = []
    root = pathlib.Path(os.path.relpath(args.info, pathlib.Path(args.output).parent)).parent
    rname = [str(root / f) for f in fname]
    fname = [os.path.normpath(f).split(".h5")[0] for f in fname]

    if args.eventmap:
        if is2d:
            opts = "-s -u"
        else:
            opts = "-s"
        ret = [
            f"EventMap_run {opts} --step {s:d} --smax {smax:d} {r} -o {f}_step={s:d}.h5"
            for s, smax, r, f in zip(step, S, rname, fname)
        ]
    elif args.relaxation is not None:
        ret = [
            f"Relaxation_Run --step {s:d} {r} -o {f}_step={s:d}.h5"
            for s, r, f in zip(step, rname, fname)
        ]
    elif args.executable is not None:
        ret = [
            f"{args.executable} --step {s:d} {r} -o {f}_step={s:d}.h5"
            for s, r, f in zip(step, rname, fname)
        ]

    if args.nsim is not None:
        sorter = np.arange(len(ret))
        np.random.shuffle(sorter)
        ret = [ret[i] for i in sorter[: args.nsim]]

    shelephant.yaml.dump(args.output, ret, force=True)

    if cli_args is not None:
        return ret


def cli_checkfastload(cli_args=None):
    """
    Check the integrity of the fast load file.
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file (read only)")
    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)

    fload = filename2fastload(args.file)
    assert os.path.isfile(fload)

    with h5py.File(args.file) as file, h5py.File(fload) as fload:
        system = allocate_system(file)

        if "meta" in fload:
            # files <11.7 do not have metadata
            metapath = f"/meta/{entry_points['cli_run']}"
            assert fload[metapath].attrs["uuid"] == file[metapath].attrs["uuid"]

        for step in tqdm.tqdm(sorted([int(i) for i in fload["QuasiStatic"]])):
            state = fload[f"/QuasiStatic/{step:d}/state"][...]
            index = fload[f"/QuasiStatic/{step:d}/index"][...]
            value = fload[f"/QuasiStatic/{step:d}/value"][...]
            system.chunk.align_at(index)
            assert np.allclose(system.chunk.left_of_align, value)
            assert np.all(system.chunk.state_at(index) == state)


def cli_generatefastload(cli_args=None):
    """
    Save the state of the random generators for fast loading of the simulation.
    The data created by this function is just to speed-up processing,
    it is completely obsolete and can be removed without hesitation.
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("--append", action="store_true", help="Append existing file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("file", type=str, help="Simulation file")
    args = tools._parse(parser, cli_args)
    progname = entry_points["cli_run"]
    metapath = f"/meta/{progname}"

    output = filename2fastload(args.file)
    if args.append:
        assert os.path.isfile(output)
    else:
        tools._check_overwrite_file(output, args.force)

    with h5py.File(args.file) as file, h5py.File(output, "r+" if args.append else "w") as fload:
        if not args.append:
            create_check_meta(fload, metapath, dev=args.develop)
            fload[metapath].attrs["uuid"] = file[metapath].attrs["uuid"]
        else:
            assert fload[metapath].attrs["uuid"] == file[metapath].attrs["uuid"]

        system = allocate_system(file)
        root = file["QuasiStatic"]
        start = np.zeros_like(system.chunk.start)

        if args.append:
            if "QuasiStatic" in output:
                stored = np.sort(np.array([int(i) for i in output["QuasiStatic"]]))

        for step in tqdm.tqdm(range(root["inc"].size)):
            if args.append:
                if step in stored:
                    continue
                load = stored[np.argmax(stored > step)]
                system.restore_quasistatic_step(root, load)

            system.restore_quasistatic_step(root, step, fastload=False)

            if np.all(system.chunk.start > start) or np.all(start == 0):
                start = np.copy(system.chunk.start)
                fload[f"/QuasiStatic/{step:d}/state"] = system.chunk.state_at(start)
                fload[f"/QuasiStatic/{step:d}/index"] = start
                fload[f"/QuasiStatic/{step:d}/value"] = system.chunk.data[..., 0]
            else:
                fload[f"/QuasiStatic/{step:d}"] = fload[f"/QuasiStatic/{step - 1:d}"]
            fload.flush()


def cli_plotstateaftersystemspanning(cli_args=None):
    """
    Plot state after system-spanning events.
    Input files: :py:func:`cli_ensembleinfo`, or ?? (TODO)
    """

    import GooseMPL as gplt  # noqa: F401
    import matplotlib.pyplot as plt  # noqa: F401

    plt.style.use(["goose", "goose-latex", "goose-autolayout"])

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))

    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Input files")

    args = tools._parse(parser, cli_args)
    assert np.all([os.path.isfile(f) for f in args.files])

    info = None
    state = None
    structure = None
    realisation = None

    for file in args.files:
        with h5py.File(file) as f:
            if "full" in f:
                info = file
            if "heightheight" in f:
                state = file
            if "q" in f:
                structure = file

    if info is not None:
        with h5py.File(info) as f:
            basedir = pathlib.Path(info).parent
            paths = f["/lookup/filepath"].asstr()[...]
            file = f["/avalanche/file"][...]
            step = f["/avalanche/step"][...]
            A = f["/avalanche/A"][...]
            N = f["/normalisation/N"][...]
            keep = A == N
            file = file[keep]
            step = step[keep]
            for i in range(file.size):
                if (basedir / paths[file[i]]).exists():
                    realisation = basedir / paths[file[i]]
                    step = step[i]
                    break

    ncols = int(state is not None) + int(structure is not None) + int(realisation is not None)
    fig, axes = gplt.subplots(ncols=ncols)
    icol = 0
    if ncols == 1:
        axes = [axes]

    if realisation is not None:
        with h5py.File(realisation) as f:
            u = f[f"/QuasiStatic/u/{step:d}"][...]
            axes[icol].plot(u - np.mean(u))
            icol += 1

    if state is not None:
        with h5py.File(state) as f:
            x = f["/heightheight/A"][...]
            y = f["/heightheight/R"][...]
            axes[icol].plot(x, y)
            axes[icol].set_xscale("log")
            axes[icol].set_yscale("log")
            keep = x < 200
            gplt.fit_powerlaw(
                x[keep], y[keep], axis=axes[icol], c="r", extrapolate=True, auto_fmt="r"
            )
            axes[icol].legend()
            icol += 1

    if structure is not None:
        with h5py.File(structure) as f:
            x = f["/q"][...]
            y = enstat.static.restore(
                first=f["first"][...],
                second=f["second"][...],
                norm=f["norm"][...],
            ).mean()
            axes[icol].plot(x, y)
            axes[icol].set_xscale("log")
            axes[icol].set_yscale("log")
            gplt.fit_powerlaw(x, y, axis=axes[icol], c="r", extrapolate=True, auto_fmt="q")
            axes[icol].legend()
            icol += 1

    plt.show()
    plt.close(fig)


def cli_stateaftersystemspanning(cli_args=None):
    """
    Extract:

    -   P(x), with x the distance to yielding.
    -   The height-height correlation.
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    output = file_defaults[funcname]

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-n", "--select", type=int, help="Select random subset")
    parser.add_argument("ensembleinfo", type=str, help="EnsembleInfo")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.ensembleinfo)
    tools._check_overwrite_file(args.output, args.force)
    basedir = os.path.dirname(args.ensembleinfo)

    with h5py.File(args.ensembleinfo) as info:
        assert np.all([os.path.exists(os.path.join(basedir, file)) for file in info["full"]])

        paths = info["/lookup/filepath"].asstr()[...]
        file = info["/avalanche/file"][...]
        step = info["/avalanche/step"][...]
        A = info["/avalanche/A"][...]
        N = info["/normalisation/N"][...]
        shape = info["/normalisation/shape"][...]
        L = min(shape)

        keep = A == N
        file = file[keep]
        step = step[keep]

    hist_x_log = enstat.histogram(bin_edges=np.logspace(-4, 1, 20001), bound_error="norm")
    hist_xr_log = enstat.histogram(bin_edges=np.logspace(-4, 1, 20001), bound_error="norm")
    hist_xl_log = enstat.histogram(bin_edges=np.logspace(-4, 1, 20001), bound_error="norm")
    hist_x_lin = enstat.histogram(bin_edges=np.linspace(1e-2, 1e0, 20001), bound_error="norm")
    hist_xr_lin = enstat.histogram(bin_edges=np.linspace(1e-2, 1e0, 20001), bound_error="norm")
    hist_xl_lin = enstat.histogram(bin_edges=np.linspace(1e-2, 1e0, 20001), bound_error="norm")

    w = int((L - L % 2) / 2)
    ensemble = eye.Ensemble([int(w - w % 2 + 1)], variance=True, periodic=True)

    if args.select is not None:
        if args.select < step.size:
            idx = np.sort(np.random.choice(np.arange(step.size), args.select, replace=False))
            file = file[idx]
            step = step[idx]

    with h5py.File(args.output, "w") as output:
        for name, hist in zip(["any", "left", "right"], [hist_x_log, hist_xl_log, hist_xr_log]):
            root = output.create_group(f"/yield_distance/{name}/log_binning")
            for key, value in hist:
                root[key] = value

        for name, hist in zip(["any", "left", "right"], [hist_x_lin, hist_xl_lin, hist_xr_lin]):
            root = output.create_group(f"/yield_distance/{name}/lin_binning")
            for key, value in hist:
                root[key] = value

        storage.create_extendible(output, "/yield_distance/any/min", dtype=np.float64)
        storage.create_extendible(output, "/yield_distance/left/min", dtype=np.float64)
        storage.create_extendible(output, "/yield_distance/right/min", dtype=np.float64)

        root = output.create_group("heightheight")
        x = ensemble.distance(0).astype(int)
        keep = x >= 0
        root["x"] = x[keep]

        root["mean"] = ensemble.result()[keep]
        root["error"] = np.sqrt(np.zeros_like(ensemble.result())[keep])

        output.flush()

        istore = 0

        for f in tqdm.tqdm(np.unique(file)):
            with h5py.File(os.path.join(basedir, paths[f])) as source:
                system = allocate_system(source)

                for s in tqdm.tqdm(np.sort(step[file == f])):
                    system.restore_quasistatic_step(source["QuasiStatic"], s)

                    xr = system.chunk.right_of_align - system.u
                    xl = system.u - system.chunk.left_of_align
                    x = np.minimum(xl, xl)

                    hist_xr_log += xr
                    hist_xl_log += xl
                    hist_x_log += x

                    hist_xr_lin += xr
                    hist_xl_lin += xl
                    hist_x_lin += x

                    storage.dset_extend1d(output, "/yield_distance/any/min", istore, np.min(x))
                    storage.dset_extend1d(output, "/yield_distance/right/min", istore, np.min(xr))
                    storage.dset_extend1d(output, "/yield_distance/left/min", istore, np.min(xl))
                    istore += 1

                    ensemble.heightheight(system.u[0, :])

                for name, hist in zip(
                    ["any", "left", "right"], [hist_x_log, hist_xl_log, hist_xr_log]
                ):
                    root = output[f"/yield_distance/{name}/log_binning"]
                    for key, value in hist:
                        root[key][...] = value

                for name, hist in zip(
                    ["any", "left", "right"], [hist_x_lin, hist_xl_lin, hist_xr_lin]
                ):
                    root = output[f"/yield_distance/{name}/lin_binning"]
                    for key, value in hist:
                        root[key][...] = value

                root = output["heightheight"]
                root["mean"][...] = ensemble.result()[keep]
                root["error"][...] = np.sqrt(ensemble.variance()[keep])

                output.flush()


def cli_structurefactor_aftersystemspanning(cli_args=None):
    """
    Extract the structure factor after a system spanning events.
    See: https://doi.org/10.1103/PhysRevLett.118.147208
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    output = file_defaults[funcname]

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("ensembleinfo", type=str, help="EnsembleInfo")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.ensembleinfo)
    tools._check_overwrite_file(args.output, args.force)
    basedir = os.path.dirname(args.ensembleinfo)

    with h5py.File(args.ensembleinfo) as info:
        assert np.all([os.path.exists(os.path.join(basedir, file)) for file in info["full"]])

        paths = info["/lookup/filepath"].asstr()[...]
        file = info["/avalanche/file"][...]
        step = info["/avalanche/step"][...]
        A = info["/avalanche/A"][...]
        N = info["/normalisation/N"][...]
        shape = info["/normalisation/shape"][...]
        L = min(shape)

        keep = A == N
        file = file[keep]
        step = step[keep]

    with h5py.File(args.output, "w") as output:
        assert L % 2 == 0
        q = np.fft.fftfreq(L)
        idx = int(L / 2)
        output["q"] = q[1:idx]
        assert np.all(q[1:idx] + np.flip(q[idx + 1 :]) == 0)
        qshape = [idx - 1 for _ in shape]
        structure = enstat.static(shape=qshape)

        output["first"] = structure.first
        output["second"] = structure.second
        output["norm"] = structure.norm

        output.flush()

        for f in tqdm.tqdm(np.unique(file)):
            with h5py.File(os.path.join(basedir, paths[f])) as source:
                for s in tqdm.tqdm(np.sort(step[file == f])):
                    u = source[f"/QuasiStatic/u/{s:d}"][...]
                    u -= u.mean()

                    if len(shape) == 2:
                        uhat = np.fft.fft2(u)
                        structure += np.real(
                            uhat[1:idx, 1:idx] * np.flip(uhat[idx + 1 :, idx + 1 :])
                        )
                    else:
                        uhat = np.fft.fft(u)
                        structure += np.real(uhat[1:idx] * np.flip(uhat[idx + 1 :]))

            output["first"][...] = structure.first
            output["second"][...] = structure.second
            output["norm"][...] = structure.norm

            output.flush()


def cli_plot(cli_args=None):
    """
    Basic plot
    """

    import GooseMPL as gplt  # noqa: F401
    import matplotlib.pyplot as plt  # noqa: F401

    plt.style.use(["goose", "goose-latex", "goose-autolayout"])

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))

    parser.add_argument("--bins", type=int, default=30, help="Number of bins.")
    parser.add_argument("-o", "--output", type=str, help="Store figure.")
    parser.add_argument("-i", "--input", type=str, help="Realisation, if input in EnsembleInfo.")
    parser.add_argument("file", type=str, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)

    with h5py.File(args.file) as file:
        if "full" in file:
            N = file["/normalisation/N"][...]
            if args.input is None:
                fname = sorted([i for i in file["full"]])[0]
                out = file["full"][fname]
                S = out["/avalanche/S"][...]
                if len(S) == 0:
                    S = []
                    for i in file["full"]:
                        S += file["full"][i]["S"][...].tolist()
                    S = np.array(S)
            else:
                fname = args.input
                out = file["full"][fname]
                S = out["S"][...]
        else:
            out = basic_output(file)
            N = Normalisation(file).N
            S = out["S"]

        A = out["A"][...]
        kick = out["kick"][...]
        u_frame = out["u_frame"][...]
        f_frame = out["f_frame"][...]
        ss = steadystate(u_frame, f_frame, kick, A, N)

    fig, axes = gplt.subplots(ncols=2)

    axes[0].plot(u_frame, f_frame, label=r"$f_\text{frame}$", marker=".")
    axes[0].plot(u_frame[A == N], f_frame[A == N], ls="none", color="r", marker="o")

    if ss is not None:
        axes[0].axvline(u_frame[ss], c="k", ls="--", lw=1)

    axes[0].set_xlabel(r"$x_\text{frame}$")
    axes[0].set_ylabel(r"$f$")
    axes[0].legend()

    axes[0].set_xlim([0, axes[0].get_xlim()[1]])
    axes[0].set_ylim([0, axes[0].get_ylim()[1]])

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")

    data = S[S > 0]
    bin_edges = gplt.histogram_bin_edges(data, bins=args.bins, mode="log")
    P, x = gplt.histogram(data, bins=bin_edges, density=True, return_edges=False)
    axes[1].plot(x, P)

    axes[1].set_xlabel(r"$S$")
    axes[1].set_ylabel(r"$P(S)$")

    if args.output is not None:
        fig.savefig(args.output)
    else:
        plt.show()

    plt.close(fig)


def cli_paraview(cli_args=None):
    """
    Write all steps to be viewed in Paraview.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))

    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, required=True, help="Appended xdmf/h5py")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    tools._check_overwrite_file(args.output, args.force)

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    tools._check_overwrite_file(f"{args.output}.h5", args.force)
    tools._check_overwrite_file(f"{args.output}.xdmf", args.force)

    with h5py.File(args.file) as file, h5py.File(f"{args.output}.h5", "w") as out, xh.TimeSeries(
        f"{args.output}.xdmf"
    ) as xdmf:
        for i in range(file["/QuasiStatic/u_frame"].size):
            u = file[f"/QuasiStatic/u/{i}"][...]

            if u.ndim == 1:
                u = u.reshape(-1, 1)

            if i == 0:
                mesh = GooseFEM.Mesh.Quad4.Regular(u.shape[0] - 1, u.shape[1] - 1)
                coor = xh.as3d(mesh.coor())
                out["coor"] = coor
                out["conn"] = mesh.conn()
                disp = np.zeros_like(coor)

            disp[:, -1] = (u - np.mean(u)).ravel()
            out[f"/disp/{i}"] = disp

            xdmf += xh.TimeStep(time=i)
            xdmf += xh.Unstructured(out["coor"], out["conn"], xh.ElementType.Quadrilateral)
            xdmf += xh.Attribute(out[f"/disp/{i}"], xh.AttributeCenter.Node, name="du")
