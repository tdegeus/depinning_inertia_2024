"""
QuasiStatic simulations, and tools to run those simulators.
"""
from __future__ import annotations

import argparse
import inspect
import os
import pathlib
import re
import textwrap
import uuid

import enstat
import FrictionQPotSpringBlock  # noqa: F401
import FrictionQPotSpringBlock as fsb
import FrictionQPotSpringBlock.Line1d as model
import GooseEYE as eye
import GooseHDF5 as g5
import h5py
import numpy as np
import prettytable
import prrng
import shelephant
import tqdm
from numpy.typing import ArrayLike

from . import storage
from . import tag
from . import tools
from ._version import version

entry_points = dict(
    cli_checkdata="QuasiStatic_CheckData",
    cli_ensembleinfo="QuasiStatic_EnsembleInfo",
    cli_generatefastload="QuasiStatic_GenerateFastLoad",
    cli_generate="QuasiStatic_Generate",
    cli_plot="QuasiStatic_Plot",
    cli_run="QuasiStatic_Run",
    cli_checkdynamics="QuasiStatic_CheckDynamics",
    cli_checkfastload="QuasiStatic_CheckFastLoad",
    cli_list_systemspanning="QuasiStatic_ReRun_ListSystemSpanning",
    cli_rerun_eventmap="QuasiStatic_ReRun_EventMap",
    cli_plotstateaftersystemspanning="QuasiStatic_PlotStateAfterSystemSpanning",
    cli_stateaftersystemspanning="QuasiStatic_StateAfterSystemSpanning",
    cli_structurefactor_aftersystemspanning="QuasiStatic_StructureAfterSystemSpanning",
)

file_defaults = dict(
    cli_ensembleinfo="QuasiStatic_EnsembleInfo.h5",
    cli_stateaftersystemspanning="QuasiStatic_StateAfterSystemSpanning.h5",
    cli_structurefactor_aftersystemspanning="QuasiStatic_StructureAfterSystemSpanning.h5",
)


def cli_checkdata(cli_args=None):
    """
    Check the version of the data, and list the required changes.
    Does not check all field of derived data (e.g. EnsembleInfo).
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

    # developer options
    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-v", "--version", action="version", version=version)

    # input file
    parser.add_argument("file", type=str, help="Simulation file (read only)")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    rename = {}
    reshape = {}
    remove = []
    add = []

    with h5py.File(args.file) as src:
        if "data_version" not in src["param"]:
            paths = g5.getdatapaths(src, fold=["/QuasiStatic/x"])
            N = src["param"]["xyield"]["initstate"].size
            is2d = "width" in src["param"]
            width = None if not is2d else src["param"]["width"][...]
            shape = [N] if not is2d else [int(N / width), width]
            shape = "[" + ", ".join([str(s) for s in shape]) + "]"
            add.append(f"/param/normalisation/shape = {shape}")

            for path in paths:
                # renaming
                for r in [
                    re.sub(r"(.*)(/)(xyield)(/)(.*)", r"\1\2potential\4\5", path),
                    re.sub(r"(.*)(/)(x)([/|_|$])(.*)", r"\1\2u\4\5", path),
                    re.sub(r"(.*)(/[[fk]_]?)(neighbours)(.*)", r"\1\2interactions\4", path),
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
                # removing nchunk
                if re.match(r"(.*)(/)(nchunk)(.*)", path):
                    remove.append(path)

            if "k2" in src["param"]:
                rename["/param/k2"] = ["/param/interactions/k1"]
                rename["/param/k4"] = ["/param/interactions/k2"]
                add.append('/param/interactions/type = "QuarticGradient"')
            elif "a1" in src["param"]:
                rename["/param/a1"] = ["/param/interactions/a1"]
                rename["/param/a2"] = ["/param/interactions/a2"]
                add.append('/param/interactions/type = "Quartic"')
            elif "alpha" in src["param"]:
                rename["/param/k_neighbours"] = ["/param/interactions/k"]
                rename["/param/alpha"] = ["/param/interactions/alpha"]
                add.append('/param/interactions/type = "LongRange"')
            else:
                rename["/param/k_neighbours"] = ["/param/interactions/k"]
                add.append('/param/interactions/type = "Laplace"')

            if "potential" in src["param"]:
                rename["/param/potential"] = ["/param/potential/type"]
            else:
                add.append('/param/potential/type = "Cuspy"')

            if "kappa" in src["param"]:
                rename["/param/kappa"] = ["/param/potential/kappa"]

            m = f"/meta/{entry_points['cli_run']}"
            if m in src:
                if "dynamics" in src[m].attrs:
                    rename[f"{m}:dynamics"] = ["/param/dynamics"]

    ret = []
    for path in rename:
        ret.append(f"Rename {path} -> {rename[path]}")
    for path in reshape:
        ret.append(f"Reshape {path} -> {reshape[path]}")
    for path in remove:
        ret.append(f"Remove {path}")
    for path in add:
        ret.append(f"Add {path}")

    print("\n".join(sorted(ret)))


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
            seed: Base seed (uint64) or uuid (str).
            N: Number of blocks (int).
            shape: Shape of the system (tuple of int).
            dt: Time step of time discretisation.
            system: Name of the system, see below (str).
        """

        self.mu = file["param"]["mu"][...]
        self.k_frame = file["param"]["k_frame"][...]
        self.eta = file["param"]["eta"][...]
        self.m = file["param"]["m"][...]
        self.N = file["param"]["normalisation"]["N"][...]
        self.shape = file["param"]["normalisation"]["shape"][...]
        self.dt = file["param"]["dt"][...]
        self.u = 1
        self.seed = None
        self.potential = str(file["/param/potentials/type"].asstr()[...])
        self.interactions = str(file["/param/interactions/type"].asstr()[...])
        self.dynamics = str(file["/param/dynamics"].asstr()[...])

        if self.interactions == "Laplace":
            self.k_interactions = file["param"]["interactions"]["k"][...]
        elif self.interactions == "Quartic":
            self.a1 = file["param"]["interactions"]["a1"][...]
            self.a2 = file["param"]["interactions"]["a2"][...]
        elif self.interactions == "QuadraticGradient":
            self.k1 = file["param"]["interactions"]["k1"][...]
            self.k2 = file["param"]["interactions"]["k2"][...]
        elif self.interactions == "LongRange":
            self.k_interactions = file["param"]["interactions"]["k"][...]
            self.alpha = file["param"]["interactions"]["alpha"][...]
        else:
            raise ValueError(f"Unknown interactions: {self.interactions:s}")

        if "realisation" in file:
            self.seed = file["realisation"]["seed"][...]

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
        if self.dynamics == "nopassing":
            extra.append("Nopassing")
        self.system = "_".join([self.system, "System", self.potential, self.interactions] + extra)

    def asdict(self):
        """
        Return relevant parameters as dictionary.
        """
        ret = dict(
            dt=self.dt,
            eta=self.eta,
            f=self.f,
            k_frame=self.k_frame,
            m=self.m,
            mu=self.mu,
            N=self.N,
            shape=self.shape,
            potential=self.potential,
            seed=self.seed,
            system=self.system,
            dynamics=self.dynamics,
            u=self.u,
        )

        if self.interactions == "Laplace":
            ret["k_interactions"] = self.k_interactions
        elif self.interactions == "Quartic":
            ret["a1"] = self.a1
            ret["a2"] = self.a2
        elif self.interactions == "QuadraticGradient":
            ret["k1"] = self.k1
            ret["k2"] = self.k2
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
    assert np.all(file_yield["initstate"][...].ravel() == np.arange(file_yield["initstate"].size))
    assert np.all(file_yield["initseq"][...].ravel() == np.zeros(file_yield["initseq"].size))

    ret = {
        "shape": file_yield["initstate"].shape,
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
    """ """

    def __init__(self, file: h5py.File):
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
        """
        Initialise system.
        """

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
        """
        Initialise system.
        """

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
        """
        Initialise system.
        """

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
        """
        Initialise system.
        """

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
        """
        Initialise system.
        """

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
        """
        Initialise system.
        """

        SystemExtra.__init__(self, file)

        model.System_Cuspy_QuarticGradient.__init__(
            self,
            m=file["param"]["m"][...],
            eta=file["param"]["eta"][...],
            mu=file["param"]["mu"][...],
            k1=file["param"]["interactions"]["k1"][...],
            k2=file["param"]["interactions"]["k2"][...],
            k_frame=file["param"]["k_frame"][...],
            dt=file["param"]["dt"][...],
            **_common_param(file),
        )


class Line1d_System_Cuspy_Quartic(model.System_Cuspy_Quartic, SystemExtra):
    def __init__(self, file: h5py.File):
        """
        Initialise system.
        """

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
        """
        Initialise system.
        """

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
        """
        Initialise system.
        """

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
    Allocate system.
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


def _compare_versions(ver, cmpver):
    if tag.greater_equal(cmpver, "6.0"):
        if tag.greater_equal(ver, cmpver):
            return True
    else:
        return tag.equal(ver, cmpver)

    return False


def create_check_meta(
    file: h5py.File = None,
    path: str = None,
    dev: bool = False,
    **kwargs,
) -> h5py.Group:
    """
    Create or read/check metadata. This function asserts that:

    -   There are no uncommitted changes.
    -   There are no version changes.

    It create metadata as attributes to a group ``path`` as follows::

        "uuid": A unique identifier that can be used to distinguish simulations if needed.
        "version": The current version of this code (see below).
        "dependencies": The current version of all relevant dependencies (see below).
        "compiler": Compiler information.

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
        assert dev or _compare_versions(version, meta.attrs["version"])
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
    nopassing: bool = False,
):
    """
    Generate a simulation file.

    :param file: HDF5 file opened for writing.
    :param N: System size.
    :param seed: Base seed.
    :param eta: Damping coefficient.
    :param dt: Time step.
    :param k_frame: Frame stiffness. Default: ``1 / L**2``.
    :param potential: Select potential.
    :param distribution: Distribution of potentials.
    :param interactions: Select interactions.
    :param nopassing: Run overdamped dynamics with no passing rule.
    """

    N = np.prod(shape)
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
        file["/param/interactions/k1"] = interactions["k1"]
        file["/param/interactions/k2"] = interactions["k2"]
    elif interactions["type"] == "LongRange":
        file["/param/interactions/k"] = interactions["k"]
        file["/param/interactions/alpha"] = interactions["alpha"]
    else:
        raise ValueError(f"Unknown interactions: {interactions['type']}")

    file["/realisation/seed"] = seed
    file["/param/m"] = 1.0
    file["/param/eta"] = eta
    file["/param/mu"] = potential["mu"]
    file["/param/k_frame"] = 1.0 / L**2
    file["/param/dt"] = dt
    file["/param/dynamics"] = "normal"
    file["/param/potentials/type"] = potential["type"]
    file["/param/potentials/initstate"] = np.arange(N).reshape(shape).astype(np.int64)
    file["/param/potentials/initseq"] = np.zeros(shape, dtype=np.int64)
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

    if nopassing:
        file["/param/dynamics"][...] = "nopassing"

    file["/param/normalisation/N"] = N
    file["/param/normalisation/shape"] = shape
    file["/param/normalisation/u"] = 1
    file["/param/data_version"] = "1.0"


def _generate_cli_options(parser):
    parser.add_argument("-n", "--nsim", type=int, default=1, help="#simulations")
    parser.add_argument("-s", "--start", type=int, default=0, help="Start simulation")
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")

    parser.add_argument("--size", type=int, help="1d system")
    parser.add_argument("--shape", nargs=2, type=int, help="2d system")

    parser.add_argument("--dt", type=float, help="Time-step")
    parser.add_argument("--eta", type=float, help="Damping coefficient")
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
    parser.add_argument("--quarticgradient", nargs=2, type=float, help="Quartic gradient: k1, k2")
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
            "k1": args.quarticgradient[0],
            "k2": args.quarticgradient[1],
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
    parser.add_argument("--nopassing", action="store_true", help="Overdamped dynamics")

    args = tools._parse(parser, cli_args)
    assert args.nopassing or args.eta is not None

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    opts = _generate_parse(args)
    n = args.size if args.shape is None else np.prod(args.shape)

    files = []

    for i in range(args.start, args.start + args.nsim):
        files += [f"id={i:04d}.h5"]
        seed = i * n

        with h5py.File(outdir / files[-1], "w") as file:
            generate(
                file=file,
                seed=seed,
                nopassing=args.nopassing,
                **opts,
            )

    executable = entry_points["cli_run"]

    opts = ["--fastload"]
    if args.nopassing:
        opts += ["--nopassing"]
    opts = " ".join(opts)
    if len(opts) > 0:
        opts = " " + opts

    commands = [f"{executable}{opts} --nstep {args.nstep:d} {file}" for file in files]
    shelephant.yaml.dump(outdir / "commands.yaml", commands, force=True)


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

    # development options
    parser.add_argument("--check", type=int, help="Rerun step to check old run / new version")
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("--fastload", action="store_true", help="Append fastload file")
    parser.add_argument("-v", "--version", action="version", version=version)

    # different loading
    parser.add_argument(
        "--fixed-step",
        action="store_true",
        help="Use a fixed loading-step instead for the event-driven protocol",
    )

    # simulation parameters
    parser.add_argument("-n", "--nstep", type=int, default=5000, help="Total #load-steps to run")
    parser.add_argument("file", type=str, help="Input/output file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    basename = os.path.basename(args.file)

    with h5py.File(args.file, "a") as file:
        system = allocate_system(file)
        meta = dict(dynamics="normal", loading="event-driven")
        if isinstance(system, Line1d_System_Cuspy_Laplace_Nopassing):
            meta["dynamics"] = "nopassing"

        if args.fixed_step:
            meta["loading"] = "fixed-step"
            dx_particle = 1e-3 * system.normalisation.u
            dx_frame = (
                dx_particle
                * (system.normalisation.k_frame + system.normalisation.mu)
                / system.normalisation.k_frame
            )

        create_check_meta(file, f"/meta/{progname}", dev=args.develop, **meta)

        if "QuasiStatic" not in file:
            ret = system.minimise()
            assert ret == 0
            system.t = 0.0

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

        for step in pbar:
            if args.fixed_step:
                kick = True
                system.u_frame += dx_frame
            else:
                kick = not kick
                system.eventDrivenStep(du, kick)

            if kick:
                inc_n = system.inc

                ret = system.minimise()
                assert ret == 0

                niter = system.inc - inc_n
                pbar.set_description(f"{basename}: step = {step:8d}, niter = {niter:8d}")
                pbar.refresh()

            if args.check is not None:
                assert root["inc"][step] == system.inc
                assert root["kick"][step] == kick
                assert np.isclose(root["u_frame"][step], system.u_frame)
                assert np.allclose(root["u"][str(step)][...], system.u)
            else:
                storage.dset_extend1d(root, "inc", step, system.inc)
                storage.dset_extend1d(root, "u_frame", step, system.u_frame)
                storage.dset_extend1d(root, "kick", step, kick)
                root["u"][str(step)] = system.u
                file.flush()

                if args.fastload:
                    with h5py.File(filename2fastload(args.file), "a") as fload:
                        if f"/QuasiStatic/{step:d}" not in fload:
                            start = np.copy(system.chunk.start)
                            fload[f"/QuasiStatic/{step:d}/state"] = system.chunk.state_at(start)
                            fload[f"/QuasiStatic/{step:d}/index"] = start
                            fload[f"/QuasiStatic/{step:d}/value"] = system.chunk.data[..., 0]
                            fload.flush()


def cli_checkdynamics(cli_args=None):
    """
    Check the detailed dynamics of a quasi-static step.
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
                assert np.allclose(output[f"/iter/{i:d}/f_inter"][...], system.f_interactions)
                assert np.allclose(output[f"/iter/{i:d}/f"][...], system.f)
                assert np.allclose(output[f"/iter/{i:d}/u"][...], system.u)
                assert np.allclose(output[f"/iter/{i:d}/v"][...], system.v)
                assert np.allclose(output[f"/iter/{i:d}/y_i"][...], system.chunk.left_of_align)
                assert np.allclose(output[f"/iter/{i:d}/y_i+1"][...], system.chunk.right_of_align)
                assert np.allclose(output[f"/iter/{i:d}/index"][...], system.chunk.index_at_align)
            except AssertionError:
                passed = (
                    np.isclose(output[f"/iter/{i:d}/f_inter"][...], system.f_interactions)
                    & np.isclose(output[f"/iter/{i:d}/f"][...], system.f)
                    & np.isclose(output[f"/iter/{i:d}/u"][...], system.u)
                    & np.isclose(output[f"/iter/{i:d}/v"][...], system.v)
                    & np.isclose(output[f"/iter/{i:d}/y_i"][...], system.chunk.left_of_align)
                    & np.isclose(output[f"/iter/{i:d}/y_i+1"][...], system.chunk.right_of_align)
                    & np.isclose(output[f"/iter/{i:d}/index"][...], system.chunk.index_at_align)
                )

                passed[0] = False

                idx = system.chunk.chunk_index_at_align.ravel()
                allp = np.arange(idx.size)
                data = system.chunk.data.reshape(-1, system.chunk.chunk_size)

                u_n = system.u
                v_n = system.v
                f_n = system.f
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
        dynamics=[],
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
                    norm = Normalisation(file).asdict()
                    seed = norm.pop("seed")
                else:
                    test = Normalisation(file).asdict()
                    seed = test.pop("seed")
                    for key in norm:
                        if key in ["potential", "system", "dynamics"]:
                            assert str(norm[key]) == str(test[key])
                        elif key == "shape":
                            assert list(norm[key]) == list(test[key])
                        else:
                            assert np.isclose(norm[key], test[key])

                out = basic_output(file)

                if i == 0:
                    fields_full = [key for key in out if key not in ["steadystate"]]
                    combine_load = {key: [] for key in fields_full}
                    combine_kick = {key: [] for key in fields_full}
                    file_load = []
                    file_kick = []

                info["seed"].append(seed)

                meta = file[f"/meta/{entry_points['cli_run']}"]
                for key in ["uuid", "version", "dynamics"]:
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
        tools.h5py_save_unique(info["dynamics"], output, "/lookup/dynamics", asstr=True)
        tools.h5py_save_unique(info["version"], output, "/lookup/version", asstr=True)
        output["files"] = output["/lookup/filepath"]


def cli_list_systemspanning(cli_args=None):
    """
    Write list of jobs to run to get an event map.
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
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("--exec", type=str, required=True, help="Executable")
    parser.add_argument("--options", type=str, default="", help="Options to pass")
    parser.add_argument("info", type=str, help="EnsembleInfo")
    parser.add_argument("output", type=str, help="Output file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.info)

    with h5py.File(args.info) as file:
        step = file["/avalanche/step"][...]
        A = file["/avalanche/A"][...]
        N = file["/normalisation/N"][...]
        fname = file["/lookup/filepath"].asstr()[...][file["/avalanche/file"][...]]

    keep = A == N
    step = step[keep]
    fname = list(fname[keep])

    root = pathlib.Path(os.path.relpath(args.info, pathlib.Path(args.output).parent)).parent
    rname = [str(root / f) for f in fname]
    fname = [os.path.normpath(f).split(".h5")[0] for f in fname]

    ret = []

    for i in range(len(step)):
        ret.append(
            f"{args.exec} {args.options} --step {step[i]:d} {rname[i]} -o {fname[i]}_step={step[i]:d}.h5"
        )

    shelephant.yaml.dump(args.output, ret, force=True)


def cli_rerun_eventmap(cli_args=None):
    """
    Write list of jobs to run to get an event map.
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
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("info", type=str, help="EnsembleInfo")
    parser.add_argument("output", type=str, help="Output file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.info)

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

    root = pathlib.Path(os.path.relpath(args.info, pathlib.Path(args.output).parent)).parent
    rname = [str(root / f) for f in fname]
    fname = [os.path.normpath(f).split(".h5")[0] for f in fname]
    opts = ["-s"]
    if is2d:
        opts += ["-u"]
    opts = " ".join(opts)

    c = [
        f"EventMap_run {opts} --step {s:d} --smax {smax:d} {r} -o {f}_step={s:d}.h5"
        for s, smax, r, f in zip(step, S, rname, fname)
    ]

    shelephant.yaml.dump(args.output, c, force=True)


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
            # may not be present in old files
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
    progname = entry_points[funcname]

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

    output = filename2fastload(args.file)
    if args.append:
        assert os.path.isfile(output)
    else:
        tools._check_overwrite_file(output, args.force)

    with h5py.File(args.file) as file, h5py.File(output, "r+" if args.append else "w") as output:
        if not args.append:
            create_check_meta(output, f"/meta/{progname}", dev=args.develop)

        system = allocate_system(file)
        root = file["QuasiStatic"]
        last_start = None
        last_step = None

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

            if last_start is not None:
                if np.all(np.equal(last_start, system.chunk.start)):
                    output[f"/QuasiStatic/{step:d}"] = output[f"/QuasiStatic/{last_step:d}"]
                    continue

            start = np.copy(system.chunk.start)
            output[f"/QuasiStatic/{step:d}/state"] = system.chunk.state_at(start)
            output[f"/QuasiStatic/{step:d}/index"] = start
            output[f"/QuasiStatic/{step:d}/value"] = system.chunk.data[..., 0]
            output.flush()
            last_start = np.copy(start)
            last_step = step


def cli_plotstateaftersystemspanning(cli_args=None):
    """
    Plot state after system-spanning events.
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


def cli_stateaftersystemspanning(cli_args=None):
    """
    Extract the distribution of P(x), with x the distance to yielding.
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
        is2d = shape.size == 2

        keep = A == N
        file = file[keep]
        step = step[keep]

    hist_x_log = enstat.histogram(bin_edges=np.logspace(-4, 1, 20001), bound_error="norm")
    hist_xr_log = enstat.histogram(bin_edges=np.logspace(-4, 1, 20001), bound_error="norm")
    hist_xl_log = enstat.histogram(bin_edges=np.logspace(-4, 1, 20001), bound_error="norm")
    hist_x_lin = enstat.histogram(bin_edges=np.linspace(1e-2, 1e0, 20001), bound_error="norm")
    hist_xr_lin = enstat.histogram(bin_edges=np.linspace(1e-2, 1e0, 20001), bound_error="norm")
    hist_xl_lin = enstat.histogram(bin_edges=np.linspace(1e-2, 1e0, 20001), bound_error="norm")

    roi = int((L - L % 2) / 2)
    roi = int(roi - roi % 2 + 1)
    w = int((roi - roi % 2) / 2 + 1)
    reshape = [w for _ in shape]
    roi = [roi for _ in shape]

    ensemble = eye.Ensemble(roi, variance=True, periodic=True)

    if args.select is not None:
        if args.select < step.size:
            idx = np.sort(np.random.choice(np.arange(step.size), args.select, replace=False))
            file = file[idx]
            step = step[idx]

    with h5py.File(args.output, "w") as output:
        for name, hist in zip(["any", "left", "right"], [hist_x_log, hist_xl_log, hist_xr_log]):
            root = output.create_group(f"/yield_distance/{name}/log")
            for key, value in hist:
                root[key] = value

        for name, hist in zip(["any", "left", "right"], [hist_x_lin, hist_xl_lin, hist_xr_lin]):
            root = output.create_group(f"/yield_distance/{name}/lin")
            for key, value in hist:
                root[key] = value

        root = output.create_group("heightheight")
        if is2d:
            Ax = ensemble.distance(0).astype(int)
            Ay = ensemble.distance(1).astype(int)
            keep = np.logical_and(Ax >= 0, Ay >= 0)
            root["Ax"] = Ax[keep].reshape(reshape)
            root["Ay"] = Ay[keep].reshape(reshape)
        else:
            A = ensemble.distance(0).astype(int)
            keep = A >= 0
            root["A"] = A[keep]

        root["R"] = ensemble.result()[keep].reshape(reshape)
        root["error"] = np.sqrt(np.zeros_like(ensemble.result())[keep]).reshape(reshape)

        output.flush()

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

                    ensemble.heightheight(system.u)

                for name, hist in zip(
                    ["any", "left", "right"], [hist_x_log, hist_xl_log, hist_xr_log]
                ):
                    root = output[f"/yield_distance/{name}/log"]
                    for key, value in hist:
                        root[key][...] = value

                for name, hist in zip(
                    ["any", "left", "right"], [hist_x_lin, hist_xl_lin, hist_xr_lin]
                ):
                    root = output[f"/yield_distance/{name}/lin"]
                    for key, value in hist:
                        root[key][...] = value

                root = output["heightheight"]
                root["R"][...] = ensemble.result()[keep].reshape(reshape)
                root["error"][...] = np.sqrt(ensemble.variance()[keep]).reshape(reshape)

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
                system = allocate_system(source)

                for s in tqdm.tqdm(np.sort(step[file == f])):
                    u = source["QuasiStatic"]["u"][str(s)][...][system.organisation]
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
    parser.add_argument("-m", "--marker", type=str, help="Marker.")
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
        f_potential = out["f_potential"][...]
        ss = steadystate(u_frame, f_frame, kick, A, N)

    opts = {}
    if args.marker is not None:
        opts["marker"] = args.marker

    fig, axes = gplt.subplots(ncols=2)

    axes[0].plot(u_frame, f_frame, label=r"$f_\text{frame}$", **opts)
    axes[0].plot(u_frame, f_potential, label=r"$f_\text{potential}$", **opts)
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

    if args.output is not None:
        fig.savefig(args.output)
    else:
        plt.show()

    plt.close(fig)
