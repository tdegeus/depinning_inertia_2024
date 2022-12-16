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

import FrictionQPotSpringBlock  # noqa: F401
import FrictionQPotSpringBlock.Line1d as model
import GooseEYE as eye
import h5py
import numpy as np
import prrng
import shelephant
import tqdm
from numpy.typing import ArrayLike

from . import storage
from . import tag
from . import tools
from ._version import version

entry_points = dict(
    cli_ensembleinfo="QuasiStatic_EnsembleInfo",
    cli_generatefastload="QuasiStatic_GenerateFastLoad",
    cli_generate="QuasiStatic_Generate",
    cli_plot="QuasiStatic_Plot",
    cli_run="QuasiStatic_Run",
    cli_stateaftersystemspanning="QuasiStatic_StateAfterSystemSpanning",
)

file_defaults = dict(
    cli_ensembleinfo="QuasiStatic_EnsembleInfo.h5",
    cli_stateaftersystemspanning="QuasiStatic_StateAfterSystemSpanning.h5",
)


def dependencies(system: model.System) -> list[str]:
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
            k_frame: Stiffness of the load-frame (float).
            k_neighbours: Stiffness of the neighbour interactions (float).
            eta: Damping (float).
            m: Mass (float).
            seed: Base seed (uint64) or uuid (str).
            N: Number of blocks (int).
            dt: Time step of time discretisation.
            system: Name of the system class (str).
            potential: Name of the potential (str).
            kappa: Weakening 'stiffness', if SemiSmooth potential is used (float).
            alpha: Interaction range, if long-range interactions are used (float).
        """

        self.alpha = None
        self.width = None
        self.kappa = None
        self.mu = file["param"]["mu"][...]
        self.k_frame = file["param"]["k_frame"][...]
        self.k_neighbours = file["param"]["k_neighbours"][...]
        self.eta = file["param"]["eta"][...]
        self.m = file["param"]["m"][...]
        self.N = file["param"]["normalisation"]["N"][...]
        self.dt = file["param"]["dt"][...]
        self.x = 1
        self.seed = None

        if "realisation" in file:
            self.seed = file["realisation"]["seed"][...]

        if "x" in file["param"]["normalisation"]:
            self.x = file["param"]["normalisation"]["x"][...]

        if "potential" not in file["param"]:
            self.potential = "Cusp"
        else:
            self.potential = file["/param/potential/name"].asstr()[...]

        if self.potential == "Cusp":
            self.system = "System"
            self.f = self.mu * self.x
        elif self.potential == "SemiSmooth":
            self.system = "SystemSemiSmooth"
            self.kappa = file["param"]["kappa"][...]
            self.f = self.mu * self.x * (1 - self.mu / (self.mu + self.kappa))
        elif self.potential == "Smooth":
            self.system = "SystemSmooth"
            self.f = self.mu * self.x / np.pi
        else:
            raise ValueError(f"Unknown potential: {self.potential:s}")

        if "alpha" in file["param"]:
            self.alpha = file["param"]["alpha"][...]
            assert self.system == "System"
            self.system = "SystemLongRange"

        if "width" in file["param"]:
            self.width = file["param"]["width"][...]
            assert self.system == "System"
            self.system = "System2d"

        if "thermal" in file["param"]:
            assert self.system == "System"
            self.system = "SystemThermalRandomForceNormal"

    def asdict(self):
        """
        Return relevant parameters as dictionary.
        """
        ret = dict(
            mu=self.mu,
            k_frame=self.k_frame,
            k_neighbours=self.k_neighbours,
            eta=self.eta,
            m=self.m,
            seed=self.seed,
            N=self.N,
            dt=self.dt,
            x=self.x,
            potential=self.potential,
            system=self.system,
        )

        if self.kappa is not None:
            ret["kappa"] = self.kappa

        return ret


class DataMap:
    """
    File interaction needed to allocate a System (or one of its subclasses).
    """

    def __init__(self, file: h5py.File, nchunk: int = None, chunk_use_max: bool = True):
        """
        Initialise system.

        :param nchunk: Overwrite the default chuck.
        :param chunk_use_max: If True, use the maximum between the option and the value in the file.
        """

        file_yield = file["param"]["xyield"]
        initstate = file["realisation"]["seed"][...] + file_yield["initstate"][...]
        initseq = file_yield["initseq"][...]
        xoffset = file_yield["xoffset"][...]

        if nchunk is None:
            nchunk = file_yield["nchunk"][...]
        elif chunk_use_max:
            nchunk = max(file_yield["nchunk"][...], nchunk)

        assert nchunk > 30
        align = prrng.alignment(margin=30, min_margin=6, buffer=5, strict=False)

        if "weibull" in file_yield:
            distribution = prrng.distribution.weibull
            parameters = [
                file_yield["weibull"]["k"][...],
                2 * file_yield["weibull"]["mean"][...],
                file_yield["weibull"]["offset"][...],
            ]
        elif "delta" in file_yield:
            distribution = prrng.distribution.delta
            parameters = [2 * file_yield["delta"]["mean"][...], 0]
        elif "random" in file_yield:
            distribution = prrng.distribution.random
            parameters = [
                2 * file_yield["random"]["mean"][...],
                file_yield["random"]["offset"][...],
            ]
        else:
            raise OSError("Distribution not supported")

        self.chunk = prrng.pcg32_tensor_cumsum_1_1(
            shape=[nchunk],
            initstate=initstate,
            initseq=initseq,
            distribution=distribution,
            parameters=parameters,
            align=align,
        )

        self.chunk += xoffset
        self.normalisation = Normalisation(file)

    def chunk_goto(
        self,
        x: ArrayLike,
        fastload: tuple(str, str) = None,
    ):
        """
        Update the yield positions to be able to accommodate a target position ``x``.
        Note: the position is not updated!

        :param x: Target position.
        :param fastload: If available ``(filename, groupname)`` of the closest fastload info.
        """

        if self.chunk.contains(x):
            return self.chunk.align(x)

        front = self.chunk.data[:, 0]
        back = self.chunk.data[:, -1]

        if np.all(x < 2 * back - front):
            return self.chunk.align(x)

        if fastload is None:
            return self.chunk.align(x)

        if len(fastload) != 2:
            return self.chunk.align(x)

        if fastload[0] is None:
            return self.chunk.align(x)

        if not os.path.exists(fastload[0]):
            return self.chunk.align(x)

        with h5py.File(fastload[0]) as loadfile:
            if fastload[1] in loadfile:
                root = loadfile[fastload[1]]
                self.chunk.restore(root["state"][...], root["value"][...], root["index"][...])

        return self.chunk.align(x)

    def restore_quasistatic_step(
        self,
        root: h5py.Group,
        step: int,
        fastload: bool = True,
    ):
        """
        Quench and restore an a quasi-static step for the relevant root.
        The ``root`` group should contain::

            root["x"][str(step)]   # Positions
            root["inc"][step]      # Increment (-> time)
            root["x_frame"][step]  # Loading frame position

        :param root: HDF5 archive opened in the right root (read-only).
        :param step: Step number.
        :param fastload: Use fastload file (if detected), see :py:func:`cli_generatefastload`.
        """

        x = root["x"][str(step)][...]

        if type(fastload) == tuple or type(fastload) == list:
            pass
        elif fastload:
            fastload = (filename2fastload(root.file.filename), f"/QuasiStatic/{step:d}")
        else:
            fastload = None

        self.chunk_goto(x, fastload)
        self.quench()
        self.inc = root["inc"][step]
        self.x_frame = root["x_frame"][step]
        self.x = x


class System(model.System, DataMap):
    def __init__(self, file: h5py.File, **kwargs):
        """
        Initialise system.
        """

        DataMap.__init__(self, file, **kwargs)

        model.System.__init__(
            self,
            m=file["param"]["m"][...],
            eta=file["param"]["eta"][...],
            mu=file["param"]["mu"][...],
            k_neighbours=file["param"]["k_neighbours"][...],
            k_frame=file["param"]["k_frame"][...],
            dt=file["param"]["dt"][...],
            chunk=self.chunk,
        )


class SystemSemiSmooth(model.SystemSemiSmooth, DataMap):
    def __init__(self, file: h5py.File, **kwargs):
        """
        Initialise system.
        """

        DataMap.__init__(self, file, **kwargs)

        model.SystemSemiSmooth.__init__(
            self,
            m=file["param"]["m"][...],
            eta=file["param"]["eta"][...],
            mu=file["param"]["mu"][...],
            kappa=file["param"]["kappa"][...],
            k_neighbours=file["param"]["k_neighbours"][...],
            k_frame=file["param"]["k_frame"][...],
            dt=file["param"]["dt"][...],
            chunk=self.chunk,
        )


class SystemSmooth(model.SystemSmooth, DataMap):
    def __init__(self, file: h5py.File, **kwargs):
        """
        Initialise system.
        """

        DataMap.__init__(self, file, **kwargs)

        model.SystemSmooth.__init__(
            self,
            m=file["param"]["m"][...],
            eta=file["param"]["eta"][...],
            mu=file["param"]["mu"][...],
            k_neighbours=file["param"]["k_neighbours"][...],
            k_frame=file["param"]["k_frame"][...],
            dt=file["param"]["dt"][...],
            chunk=self.chunk,
        )


class SystemLongRange(model.SystemLongRange, DataMap):
    def __init__(self, file: h5py.File, **kwargs):
        """
        Initialise system.
        """

        DataMap.__init__(self, file, **kwargs)

        model.SystemLongRange.__init__(
            self,
            m=file["param"]["m"][...],
            eta=file["param"]["eta"][...],
            mu=file["param"]["mu"][...],
            k_neighbours=file["param"]["k_neighbours"][...],
            alpha=file["param"]["alpha"][...],
            k_frame=file["param"]["k_frame"][...],
            dt=file["param"]["dt"][...],
            chunk=self.chunk,
        )


class System2d(model.System2d, DataMap):
    def __init__(self, file: h5py.File, **kwargs):
        """
        Initialise system.
        """

        DataMap.__init__(self, file, **kwargs)

        model.System2d.__init__(
            self,
            m=file["param"]["m"][...],
            eta=file["param"]["eta"][...],
            mu=file["param"]["mu"][...],
            k_neighbours=file["param"]["k_neighbours"][...],
            k_frame=file["param"]["k_frame"][...],
            dt=file["param"]["dt"][...],
            chunk=self.chunk,
            width=file["param"]["width"][...],
        )


class SystemThermalRandomForceNormal(model.SystemThermalRandomForceNormal, DataMap):
    def __init__(self, file: h5py.File, **kwargs):
        """
        Initialise system.
        """

        DataMap.__init__(self, file, **kwargs)

        model.SystemThermalRandomForceNormal.__init__(
            self,
            m=file["param"]["m"][...],
            eta=file["param"]["eta"][...],
            mu=file["param"]["mu"][...],
            k_neighbours=file["param"]["k_neighbours"][...],
            k_frame=file["param"]["k_frame"][...],
            dt=file["param"]["dt"][...],
            chunk=self.chunk,
            mean=file["param"]["thermal"]["mean"][...],
            stddev=file["param"]["thermal"]["stddev"][...],
            seed=file["param"]["thermal"]["seed"][...],
            dinc_init=file["param"]["thermal"]["dinc_init"][...],
            dinc=file["param"]["thermal"]["dinc"][...],
        )


def allocate_system(file: h5py.File, **kwargs):
    """
    Allocate system.
    """

    norm = Normalisation(file)

    if norm.system == "System":
        return System(file, **kwargs)

    if norm.system == "SystemSemiSmooth":
        return SystemSemiSmooth(file, **kwargs)

    if norm.system == "SystemSmooth":
        return SystemSmooth(file, **kwargs)

    if norm.system == "SystemThermalRandomForceNormal":
        return SystemThermalRandomForceNormal(file, **kwargs)

    if norm.system == "SystemLongRange":
        return SystemLongRange(file, **kwargs)

    if norm.system == "System2d":
        return System2d(file, **kwargs)


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

    deps = sorted(list(set(list(model.version_dependencies()) + ["prrng=" + prrng.version()])))

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
    N: int,
    seed: int,
    eta: float = None,
    dt: float = None,
    distribution: str = "weibull",
):
    """
    Generate a simulation file.

    :param file: HDF5 file opened for writing.
    :param N: System size.
    :param seed: Base seed.
    :param eta: Damping coefficient.
    :param dt: Time step.
    """

    N = int(N)

    if eta is None and dt is None:
        eta = 1
        dt = 1
    elif dt is None:
        known_eta = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
        known_dt = np.array([5e-3, 5e-3, 5e-2, 5e-2, 5e-3, 5e-4])
        dt = np.interp(eta, known_eta, known_dt)
    else:
        assert eta is not None

    file["/realisation/seed"] = seed
    file["/param/m"] = 1.0
    file["/param/eta"] = eta
    file["/param/mu"] = 1.0
    file["/param/k_neighbours"] = 1.0
    file["/param/k_frame"] = 1.0 / N
    file["/param/dt"] = dt
    file["/param/xyield/initstate"] = np.arange(N).astype(np.int64)
    file["/param/xyield/initseq"] = np.zeros(N, dtype=np.int64)
    file["/param/xyield/nchunk"] = min(5000, max(1000, int(2 * N)))
    file["/param/xyield/xoffset"] = -100
    file["/param/xyield/dx"] = 1e-3

    if distribution.lower() == "weibull":
        file["/param/xyield/weibull/offset"] = 1e-5
        file["/param/xyield/weibull/mean"] = 1
        file["/param/xyield/weibull/k"] = 2
    elif distribution.lower() == "delta":
        file["/param/xyield/delta/mean"] = 1
    elif distribution.lower() == "random":
        file["/param/xyield/random/offset"] = 1e-5
        file["/param/xyield/random/mean"] = 1
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    file["/param/potential/name"] = "Cusp"
    file["/param/normalisation/N"] = N
    file["/param/normalisation/x"] = 1


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

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("--dt", type=float, help="Time-step")
    parser.add_argument("--eta", type=float, help="Damping coefficient")
    parser.add_argument("--fastload", action="store_true", help="Store fastload file")
    parser.add_argument("--nopassing", action="store_true", help="Job scripts for overdamped run")
    parser.add_argument("--nstep", type=int, default=20000, help="#load-steps to run")
    parser.add_argument("--kframe", type=float, default=1, help="k_frame = X / N, with X this arg.")
    parser.add_argument("--distribution", type=str, default="Weibull", help="Distribution type")
    parser.add_argument("--smooth", action="store_true", help="Smooth potential")
    parser.add_argument("--kappa", type=float, help="SemiSmooth potential with slope kappa")
    parser.add_argument("--alpha", type=float, help="Long range interaction with this exponent")
    parser.add_argument("--width", type=int, help="Run in 2d with this width")

    parser.add_argument("-n", "--nsim", type=int, default=1, help="#simulations")
    parser.add_argument("-N", "--size", type=int, default=5000, help="#particles")
    parser.add_argument("-s", "--start", type=int, default=0, help="Start simulation")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("outdir", type=str, help="Output directory")

    args = tools._parse(parser, cli_args)
    assert args.nopassing or args.eta is not None

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = []

    for i in range(args.start, args.start + args.nsim):

        files += [f"id={i:04d}.h5"]
        seed = i * args.size

        with h5py.File(outdir / files[-1], "w") as file:
            generate(
                file=file,
                N=args.size,
                seed=seed,
                eta=args.eta,
                dt=args.dt,
                distribution=args.distribution,
            )

            file["/param/k_frame"][...] = float(args.kframe / args.size)

            if args.smooth:
                file["/param/potential/name"][...] = "Smooth"

            if args.kappa:
                file["/param/potential/name"][...] = "SemiSmooth"
                file["/param/kappa"] = args.kappa

            if args.alpha:
                file["/param/alpha"] = args.alpha

            if args.width:
                file["/param/width"] = args.width

    executable = entry_points["cli_run"]

    opts = []
    if args.nopassing:
        opts += ["--nopassing"]
    if args.fastload:
        opts += ["--fastload"]
    opts = " ".join(opts)
    if len(opts) > 0:
        opts = " " + opts

    commands = [f"{executable}{opts} --nstep {args.nstep:d} {file}" for file in files]
    shelephant.yaml.dump(outdir / "commands.yaml", commands)


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

    # different dynamics
    parser.add_argument(
        "--nopassing",
        action="store_true",
        help="Use no-passing rule (instead of inertial dynamics: assume m = 0)",
    )

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

        if args.nopassing:
            minimise = system.minimise_nopassing
            meta["dynamics"] = "nopassing"
        else:
            minimise = system.minimise

        if args.fixed_step:
            meta["loading"] = "fixed-step"
            dx_particle = 1e-3 * system.normalisation.x
            dx_frame = (
                dx_particle
                * (system.normalisation.k_frame + system.normalisation.mu)
                / system.normalisation.k_frame
            )

        create_check_meta(file, f"/meta/{progname}", dev=args.develop, **meta)

        if "QuasiStatic" not in file:
            ret = minimise()
            assert ret == 0
            system.t = 0.0

            file.create_group("/QuasiStatic/x").create_dataset("0", data=system.x)
            root = file["/QuasiStatic"]
            storage.create_extendible(root, "inc", np.uint64, desc="'Time' (increment number).")
            storage.create_extendible(root, "x_frame", np.float64, desc="Position of load frame.")
            storage.create_extendible(root, "kick", bool, desc="Kick used.")
            storage.dset_extend1d(root, "inc", 0, system.inc)
            storage.dset_extend1d(root, "x_frame", 0, system.x_frame)
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
        dx = file["/param/xyield/dx"][...]
        system.restore_quasistatic_step(root, step - 1)

        desc = f"{basename}: step = {step:8d}, niter = {'-':8s}"
        pbar = tqdm.tqdm(range(step, step + args.nstep), desc=desc)

        for step in pbar:

            if args.fixed_step:
                kick = True
                system.x_frame += dx_frame
            else:
                kick = not kick
                system.eventDrivenStep(dx, kick)

            if kick:
                inc_n = system.inc

                ret = minimise()
                assert ret == 0

                niter = system.inc - inc_n
                pbar.set_description(f"{basename}: step = {step:8d}, niter = {niter:8d}")
                pbar.refresh()

            if args.check is not None:
                assert root["inc"][step] == system.inc
                assert root["kick"][step] == kick
                assert np.isclose(root["x_frame"][step], system.x_frame)
                assert np.allclose(root["x"][str(step)][...], system.x)
            else:
                storage.dset_extend1d(root, "inc", step, system.inc)
                storage.dset_extend1d(root, "x_frame", step, system.x_frame)
                storage.dset_extend1d(root, "kick", step, kick)
                root["x"][str(step)] = system.x
                file.flush()

                if args.fastload:
                    with h5py.File(filename2fastload(args.file), "a") as fload:
                        if f"/QuasiStatic/{step:d}" not in fload:
                            i = system.chunk.start
                            fload[f"/QuasiStatic/{step:d}/state"] = system.chunk.state_at(i)
                            fload[f"/QuasiStatic/{step:d}/index"] = i
                            fload[f"/QuasiStatic/{step:d}/value"] = system.chunk.data[:, 0]
                            fload.flush()


def steadystate(
    x_frame: ArrayLike, f_frame: ArrayLike, kick: ArrayLike, A: ArrayLike, N: int, **kwargs
) -> int:
    """
    Estimate the first step of the steady-state. Constraints:

    -   Start with elastic loading.
    -   Sufficiently low tangent modulus.
    -   All blocks yielded at least once.

    .. note::

        Keywords arguments that are not explicitly listed are ignored.

    :param x_frame: Position of the load frame [nstep].
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
    tangent[1:] = (f_frame[1:] - f_frame[0]) / (x_frame[1:] - x_frame[0])

    i_yield = np.argmax(A == N)
    i_tangent = np.argmax(tangent <= 0.95 * tangent[1])
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
        x_frame: Position of the load frame [nstep].
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
    ret["x_frame"] = root["x_frame"][...]
    ret["f_frame"] = np.empty((nstep), dtype=float)
    ret["f_potential"] = np.empty((nstep), dtype=float)
    ret["S"] = np.empty((nstep), dtype=int)
    ret["A"] = np.empty((nstep), dtype=int)
    ret["kick"] = root["kick"][...].astype(bool)
    ret["step"] = steps

    for j, step in enumerate(tqdm.tqdm(steps)):

        system.restore_quasistatic_step(root, step)

        if j == 0:
            i_n = system.i

        i = system.i
        ret["x_frame"][step] = system.x_frame
        ret["f_frame"][step] = np.mean(system.f_frame)
        ret["f_potential"][step] = -np.mean(system.f_potential)
        ret["S"][step] = np.sum(i - i_n)
        ret["A"][step] = np.sum(i != i_n)
        i_n = np.copy(i)

    ret["steadystate"] = steadystate(N=system.N, **ret)
    ret["x_frame"] /= system.normalisation.x
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
                        if key in ["potential", "system"]:
                            assert str(norm[key]) == str(test[key])
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

        for key in ["x_frame", "f_frame", "f_potential"]:
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

            system.restore_quasistatic_step(root, step)

            if last_start is not None:
                if np.all(np.equal(last_start, system.chunk.start)):
                    output[f"/QuasiStatic/{step:d}"] = output[f"/QuasiStatic/{last_step:d}"]
                    continue

            i = system.chunk.start
            output[f"/QuasiStatic/{step:d}/state"] = system.chunk.state_at(i)
            output[f"/QuasiStatic/{step:d}/index"] = i
            output[f"/QuasiStatic/{step:d}/value"] = system.chunk.data[:, 0]
            output.flush()
            last_start = np.copy(i)
            last_step = step


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

        keep = A == N
        file = file[keep]
        step = step[keep]

    bin_edges = np.logspace(-4, 1, 20001)
    count_x = np.zeros(bin_edges.size - 1, dtype=np.int64)
    count_xr = np.zeros(bin_edges.size - 1, dtype=np.int64)
    count_xl = np.zeros(bin_edges.size - 1, dtype=np.int64)

    roi = int((N - N % 2) / 2)
    roi = int(roi - roi % 2 + 1)
    ensemble = eye.Ensemble([roi], variance=True, periodic=True)

    if args.select is not None:
        if args.select < step.size:
            idx = np.sort(np.random.choice(np.arange(step.size), args.select, replace=False))
            file = file[idx]
            step = step[idx]

    with h5py.File(args.output, "w") as output:

        output["/yield_distance/bin_edges"] = bin_edges
        output["/yield_distance/count_right"] = count_xr
        output["/yield_distance/count_left"] = count_xl
        output["/yield_distance/count_any"] = count_x
        lower = 0
        lower_r = 0
        lower_l = 0
        upper = 0
        upper_r = 0
        upper_l = 0

        R = ensemble.result()
        V = np.zeros_like(R)
        A = ensemble.distance(0).astype(int)

        output["/heightheight/A"] = A[A >= 0]
        output["/heightheight/R"] = R[A >= 0]
        output["/heightheight/error"] = np.sqrt(V[A >= 0])

        output.flush()

        for f in tqdm.tqdm(np.unique(file)):

            with h5py.File(os.path.join(basedir, paths[f])) as source:

                system = allocate_system(source)

                for s in tqdm.tqdm(np.sort(step[file == f])):

                    system.restore_quasistatic_step(source["QuasiStatic"], s)

                    xr = system.y_right() - system.x
                    xl = system.x - system.y_left()
                    x = np.minimum(xl, xl)

                    i_x = np.digitize(x, bin_edges) - 1
                    i_xr = np.digitize(xr, bin_edges) - 1
                    i_xl = np.digitize(xl, bin_edges) - 1
                    n = bin_edges.size - 1

                    lower += np.sum(i_x < 0)
                    lower_r += np.sum(i_xr < 0)
                    lower_l += np.sum(i_xl < 0)
                    upper += np.sum(i_x >= n)
                    upper_r += np.sum(i_xr >= n)
                    upper_l += np.sum(i_xl >= n)

                    count_x += np.bincount(i_x[np.logical_and(i_x >= 0, i_x < n)], minlength=n)
                    count_xr += np.bincount(i_xr[np.logical_and(i_xr >= 0, i_xr < n)], minlength=n)
                    count_xl += np.bincount(i_xl[np.logical_and(i_xl >= 0, i_xl < n)], minlength=n)

                    ensemble.heightheight(system.x)

                output["/yield_distance/bin_edges"][...] = bin_edges
                output["/yield_distance/count_right"][...] = count_xr
                output["/yield_distance/count_left"][...] = count_xl
                output["/yield_distance/count_any"][...] = count_x

                R = ensemble.result()
                V = ensemble.variance()
                A = ensemble.distance(0).astype(int)

                output["/heightheight/A"][...] = A[A >= 0]
                output["/heightheight/R"][...] = R[A >= 0]
                output["/heightheight/error"][...] = np.sqrt(V[A >= 0])

                output.flush()

        count = output.create_group("/yield_distance/ignored")
        count.attrs["lower"] = lower
        count.attrs["lower_right"] = lower_r
        count.attrs["lower_left"] = lower_l
        count.attrs["upper"] = upper
        count.attrs["upper_right"] = upper_r
        count.attrs["upper_left"] = upper_l


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
            S = out["S"]

        x_frame = out["x_frame"][...]
        f_frame = out["f_frame"][...]
        f_potential = out["f_potential"][...]
        steadystate = None
        if "steadystate" in out:
            steadystate = out["steadystate"][...]

    opts = {}
    if args.marker is not None:
        opts["marker"] = args.marker

    fig, axes = gplt.subplots(ncols=2)

    axes[0].plot(x_frame, f_frame, label=r"$f_\text{frame}$", **opts)
    axes[0].plot(x_frame, f_potential, label=r"$f_\text{potential}$", **opts)

    if steadystate is not None:
        axes[0].axvline(x_frame[steadystate], c="k", ls="--", lw=1)

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
