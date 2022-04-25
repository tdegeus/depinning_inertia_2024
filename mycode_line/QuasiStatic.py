"""
??
"""
from __future__ import annotations

import argparse
import inspect
import os
import re
import textwrap
import uuid

import FrictionQPotSpringBlock  # noqa: F401
import FrictionQPotSpringBlock.Line1d as model
import h5py
import numpy as np
import prrng
import tqdm
from numpy.typing import ArrayLike

from . import slurm
from . import storage
from . import tag
from . import tools
from ._version import version


entry_points = dict(
    cli_ensembleinfo="EnsembleInfo",
    cli_generate="Run_generate",
    cli_run="Run",
)


file_defaults = dict(
    cli_ensembleinfo="EnsembleInfo.h5",
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


class System(model.System):
    """
    Similar to :py:class:`model.System`, but with file interaction.
    """

    def __init__(self, file: h5py.File, nchunk: int = None):
        """
        Initialise system.

        :param nchunk: Overwrite the default chuck
        """

        file_yield = file["param"]["xyield"]
        initstate = file_yield["initstate"][...]
        initseq = file_yield["initseq"][...]
        xoffset = file_yield["xoffset"][...]

        self.generators = prrng.pcg32_array(initstate, initseq)
        self.nchunk = file_yield["nchunk"][...] if nchunk is None else nchunk
        self.nbuffer = file_yield["nbuffer"][...]
        self.state = self.generators.state()
        self.state_istart = np.zeros(initstate.size, dtype=int)

        if "weibull" in file_yield:
            self.distribution = dict(
                type="weibull",
                offset=file_yield["weibull"]["offset"][...],
                mean=file_yield["weibull"]["mean"][...],
                k=file_yield["weibull"]["k"][...],
            )
        else:
            raise OSError("Distribution not supported")

        super().__init__(
            m=file["param"]["m"][...],
            eta=file["param"]["eta"][...],
            mu=file["param"]["mu"][...],
            k_neighbours=file["param"]["k_neighbours"][...],
            k_frame=file["param"]["k_frame"][...],
            dt=file["param"]["dt"][...],
            x_yield=np.cumsum(self._draw_chunk(), axis=1) + xoffset,
            istart=self.state_istart,
        )

    def _draw_chunk(self):
        """
        Draw chunk of yield distances.
        """

        if self.distribution["type"] == "weibull":
            ret = self.generators.weibull([self.nchunk], self.distribution["k"])
            ret *= 2.0 * self.distribution["mean"]
            ret += self.distribution["offset"]
            return ret

    def _chuck_shift(self, shift: ArrayLike):
        """
        Apply shift to current chunk.

        :param shift: Shift per particle.
        """

        self.generators.restore(self.state)
        self.generators.advance(shift)
        self.state = self.generators.state()
        self.state_istart += shift
        self.shift_dy(istart=self.state_istart, dy=self._draw_chunk())

    def chunk_rshift(self):
        """
        Shift chunk in positive direction. Asserts that positions are in the current chunk.
        """

        y = self.y()
        x = self.x()
        assert np.all(y[:, -1] > x)
        assert np.all(y[:, 0] < x)
        shift = np.argmax(self.y() >= self.x().reshape(-1, 1), axis=1) - self.nbuffer
        assert np.all(shift > self.nbuffer - self.nchunk)
        shift = np.where(shift < 0, 0, shift)
        self._chuck_shift(shift)

    def _chunk_goto(self):
        """
        Shift until the positions are in the current chunk.
        """

        x = self.x()

        while True:
            y = self.y()
            if np.all(np.logical_and(y[:, 0] < x, y[:, -1] > x)):
                return
            shift = np.argmax(self.y() >= self.x().reshape(-1, 1), axis=1) - self.nbuffer
            assert np.all(shift > self.nbuffer - self.nchunk)
            shift = np.where(shift < 0, 0, shift)
            shift = np.where(y[:, -1] <= x, self.nchunk - 1, shift)
            self._chuck_shift(shift)

    def restore_inc(self, file: h5py.File, inc: int):
        """
        Restore an increment.

        :param file: Open simulation HDF5 archive (read-only).
        :param inc: Increment number.
        """

        self.quench()
        self.set_inc(file["/inc"][inc])
        self.set_x_frame(file["/x_frame"][inc])
        self.set_x(file[f"/x/{inc:d}"][...])
        self._chunk_goto()


def create_check_meta(
    file: h5py.File,
    path: str,
    ver: str = version,
    deps: str = dependencies(model),
    dev: bool = False,
) -> h5py.Group:
    """
    Create or read/check metadata. This function asserts that:

    -   There are no uncommitted changes.
    -   There are no version changes.

    It create metadata as attributes to a group ``path`` as follows::

        "uuid": A unique identifier that can be used to distinguish simulations if needed.
        "version": The current version of this code (see below).
        "dependencies": The current version of all relevant dependencies (see below).

    :param file: HDF5 archive.
    :param path: Path in ``file`` to store/read metadata.
    :param ver: Version string.
    :param deps: List of dependencies.
    :param dev: Allow uncommitted changes.
    :return: Group to metadata.
    """

    assert dev or not tag.has_uncommitted(ver)
    assert dev or not tag.any_has_uncommitted(deps)

    if path not in file:
        meta = file.create_group(path)
        meta.attrs["uuid"] = str(uuid.uuid4())
        meta.attrs["version"] = ver
        meta.attrs["dependencies"] = deps
        return meta

    meta = file[path]
    assert dev or tag.equal(ver, meta.attrs["version"])
    assert dev or tag.all_equal(deps, meta.attrs["dependencies"])
    return meta


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
    parser.add_argument("--nopassing", action="store_true", help="Job scripts for overdamped run")
    parser.add_argument("--eta", type=float, default=2.0 * np.sqrt(3.0) / 10.0, help="Damping")
    parser.add_argument("--dt", type=float, default=0.1, help="Time-step")
    parser.add_argument("-n", "--nsim", type=int, default=1, help="#simulations")
    parser.add_argument("-N", "--size", type=int, default=5000, help="#particles")
    parser.add_argument("-s", "--start", type=int, default=0, help="Start simulation")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-w", "--time", type=str, default="72h", help="Walltime")
    parser.add_argument("outdir", type=str, help="Output directory")

    args = tools._parse(parser, cli_args)

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    files = []

    for i in range(args.start, args.start + args.nsim):

        files += [f"id={i:04d}.h5"]
        seed = i * args.size

        with h5py.File(os.path.join(args.outdir, files[-1]), "w") as file:

            file["/meta/normalisation/N"] = args.size
            file["/meta/seed_base"] = seed
            file["/param/m"] = 1.0
            file["/param/eta"] = args.eta
            file["/param/mu"] = 1.0
            file["/param/k_neighbours"] = 1.0
            file["/param/k_frame"] = 1.0 / args.size
            file["/param/dt"] = args.dt
            file["/param/xyield/initstate"] = seed + np.arange(args.size).astype(np.int64)
            file["/param/xyield/initseq"] = np.zeros(args.size, dtype=np.int64)
            file["/param/xyield/nchunk"] = 5000
            file["/param/xyield/nbuffer"] = 300
            file["/param/xyield/xoffset"] = -100.0
            file["/param/xyield/weibull/offset"] = 1e-5
            file["/param/xyield/weibull/mean"] = 1.0
            file["/param/xyield/weibull/k"] = 2.0
            file["/event_driven/dx"] = 1e-3

    executable = entry_points["cli_run"]

    if args.nopassing:
        commands = [f"{executable} --nopassing {file}" for file in files]
    else:
        commands = [f"{executable} {file}" for file in files]

    slurm.serial_group(
        commands,
        basename=executable,
        group=1,
        outdir=args.outdir,
        sbatch={"time": args.time},
    )


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

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("--nopassing", action="store_true", help="Job scripts for overdamped run")
    parser.add_argument("-n", "--ninc", type=int, default=1000, help="#increments to run")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)

    basename = os.path.basename(args.file)

    with h5py.File(args.file, "a") as file:

        system = System(file)
        meta = create_check_meta(file, f"/meta/{progname}", dev=args.develop)

        if args.nopassing:
            minimise = system.minimise_nopassing
            meta.attrs["dynamics"] = "nopassing"
        else:
            minimise = system.minimise
            meta.attrs["dynamics"] = "normal"

        if "stored" not in file:
            minimise()
            system.set_t(0.0)
            file["/x/0"] = system.x()
            storage.create_extendible(file, "/stored", np.uint64, desc="List of stored increments")
            storage.create_extendible(file, "/inc", np.uint64, desc="Time.")
            storage.create_extendible(file, "/x_frame", np.float64, desc="Position of load frame.")
            storage.create_extendible(file, "/event_driven/kick", bool, desc="Kick used.")
            storage.dset_extend1d(file, "/stored", 0, 0)
            storage.dset_extend1d(file, "/inc", 0, system.inc())
            storage.dset_extend1d(file, "/x_frame", 0, system.x_frame())
            storage.dset_extend1d(file, "/event_driven/kick", 0, True)
            file.flush()

        inc = int(file["/stored"][-1])
        inc0 = inc
        kick = file["/event_driven/kick"][inc]
        dx = file["/event_driven/dx"][...]
        system.restore_inc(file, inc)

        pbar = tqdm.tqdm(total=args.ninc, desc=f"{basename}: inc = {inc:8d}, niter = {'-':8s}")

        for inc in range(inc + 1, inc + 1 + args.ninc):

            if np.any(system.i() - system.istart() > system.nchunk - system.nbuffer):
                system.chunk_rshift()

            kick = not kick
            system.eventDrivenStep(dx, kick)

            if kick:
                niter = minimise()
                pbar.n = inc - inc0
                pbar.set_description(f"{basename}: inc = {inc:8d}, niter = {niter:8d}")
                pbar.refresh()

            storage.dset_extend1d(file, "/stored", inc, inc)
            storage.dset_extend1d(file, "/inc", inc, system.inc())
            storage.dset_extend1d(file, "/x_frame", inc, system.x_frame())
            storage.dset_extend1d(file, "/event_driven/kick", inc, kick)
            file[f"/x/{inc:d}"] = system.x()
            file.flush()

        pbar.set_description(f"{basename}: inc = {inc:8d}, {'completed':16s}")


def normalisation(file: h5py.File):
    """
    Read normalisation from file (or use default value in "classic" mode).

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
    """

    ret = {}
    ret["mu"] = file["param"]["mu"][...]
    ret["k_frame"] = file["param"]["k_frame"][...]
    ret["k_neighbours"] = file["param"]["k_neighbours"][...]
    ret["eta"] = file["param"]["eta"][...]
    ret["m"] = file["param"]["m"][...]
    ret["seed"] = file["meta"]["seed_base"][...]
    ret["N"] = file["meta"]["normalisation"]["N"][...]
    ret["dt"] = file["param"]["dt"][...]
    return ret


def steadystate(
    x_frame: ArrayLike, f_frame: ArrayLike, kick: ArrayLike, A: ArrayLike, N: int, **kwargs
) -> int:
    """
    Estimate the first increment of the steady-state. Constraints:
    -   Start with elastic loading.
    -   Sufficiently low tangent modulus.
    -   All blocks yielded at least once.

    .. note::

        Keywords arguments that are not explicitly listed are ignored.

    :param x_frame: Position of the load frame [ninc].
    :param f_frame: Average driving force [ninc].
    :param kick: Whether a kick was applied [ninc].
    :param A: Number of blocks that yielded at least once [ninc].
    :param N: Number of blocks.
    :return: Increment number.
    """

    if f_frame.size <= 2:
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
        x_frame: Position of the load frame [ninc].
        f_frame: Average driving force [ninc].
        f_potential: Average elastic force [ninc].
        S: Number of times a particle yielded [ninc].
        A: Number of particles that yielded at least once [ninc].
        kick: Increment started with a kick (True), or contains only elastic loading (False) [ninc].
        inc: Increment numbers == np.arange(ninc).
        steadystate: Increment number where the steady state starts (int).
        seed: Base seed (uint64).
        mu: Elastic stiffness (float).
        k_frame: Stiffness of the load-frame (float).
        k_neighbours: Stiffness of the neighbour interactions (float).
        eta: Damping (float).
        m: Mass (float).
        seed: Base seed (uint64) or uuid (str).
        N: Number of blocks (int).
        dt: Time step of time discretisation.
    """

    system = System(file)
    incs = file["/stored"][...]
    ninc = incs.size
    assert ninc >= 1
    assert all(incs == np.arange(ninc))
    system.restore_inc(file, 0)
    i_n = system.i()

    ret = normalisation(file)
    ret["x_frame"] = file["x_frame"][...]
    ret["f_frame"] = np.empty((ninc), dtype=float)
    ret["f_potential"] = np.empty((ninc), dtype=float)
    ret["S"] = np.empty((ninc), dtype=int)
    ret["A"] = np.empty((ninc), dtype=int)
    ret["kick"] = file["event_driven"]["kick"][...].astype(bool)
    ret["inc"] = incs

    for inc in incs:

        system.restore_inc(file, inc)
        i = system.i()
        ret["x_frame"][inc] = system.x_frame()
        ret["f_frame"][inc] = np.mean(system.f_frame())
        ret["f_potential"][inc] = -np.mean(system.f_potential())
        ret["S"][inc] = np.sum(i - i_n)
        ret["A"][inc] = np.sum(i != i_n)
        i_n = np.copy(i)

    ret["steadystate"] = steadystate(**ret)

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
        dependencies=[],
    )

    fields_full = ["x_frame", "f_frame", "f_potential", "S", "A", "kick", "inc"]
    combine_load = {key: [] for key in fields_full}
    combine_kick = {key: [] for key in fields_full}
    file_load = []
    file_kick = []

    fmt = "{:" + str(max(len(i) for i in info["filepath"])) + "s}"
    pbar = tqdm.tqdm(info["filepath"])
    pbar.set_description(fmt.format(""))

    with h5py.File(args.output, "w") as output:

        for i, (filename, filepath) in enumerate(zip(pbar, args.files)):

            pbar.set_description(fmt.format(filename), refresh=True)

            with h5py.File(filepath, "r") as file:

                if i == 0:
                    norm = normalisation(file)
                else:
                    test = normalisation(file)
                    for key in norm:
                        if key not in ["seed"]:
                            assert np.isclose(norm[key], test[key])

                out = basic_output(file)

                for key in fields_full:
                    output[f"/full/{filename}/{key}"] = out[key]
                if out["steadystate"] is not None:
                    output[f"/full/{filename}/steadystate"] = out["steadystate"]
                output.flush()

                info["seed"].append(out["seed"])

                meta = file[f"/meta/{entry_points['cli_run']}"]
                for key in ["uuid", "version", "dependencies", "dynamics"]:
                    info[key].append(meta.attrs[key])

                if out["steadystate"] is None:
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

        for key in ["A", "inc"]:
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
        assert all(np.equal(combine_kick["inc"][ss], combine_load["inc"][ss] + 1))
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
        tools.h5py_save_unique(
            [";".join(i) for i in info["dependencies"]], output, "/lookup/dependencies", split=";"
        )
        output["files"] = output["/lookup/filepath"]

        # metadata for this program

        meta = create_check_meta(output, f"/meta/{progname}", dev=args.develop)
