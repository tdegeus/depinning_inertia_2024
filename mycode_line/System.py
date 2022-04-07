"""
??
"""
from __future__ import annotations

import argparse
import inspect
import os
import re
import sys
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
from . import tag
from . import tools
from . import storage
from ._version import version


entry_points = dict(
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
                type = "weibull",
                offset = file_yield["weibull"]["offset"][...],
                mean = file_yield["weibull"]["mean"][...],
                k = file_yield["weibull"]["k"][...],
            )
        else:
            raise OSError("Distribution not supported")

        super().__init__(
            m = file["param"]["m"][...],
            eta = file["param"]["eta"][...],
            mu = file["param"]["mu"][...],
            k_neighbours = file["param"]["k_neighbours"][...],
            k_frame = file["param"]["k_frame"][...],
            dt = file["param"]["dt"][...],
            x_yield = np.cumsum(self._draw_chunk(), axis=1) + xoffset,
            istart = self.state_istart,
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

    def rshift_chunk(self):

        shift = np.argmax(self.y() >= self.x().reshape(-1, 1), axis=1) - self.nbuffer
        assert np.min(shift) > 0
        self.generators.restore(self.state)
        self.generators.advance(shift)
        self.state = self.generators.state()
        self.state_istart += shift
        self.shift_dy(istart=self.state_istart, dy=self._draw_chunk())

    def restore_inc(self, file: h5py.File, inc: int):
        """
        Restore an increment.

        :param file: Open simulation HDF5 archive (read-only).
        :param inc: Increment number.
        """

        self.quench()
        self.set_t(file["/t"][inc])
        self.set_x(file[f"/x/{inc:d}"][...])


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
    parser.add_argument("--overdamped", action="store_true", help="Job scripts for overdamped run")
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

            file["/param/m"] = 1.0,
            file["/param/eta"] = args.eta,
            file["/param/mu"] = 1.0,
            file["/param/k_neighbours"] = 1.0,
            file["/param/k_frame"] = 1.0 / args.size,
            file["/param/dt"] = args.dt,
            file["/param/xyield/initstate"] = seed + np.arange(args.size).astype(np.int64)
            file["/param/xyield/initseq"] = np.zeros(args.size, dtype=np.int64)
            file["/param/xyield/nchunk"] = 1500
            file["/param/xyield/nbuffer"] = 200
            file["/param/xyield/xoffset"] = -100.0
            file["/param/xyield/weibull/offset"] = 1e-5
            file["/param/xyield/weibull/mean"] = 1.0
            file["/param/xyield/weibull/k"] = 2.0
            file["/event_driven/dx"] = 1e-3

    executable = entry_points["cli_run"]

    if args.overdamped:
        executable = f"{executable} --overdamped"

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
    parser.add_argument("--overdamped", action="store_true", help="Job scripts for overdamped run")
    parser.add_argument("-n", "--ninc", type=int, default=1000, help="#increments to run")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)

    basename = os.path.basename(args.file)

    with h5py.File(args.file, "a") as file:

        system = System(file)
        meta = create_check_meta(file, f"/meta/{progname}", dev=args.develop)

        if args.overdamped:
            minimise = system.minimise_nopassing
        else:
            minimise = system.minimise

        if "stored" not in file:
            minimise()
            system.set_t(0.0)
            file["/x/0"] = system.x()
            storage.create_extendible(file, "/stored", np.uint64, desc="List of stored increments")
            storage.create_extendible(file, "/t", np.float64, desc=f"Time (end of increment).")
            storage.create_extendible(file, "/event_driven/kick", bool, desc=f"Kick used.")
            storage.dset_extend1d(file, "/stored", 0, 0)
            storage.dset_extend1d(file, "/t", 0, system.t())
            storage.dset_extend1d(file, "/event_driven/kick", 0, True)
            file.flush()

        inc = int(file["/stored"][-1])
        inc0 = inc
        kick = file["/event_driven/kick"][inc]
        dx = file["/event_driven/dx"][...]
        system.restore_inc(file, inc)

        pbar = tqdm.tqdm(
            total=args.ninc, desc=f"{basename}: inc = {inc:8d}, niter = {'-':8s}"
        )

        for inc in range(inc + 1, inc + 1 + args.ninc):

            if np.any(system.i() - system.istart() > system.nchunk - system.nbuffer):
                system.rshift_chunk()

            kick = not kick
            system.eventDrivenStep(dx, kick)

            if kick:
                niter = minimise()
                pbar.n = inc - inc0
                pbar.set_description(f"{basename}: inc = {inc:8d}, niter = {niter:8d}")
                pbar.refresh()

            storage.dset_extend1d(file, "/stored", inc, inc)
            storage.dset_extend1d(file, "/t", inc, system.t())
            storage.dset_extend1d(file, "/event_driven/kick", inc, kick)
            file[f"/x/{inc:d}"] = system.x()
            file.flush()

        pbar.set_description(f"{basename}: inc = {inc:8d}, {'completed':16s}")

