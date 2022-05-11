"""
??
"""
from __future__ import annotations

import argparse
import inspect
import os
import textwrap

import FrictionQPotSpringBlock  # noqa: F401
import GooseHDF5 as g5
import h5py
import numpy as np
import tqdm

from . import QuasiStatic
from . import slurm
from . import storage
from . import tools
from ._version import version

basename = os.path.splitext(os.path.basename(__file__))[0]

entry_points = dict(
    cli_run="Trigger_run",
    cli_generate="Trigger_generate",
)


def replace_ep(doc: str) -> str:
    """
    Replace ``:py:func:`...``` with the relevant entry_point name
    """
    for ep in entry_points:
        doc = doc.replace(rf":py:func:`{ep:s}`", entry_points[ep])
    return doc


def cli_run(cli_args=None):
    """
    Trigger events from set state(s) once or multiple times.
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
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    basename = os.path.basename(args.file)

    with h5py.File(args.file, "a") as file:

        QuasiStatic.create_check_meta(file, f"/meta/{progname}", dev=args.develop)
        branches = file["stored"][...]
        n = branches.size
        pbar = tqdm.tqdm(total=n, desc=f"{basename}: branch = {0:8d}, p = {0:8d}, S = {0:8d}")
        dx = file["/event_driven/dx"][...]
        system = QuasiStatic.System(file)
        N = system.N()

        for ibranch in branches:

            for p in range(N):
                system.restore_quasistatic_step(file[f"/branch/{ibranch:d}"], 0)
                inc = system.inc()
                i_n = system.i()
                system.trigger(p=p, eps=dx, direction=1)

                while True:
                    niter = system.minimise_timeactivity(nmargin=30)
                    if niter > 0:
                        break
                    system.chunk_rshift()

                S = np.sum(system.i() - i_n)

                if S > 0:
                    break

            pbar.n = ibranch
            pbar.set_description(f"{basename}: branch = {ibranch:8d}, p = {p:8d}, S = {S:8d}")
            pbar.refresh()

            file["/output/completed"][ibranch] = True
            file["/output/S"][ibranch] = S
            file["/output/A"][ibranch] = np.sum(system.i() != i_n)
            file["/output/T"][ibranch] = system.quasistaticActivityLast() - inc
            file["/output/p"][ibranch] = p
            file["/output/f_frame"][ibranch] = np.mean(system.f_frame())

            storage.dset_extend1d(file, f"/branch/{ibranch:d}/inc", 1, system.inc())
            storage.dset_extend1d(file, f"/branch/{ibranch:d}/x_frame", 1, system.x_frame())
            file[f"/branch/{ibranch:d}/x/1"] = system.x()
            file.flush()


def _get_force_increment(step, force, kick, target_force):
    """
    :return:
        ``step``: Step to load or to start elastic loading.
        ``target_force``: Target force (modified if needed).
        ``load``: If ``True``, elastic loading until ``target_force`` is needed.
    """

    s = step.reshape(-1, 2)
    f = force.reshape(-1, 2)
    k = kick.reshape(-1, 2)

    t = np.sum(f < target_force, axis=1)
    assert np.all(k[:, 0])
    assert np.sum(t) > 0

    if np.sum(t == 1) >= 1:
        j = np.argmax(t == 1)
        if np.abs(f[j, 0] - target_force) / f[j, 0] < 1e-4:
            return s[j, 0], f[j, 0], False
        if np.abs(f[j, 1] - target_force) / f[j, 1] < 1e-4:
            return s[j, 1], f[j, 1], False
        return s[j, 0], target_force, True

    j = np.argmax(t == 0)
    return s[j, 0], f[j, 0], False


def cli_generate(cli_args=None):
    """
    Branch simulation:

    -   (default) After each system-spanning event.
    -   (--delta-f) At a fixed force increment (of the frame force) after a system-spanning event.
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

    parser.add_argument("--delta-f", type=float, help="Advance to fixed force")
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-o", "--outdir", type=str, default=".", help="Output directory")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-w", "--time", type=str, default="72h", help="Walltime")
    parser.add_argument("ensembleinfo", type=str, help="EnsembleInfo (read-only)")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.ensembleinfo)

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    basedir = os.path.dirname(args.ensembleinfo)

    with h5py.File(args.ensembleinfo, "r") as info:

        files = sorted(info["full"])
        assert np.all([os.path.exists(os.path.join(basedir, file)) for file in files])
        assert not np.any([os.path.exists(os.path.join(args.outdir, file)) for file in files])

        N = info["/normalisation/N"][...]

        for filename in tqdm.tqdm(files):

            with h5py.File(os.path.join(basedir, filename)) as source, h5py.File(
                os.path.join(args.outdir, filename), "w"
            ) as dest:

                g5.copy(source, dest, ["/event_driven", "/param"])

                system = QuasiStatic.System(source)

                steadystate = info["full"][filename]["steadystate"][...]
                A = info["full"][filename]["A"][...]
                step = info["full"][filename]["step"][...]
                kick = info["full"][filename]["kick"][...]
                f_frame = info["full"][filename]["f_frame"][...]
                assert np.all(A[~kick] == 0)
                ss = step[np.logical_and(A == N, step > steadystate)]
                n = ss.size - 1

                QuasiStatic.create_check_meta(dest, f"/meta/{progname}", dev=args.develop)

                key = "/output/completed"
                dest[key] = np.zeros(n, dtype=bool)
                dest[key].attrs["desc"] = "Signal if trigger was successfully completed"

                key = "/output/S"
                dest[key] = np.zeros(n, dtype=np.int64)
                dest[key].attrs["desc"] = "Total number of yielding events during event"

                key = "/output/A"
                dest[key] = np.zeros(n, dtype=np.int64)
                dest[key].attrs["desc"] = "Number of yielding particles during event"

                key = "/output/T"
                dest[key] = np.zeros(n, dtype=np.int64)
                dest[key].attrs["desc"] = "Event duration"

                key = "/output/step_c"
                dest[key] = np.zeros(n, dtype=np.int64)
                dest[key].attrs["desc"] = "Quasi-static-load-step of last system-spanning event"

                key = "/output/step"
                dest[key] = np.zeros(n, dtype=np.int64)
                dest[key].attrs["desc"] = "Quasi-static-load-step on which configuration is based"

                key = "/output/loaded"
                dest[key] = np.zeros(n, dtype=bool)
                dest[key].attrs["desc"] = "True if 'elastic' loading was applied to 'step'"

                key = "/output/p"
                dest[key] = np.zeros(n, dtype=np.int64)
                dest[key].attrs["desc"] = "Particle that was pushed (particle from p = 0 are tried)"

                key = "/output/f_frame_0"
                dest[key] = np.zeros(n, dtype=np.float64)
                dest[key].attrs["desc"] = "Frame force before the trigger"

                key = "/output/f_frame"
                dest[key] = np.zeros(n, dtype=np.float64)
                dest[key].attrs["desc"] = "Frame force after the trigger"

                dest["/stored"] = np.arange(n, dtype=np.int64)

                for ibranch, (start, stop) in enumerate(zip(tqdm.tqdm(ss[:-1]), ss[1:])):

                    if args.delta_f is None:
                        s, f, load = start, f_frame[start], False
                    else:
                        s, f, load = _get_force_increment(
                            step=step[start:stop],
                            force=f_frame[start:stop],
                            kick=kick[start:stop],
                            target_force=f_frame[start] + args.delta_f,
                        )

                    x = source[f"/x/{s:d}"][...]
                    x_frame = source["/x_frame"][s]
                    inc = source["/inc"][s]

                    if load:
                        system.restore_quasistatic_step(source, s)
                        system.advanceToFixedForce(f)
                        x = system.x()
                        x_frame = system.x_frame()

                    dest["/output/step_c"][ibranch] = start
                    dest["/output/step"][ibranch] = s
                    dest["/output/loaded"][ibranch] = load
                    dest["/output/f_frame_0"][ibranch] = f

                    dest[f"/branch/{ibranch:d}/x/0"] = x
                    storage.create_extendible(dest, f"/branch/{ibranch:d}/inc", np.uint64)
                    storage.create_extendible(dest, f"/branch/{ibranch:d}/x_frame", np.float64)
                    storage.dset_extend1d(dest, f"/branch/{ibranch:d}/inc", 0, inc)
                    storage.dset_extend1d(dest, f"/branch/{ibranch:d}/x_frame", 0, x_frame)
                    dest.flush()

    executable = entry_points["cli_run"]
    commands = [f"{executable} {file}" for file in files]

    slurm.serial_group(
        commands,
        basename=executable,
        group=1,
        outdir=args.outdir,
        sbatch={"time": args.time},
    )
