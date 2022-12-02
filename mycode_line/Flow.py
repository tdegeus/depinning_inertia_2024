"""
??
"""
from __future__ import annotations

import argparse
import inspect
import os
import pathlib
import re
import textwrap
from collections import defaultdict

import FrictionQPotSpringBlock  # noqa: F401
import GooseHDF5 as g5
import h5py
import numpy as np
import shelephant
import tqdm

from . import QuasiStatic
from . import storage
from . import tools
from ._version import version

basename = os.path.splitext(os.path.basename(__file__))[0]

entry_points = dict(
    cli_ensembleinfo="Flow_EnsembleInfo",
    cli_generate="Flow_Generate",
    cli_plot="Flow_Plot",
    cli_run="Flow_Run",
)


file_defaults = dict(
    cli_ensembleinfo="Flow_EnsembleInfo.h5",
)


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
        if key in ["gammadot", "jump"]:
            info[key] = float(info[key])
        else:
            info[key] = int(info[key])

    return info


def ensemble_average(file: h5py.File | dict, interval: int = 100):
    """
    Ensemble average from a file written by :py:func:`cli_enembleinfo`.

    :param file: Ensemble info (opened HDF5 archive).
    :param interval: Average of the last ``interval`` output steps.
    """

    f_frame = defaultdict(list)
    f_potential = defaultdict(list)

    f_frame_std = defaultdict(list)
    f_potential_std = defaultdict(list)

    eta = file["/param/eta"][...]
    root = file["/full"]

    for config in root:

        gammadot = root[config]["gammadot"][...]
        frame = root[config]["f_frame"][...]
        potential = root[config]["f_potential"][...]

        b3 = -3 * interval
        b2 = -2 * interval
        b1 = -1 * interval

        f = np.mean(frame[b1:])
        df = np.std(frame[b1:])

        p0 = np.mean(potential[b3:b2])
        p1 = np.mean(potential[b2:b1])
        p = np.mean(potential[b1:])
        dp = np.std(potential[b1:])

        if p0 <= p - dp or p0 >= p + dp:
            continue

        if p1 <= p - dp or p1 >= p + dp:
            continue

        f_frame[f"{gammadot:.1f}"].append(f)
        f_potential[f"{gammadot:.1f}"].append(p)

        f_frame_std[f"{gammadot:.1f}"].append(df)
        f_potential_std[f"{gammadot:.1f}"].append(dp)

    n = max([len(v) for v in f_frame.values()])
    rm = []

    for key, value in f_frame.items():
        if len(value) < n:
            rm.append(key)

    for key in rm:
        del f_frame[key]
        del f_potential[key]
        del f_frame_std[key]
        del f_potential_std[key]

    for key in f_frame:
        f_frame[key] = np.array(f_frame[key])
        f_potential[key] = np.array(f_potential[key])
        f_frame_std[key] = np.array(f_frame_std[key])
        f_potential_std[key] = np.array(f_potential_std[key])

    for key in f_frame:
        f_frame[key] = np.mean(f_frame[key])
        f_potential[key] = np.mean(f_potential[key])
        f_frame_std[key] = np.max(f_potential_std[key])
        f_potential_std[key] = np.max(f_potential_std[key])

    x = []
    yf = []
    yp = []
    ef = []
    ep = []

    for key in f_frame:

        if -f_potential[key] - f_potential_std[key] < 0:
            continue

        if np.abs((-f_potential[key] + eta * float(key) - f_frame[key]) / f_frame[key]) > 1e-2:
            continue

        x.append(float(key))
        yf.append(f_frame[key])
        yp.append(-f_potential[key])
        ef.append(f_frame_std[key])
        ep.append(f_potential_std[key])

    gammmadot = np.array(x)
    f_frame = np.array(yf)
    f_potential = np.array(yp)
    f_frame_std = np.array(ef)
    f_potential_std = np.array(ep)

    sorter = np.argsort(gammmadot)
    gammmadot = gammmadot[sorter]
    f_frame = f_frame[sorter]
    f_potential = f_potential[sorter]
    f_frame_std = f_frame_std[sorter]
    f_potential_std = f_potential_std[sorter]

    return gammmadot, f_frame, f_potential, f_frame_std, f_potential_std


def cli_ensembleinfo(cli_args=None):
    """
    Extract basic output and combine into a single file.
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
    output = file_defaults[funcname]

    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("files", nargs="*", type=str, help="Input files")

    args = tools._parse(parser, cli_args)
    assert len(args.files) > 0
    assert all([os.path.isfile(file) for file in args.files])
    tools._check_overwrite_file(args.output, args.force)

    with h5py.File(args.output, "w") as output:

        QuasiStatic.create_check_meta(output, f"/meta/{progname}", dev=args.develop)

        for i, filepath in enumerate(tqdm.tqdm(args.files)):

            fname = os.path.relpath(filepath, os.path.dirname(args.output))
            fname = fname.replace("/", "_")

            with h5py.File(filepath) as file:

                output[f"/full/{fname}/f_frame"] = file["/Flow/output/f_frame"][...]
                output[f"/full/{fname}/f_potential"] = file["/Flow/output/f_potential"][...]
                output[f"/full/{fname}/f_damping"] = file["/Flow/output/f_damping"][...]
                output[f"/full/{fname}/gammadot"] = file["/Flow/gammadot"][...]

                if i == 0:
                    g5.copy(file, output, "/param")

        gammmadot, f_frame, f_potential, f_frame_std, f_potential_std = ensemble_average(output)
        output["/average/gammadot"] = gammmadot
        output["/average/mean/f_frame"] = f_frame
        output["/average/mean/f_potential"] = f_potential
        output["/average/std/f_frame"] = f_frame_std
        output["/average/std/f_potential"] = f_potential_std


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

    parser.add_argument(
        "--nstep",
        type=lambda x: int(float(x)),
        help="#output steps to run.",
    )
    parser.add_argument(
        "--output",
        type=lambda x: int(float(x)),
        help="Number of time-steps between writing global output variables.",
    )
    parser.add_argument(
        "--snapshot",
        type=lambda x: int(float(x)),
        default=0,
        help="Write snapshot every n output steps.",
    )
    parser.add_argument(
        "-n",
        "--nsim",
        type=lambda x: int(float(x)),
        default=1,
        help="#simulations",
    )
    parser.add_argument(
        "-N",
        "--size",
        type=lambda x: int(float(x)),
        default=5000,
        help="#particles",
    )
    parser.add_argument(
        "--develop",
        action="store_true",
        help="Allow uncommitted changes",
    )
    parser.add_argument(
        "--dt",
        type=float,
        help="Time-step",
    )
    parser.add_argument(
        "--eta",
        type=float,
        required=True,
        help="Damping coefficient.",
    )
    parser.add_argument(
        "--gammadot",
        type=float,
        required=True,
        help="Driving rate.",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=0,
        help="Start simulation (correct seed if extending ensemble.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=version,
    )
    parser.add_argument(
        "outdir",
        type=str,
        help="Output directory",
    )

    args = tools._parse(parser, cli_args)

    known_gammadot = np.array([1e-2, 1e-1, 1e-0, 1e1])
    known_output = np.array([1e3, 1e3, 1e3, 1e2])
    known_nstep = np.array([1e4, 1e4, 1e4, 1e4])
    if args.eta > 1e0:
        known_nstep *= 1e5 * np.ones_line(known_nstep)
    if args.output is None:
        args.output = int(np.interp(args.gammadot, known_gammadot, known_output))
    if args.nstep is None:
        args.nstep = int(np.interp(args.gammadot, known_gammadot, known_nstep))

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = []

    for i in range(args.start, args.start + args.nsim):

        files += [f"id={i:04d}.h5"]
        seed = i * args.size

        with h5py.File(outdir / files[-1], "w") as file:
            QuasiStatic.generate(
                file=file,
                N=args.size,
                seed=seed,
                eta=args.eta,
                dt=args.dt,
            )
            file["/Flow/gammadot"] = args.gammadot
            file["/Flow/output/interval"] = args.output
            file["/Flow/snapshot/interval"] = args.output * args.snapshot

    executable = entry_points["cli_run"]
    commands = [f"{executable} --nstep {args.nstep:d} {file}" for file in files]
    shelephant.yaml.dump(outdir / "commands.yaml", commands)


def run_create_extendible(file: h5py.File):
    """
    Create extendible datasets used in :py:func:`cli_run`.
    """

    storage.create_extendible(file, "/Flow/output/f_frame", np.float64)
    storage.create_extendible(file, "/Flow/output/f_potential", np.float64)
    storage.create_extendible(file, "/Flow/output/f_damping", np.float64)
    storage.create_extendible(file, "/Flow/output/x", np.float64)
    storage.create_extendible(file, "/Flow/output/inc", np.uint32)
    storage.create_extendible(file, "/Flow/snapshot/inc", np.uint32)


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
    parser.add_argument(
        "-n", "--nstep", type=lambda x: int(float(x)), default=1000, help="#output steps to run"
    )
    parser.add_argument(
        "--snapshot",
        type=lambda x: int(float(x)),
        help="Write snapshot every n output steps.",
    )
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    pbar = tqdm.tqdm(range(args.nstep), desc=args.file)

    with h5py.File(args.file, "a") as file:

        restart = False

        if "/Flow/snapshot/x" in file:
            restart = True

        inc = 0
        snapshot = file["/Flow/snapshot/interval"][...]
        output = file["/Flow/output/interval"][...]
        gammadot = file["/Flow/gammadot"][...]
        assert snapshot % output == 0

        if args.snapshot:
            snapshot = output * args.snapshot

        system = QuasiStatic.allocate_system(file)
        QuasiStatic.create_check_meta(file, f"/meta/{progname}", dev=args.develop)

        if restart:
            dt = file["/param/dt"][...]
            inc = file["/Flow/snapshot/inc"][-1]
            system.x = file[f"/Flow/snapshot/x/{inc:d}"][...]
            system.v = file[f"/Flow/snapshot/v/{inc:d}"][...]
            system.a = file[f"/Flow/snapshot/a/{inc:d}"][...]
            system.x_frame = gammadot * dt * inc
            i = int(inc / output)
            assert np.isclose(file["/Flow/output/f_frame"][i], np.mean(system.f_frame))
            assert np.isclose(file["/Flow/output/f_potential"][i], np.mean(system.f_potential))
            assert np.isclose(file["/Flow/output/f_damping"][i], np.mean(system.f_damping))
            assert np.isclose(file["/Flow/output/x"][i], np.mean(system.x))
        else:
            run_create_extendible(file)

        output_fields = [
            "/Flow/output/inc",
            "/Flow/output/f_frame",
            "/Flow/output/f_potential",
            "/Flow/output/f_damping",
            "/Flow/output/x",
        ]

        for istep in pbar:

            ret = system.flowSteps(output, gammadot)
            assert ret != 0
            inc += output

            if snapshot > 0:
                if inc % snapshot == 0:
                    st = True
                    if "/Flow/snapshot/x" in file:
                        if str(inc) in file["/Flow/snapshot/x"]:
                            st = False
                    if st:
                        i = file["/Flow/snapshot/inc"].size
                        for key in ["/Flow/snapshot/inc"]:
                            file[key].resize((i + 1,))
                        file["/Flow/snapshot/inc"][i] = inc
                        file[f"/Flow/snapshot/x/{inc:d}"] = system.x
                        file[f"/Flow/snapshot/v/{inc:d}"] = system.v
                        file[f"/Flow/snapshot/a/{inc:d}"] = system.a
                        file.flush()

            if output > 0:
                if inc % output == 0:
                    i = int(inc / output)
                    for key in output_fields:
                        if file[key].size <= i:
                            file[key].resize((i + 1,))
                    file["/Flow/output/inc"][i] = inc
                    file["/Flow/output/f_frame"][i] = np.mean(system.f_frame)
                    file["/Flow/output/f_potential"][i] = np.mean(system.f_potential)
                    file["/Flow/output/f_damping"][i] = np.mean(system.f_damping)
                    file["/Flow/output/x"][i] = np.mean(system.x)
                    file.flush()


def cli_plot(cli_args=None):
    """
    Basic plot
    """

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

    parser.add_argument("-m", "--marker", type=str, help="Marker.")
    parser.add_argument("-o", "--output", type=str, help="Store figure.")
    parser.add_argument("file", type=str, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)

    with h5py.File(args.file) as file:

        x_frame = file["/Flow/gammadot"][...] * file["/param/dt"] * file["/Flow/output/inc"][...]
        f_frame = file["/Flow/output/f_frame"][...]
        f_potential = file["/Flow/output/f_potential"][...]

    opts = {}
    if args.marker is not None:
        opts["marker"] = args.marker

    fig, ax = plt.subplots()
    ax.plot(x_frame, f_frame, label=r"$f_\text{frame}$", c="k", **opts)
    ax.plot(x_frame, -f_potential, label=r"$f_\text{potential}$", c="r", **opts)
    ax.set_xlabel(r"$x_\text{frame}$")
    ax.set_ylabel(r"$f$")
    ax.legend()
    if args.output is not None:
        fig.savefig(args.output)
    else:
        plt.show()
    plt.close(fig)
