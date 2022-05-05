"""
??
"""
from __future__ import annotations

import argparse
import inspect
import os
import re
import textwrap

import FrictionQPotSpringBlock  # noqa: F401
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
    cli_ensembleinfo="Flow_ensembleinfo",
    cli_generate="Flow_generate",
    cli_plot="Flow_plot",
    cli_run="Flow_run",
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
        "-w",
        "--time",
        type=str,
        default="72h",
        help="Walltime",
    )
    parser.add_argument(
        "outdir",
        type=str,
        help="Output directory",
    )

    args = tools._parse(parser, cli_args)

    known_gammadot = np.array([1e-2, 1e-1, 1e-0, 1e1])
    known_output = np.array([1e3, 1e3, 1e3, 1e2])
    known_nstep = np.array([1e5, 1e5, 1e5, 1e5])
    if args.output is None:
        args.output = int(np.interp(args.gammadot, known_gammadot, known_output))
    if args.nstep is None:
        args.nstep = int(np.interp(args.gammadot, known_gammadot, known_nstep))

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    files = []

    for i in range(args.start, args.start + args.nsim):

        files += [f"id={i:04d}.h5"]
        seed = i * args.size

        with h5py.File(os.path.join(args.outdir, files[-1]), "w") as file:
            QuasiStatic.generate(
                file=file,
                N=args.size,
                seed=seed,
                eta=args.eta,
                dt=args.dt,
            )
            file["/flow/gammadot"] = args.gammadot
            file["/output/interval"] = args.output
            file["/snapshot/interval"] = args.output * args.snapshot

    executable = entry_points["cli_run"]
    slurm.serial_group(
        [f"{executable} --nstep {args.nstep:d} {file}" for file in files],
        basename=executable,
        group=1,
        outdir=args.outdir,
        sbatch={"time": args.time},
    )


def run_create_extendible(file: h5py.File):
    """
    Create extendible datasets used in :py:func:`cli_run`.
    """

    storage.create_extendible(file, "/output/f_frame", np.float64)
    storage.create_extendible(file, "/output/f_potential", np.float64)
    storage.create_extendible(file, "/output/f_damping", np.float64)
    storage.create_extendible(file, "/output/x", np.float64)
    storage.create_extendible(file, "/output/inc", np.uint32)
    storage.create_extendible(file, "/snapshot/inc", np.uint32)


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
    parser.add_argument("-n", "--nstep", type=int, default=1000, help="#output steps to run")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    pbar = tqdm.tqdm(total=args.nstep, desc=args.file)

    with h5py.File(args.file, "a") as file:

        inc = 0
        snapshot = file["/snapshot/interval"][...]
        output = file["/output/interval"][...]
        gammadot = file["/flow/gammadot"][...]
        assert snapshot % output == 0
        run_create_extendible(file)

        system = QuasiStatic.System(file)
        QuasiStatic.create_check_meta(file, f"/meta/{progname}", dev=args.develop)
        output_fields = ["/output/inc", "/output/f_frame", "/output/f_potential", "/output/x"]

        for istep in range(args.nstep):

            system.chunk_rshift()
            ret = system.flowSteps_boundcheck(output, gammadot)
            if ret == 0:
                raise RuntimeError("Ran out-of-bounds: reduce output interval")
            pbar.n = istep + 1
            pbar.refresh()
            inc += output

            if snapshot > 0:
                if inc % snapshot == 0:
                    i = int(inc / snapshot)
                    for key in ["/snapshot/inc"]:
                        file[key].resize((i + 1,))
                    file["/snapshot/inc"][i] = inc
                    file[f"/snapshot/x/{inc:d}"] = system.x()
                    file[f"/snapshot/v/{inc:d}"] = system.v()
                    file[f"/snapshot/a/{inc:d}"] = system.a()
                    file.flush()

            if output > 0:
                if inc % output == 0:
                    i = int(inc / output)
                    for key in output_fields:
                        file[key].resize((i + 1,))
                    file["/output/inc"][i] = inc
                    file["/output/f_frame"][i] = np.mean(system.f_frame())
                    file["/output/f_potential"][i] = np.mean(system.f_potential())
                    file["/output/f_damping"][i] = np.mean(system.f_damping())
                    file["/output/x"][i] = np.mean(system.x())
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

        x_frame = file["/flow/gammadot"][...] * file["/param/dt"] * file["/output/inc"][...]
        f_frame = file["/output/f_frame"][...]
        f_potential = file["/output/f_potential"][...]

    opts = {}
    if args.marker is not None:
        opts["marker"] = args.marker

    fig, ax = plt.subplots()
    ax.plot(x_frame, f_frame, label=r"$f_\text{frame}$", **opts)
    ax.plot(x_frame, -f_potential, label=r"$f_\text{potential}$", **opts)
    ax.set_xlabel(r"$x_\text{frame}$")
    ax.set_ylabel(r"$f$")
    ax.legend()
    if args.output is not None:
        fig.savefig(args.output)
    else:
        plt.show()
    plt.close(fig)
