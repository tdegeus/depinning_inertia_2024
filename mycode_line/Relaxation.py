"""
Rerun step (quasi-static step, or trigger) to extract the dynamic evolution of fields.
"""
from __future__ import annotations

import argparse
import inspect
import os
import textwrap

import enstat
import FrictionQPotSpringBlock  # noqa: F401
import GooseHDF5
import h5py
import numpy as np
import tqdm

from . import Dynamics
from . import QuasiStatic
from . import tools
from ._version import version

entry_points = dict(
    cli_run="Relaxation_Run",
    cli_ensembleinfo="Relaxation_EnsembleInfo",
)

file_defaults = dict(
    cli_ensembleinfo="Relaxation_EnsembleInfo.h5",
)

data_version = "2.0"


def replace_ep(doc: str) -> str:
    """
    Replace ``:py:func:`...``` with the relevant entry_point name
    """
    for ep in entry_points:
        doc = doc.replace(rf":py:func:`{ep:s}`", entry_points[ep])
    return doc


def cli_run(cli_args=None):
    """
    Rerun an system-spanning event and store average output from the moment that the event spans
    the system.
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

    # developer options
    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-v", "--version", action="version", version=version)

    # input selection
    parser.add_argument("--step", type=int, help="Step to run (state: step - 1, then trigger)")
    parser.add_argument("--branch", type=int, help="Branch (if 'Trigger')")

    # output selection
    parser.add_argument("--t-step", type=int, default=1, help="Save every t-step")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file")

    # input files
    parser.add_argument("file", type=str, help="Simulation from which to run (read-only)")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    assert os.path.abspath(args.file) != os.path.abspath(args.output)
    tools._check_overwrite_file(args.output, args.force)

    # basic assertions
    with h5py.File(args.file) as src:
        if args.branch is not None:
            assert f"/Trigger/branches/{args.branch:d}/u/{args.step - 1:d}" in src
        else:
            assert f"/QuasiStatic/u/{args.step - 1:d}" in src

    with h5py.File(args.output, "w") as file:
        with h5py.File(args.file) as src:
            GooseHDF5.copy(src, file, ["/param", "/meta", "/realisation"])

        meta = QuasiStatic.create_check_meta(file, f"/meta/{progname}", dev=args.develop)
        meta.attrs["file"] = os.path.basename(args.file)
        meta.attrs["step"] = args.step
        meta.attrs["t-step"] = args.t_step

        file.create_group("Relaxation")
        file.flush()

        system, info = Dynamics.restore_system(
            filepath=args.file, step=args.step, branch=args.branch, apply_trigger=True
        )

        if args.branch is not None:
            meta.attrs["branch"] = args.branch
            meta.attrs["p"] = info["p"]

        # rerun dynamics and store every other time

        pbar = tqdm.tqdm(total=info["duration"])
        pbar.set_description(args.output)
        inc_n = system.inc

        systemspanning = False

        while not systemspanning:
            system.timeStepsUntilEvent()
            systemspanning = np.all(np.not_equal(system.chunk.index_at_align, info["i_n"]))
            pbar.n = system.inc - inc_n
            pbar.update()

        with GooseHDF5.ExtendableList(file, "v", np.float64) as ret_v, GooseHDF5.ExtendableList(
            file, "f_frame", np.float64
        ) as ret_fext, GooseHDF5.ExtendableList(file, "f_potential", np.float64) as ret_fpot:
            while system.inc - inc_n < info["duration"]:
                ret_v.append(np.mean(system.v))
                ret_fext.append(np.mean(system.f_frame))
                ret_fpot.append(np.mean(system.f_potential))
                pbar.n = system.inc - inc_n
                pbar.update()
                system.timeSteps(args.t_step)

        meta.attrs["completed"] = 1


def cli_ensembleinfo(cli_args=None):
    """
    Bin the data from several run.
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
    output = file_defaults[funcname]

    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("--bins", type=int, default=101, help="Number of bins")
    parser.add_argument("--vmin", type=float, default=0, help="Range of v: min(v):max(b):bins")
    parser.add_argument("--vmax", type=float, default=4, help="Range of v: min(v):max(b):bins")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument(
        "files", nargs="*", type=str, help=replace_ep("Output files of :py:func:`cli_run`")
    )

    args = tools._parse(parser, cli_args)
    assert all([os.path.isfile(f) for f in args.files])
    tools._check_overwrite_file(args.output, args.force)

    bin_edges = np.linspace(args.vmin, args.vmax, args.bins)
    binned = enstat.binned(bin_edges, names=["v", "f_potential", "f_frame"], bound_error="ignore")

    with h5py.File(args.output, "w") as output:
        for i, filepath in enumerate(tqdm.tqdm(args.files)):
            with h5py.File(filepath) as file:
                binned.add_sample(
                    v=file["v"][...],
                    f_potential=-file["f_potential"][...],
                    f_frame=file["f_frame"][...],
                )
                if i == 0:
                    GooseHDF5.copy(file, output, ["/param"])

        output["/data/v/first"] = binned["v"].first
        output["/data/v/second"] = binned["v"].second
        output["/data/v/norm"] = binned["v"].norm

        output["/data/f_potential/first"] = binned["f_potential"].first
        output["/data/f_potential/second"] = binned["f_potential"].second
        output["/data/f_potential/norm"] = binned["f_potential"].norm

        output["/data/f_frame/first"] = binned["f_frame"].first
        output["/data/f_frame/second"] = binned["f_frame"].second
        output["/data/f_frame/norm"] = binned["f_frame"].norm

        output["/source/files"] = sorted(args.files)
