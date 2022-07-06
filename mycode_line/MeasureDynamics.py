"""
Rerun step (quasi-static step, or trigger) to extract event map.
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
import textwrap

import FrictionQPotSpringBlock  # noqa: F401
import h5py
import numpy as np

from . import QuasiStatic
from . import tools
from ._version import version


entry_points = dict(
    cli_run="MeasureDynamics",
)


file_defaults = dict(
    cli_run="MeasureDynamics.h5",
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
    Rerun increment and store forces at a fixed time interval.

    Tip: use "--smax" to truncate when (known) S is reached to not waste time on the final stage of
    energy minimisation.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    progname = entry_points[funcname]
    output = file_defaults[funcname]

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("--smax", type=int, help="Truncate at a maximum total S")
    parser.add_argument("-i", "--delta-inc", type=int, help="Increment interval")
    parser.add_argument("-t", "--delta-t", type=float, help="Time interval")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output file")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-s", "--step", required=True, type=int, help="Step number")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    tools._check_overwrite_file(args.output, args.force)

    assert args.delta_inc is not None or args.delta_t is not None
    assert not (args.delta_inc is not None and args.delta_t is not None)

    with h5py.File(args.file, "r") as file, h5py.File(args.output, "w") as output:

        stored = file["/stored"][...]
        assert args.step in stored

        dinc = args.delta_inc
        if dinc is None:
            dinc = int(args.delta_t / file["/param/dt"][...])

        system = QuasiStatic.System(file)

        if args.smax is None:
            args.smax = sys.maxsize

        if "branch" in file:
            system.restore_quasistatic_step(file[f"/branch/{args.step:d}"], 0)  # Trigger
        else:
            assert args.step - 1 in stored
            system.restore_quasistatic_step(file, args.step - 1)  # QuasiStatic

        N = system.N

        fpot = output.create_dataset(
            "f_potential", (1, N), maxshape=(None, N), chunks=(1, N), dtype=np.float64
        )
        fpot[0, :] = system.f_potential

        fframe = output.create_dataset(
            "f_frame", (1, N), maxshape=(None, N), chunks=(1, N), dtype=np.float64
        )
        fframe[0, :] = system.f_frame

        T = [system.t]
        inc_n = system.inc

        dx = file["/event_driven/dx"][...]

        if "branch" in file:
            system.trigger(p=file["/output/p"][args.step], eps=dx, direction=1)
        else:
            system.eventDrivenStep(dx, file["/event_driven/kick"][args.step])

        while True:

            niter = system.minimise(
                nmargin=10,
                niter_tol=min(10, dinc - 1),
                max_iter=dinc - (system.inc - inc_n) % dinc,
                max_iter_is_error=False,
            )

            if niter > 0:
                break

            if niter < 0:
                system.chunk_rshift()

            if (system.inc - inc_n) % dinc == 0:
                i = int((system.inc - inc_n) / dinc)

                fpot.resize((i + 1, N))
                fpot[i, :] = system.f_potential

                fframe.resize((i + 1, N))
                fframe[i, :] = system.f_frame

                for j in range(len(T) - (i + 1)):
                    T.append(-1)

                T.append(system.t)

        output["t"] = np.array(T)

        meta = QuasiStatic.create_check_meta(output, f"/meta/{progname}", dev=args.develop)
        meta.attrs["file"] = args.file
        meta.attrs["step"] = args.step
        meta.attrs["Smax"] = args.smax if args.smax else sys.maxsize
