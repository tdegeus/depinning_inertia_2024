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
import tqdm

from . import QuasiStatic
from . import tools
from ._version import version


entry_points = dict(
    cli_run="EventMap_run",
    cli_basic_output="EventMap_info",
)


file_defaults = dict(
    cli_run="EventMap.h5",
    cli_basic_output="EventMap_info.h5",
)


def replace_ep(doc: str) -> str:
    """
    Replace ``:py:func:`...``` with the relevant entry_point name
    """
    for ep in entry_points:
        doc = doc.replace(rf":py:func:`{ep:s}`", entry_points[ep])
    return doc


def runinc_event_basic(system: QuasiStatic.System, file: h5py.File, step: int, Smax=None) -> dict:
    """
    Rerun increment and get basic event information.

    :param system: The system (modified: increment loaded/rerun).
    :param file: Open simulation HDF5 archive (read-only).
    :param step: The step (or branch) to rerun.
    :param Smax: Stop at given S (to avoid spending time on final energy minimisation).
    :return: A dictionary as follows::

        r: Position of yielding event (block index).
        t: Time of each yielding event (real units).
        S: Size (signed) of the yielding event.
    """

    stored = file["/stored"][...]
    assert step in stored

    if Smax is None:
        Smax = sys.maxsize

    if "branch" in file:
        system.restore_quasistatic_step(file[f"/branch/{step:d}"], 0)  # Trigger
    else:
        assert step - 1 in stored
        system.restore_quasistatic_step(file, step - 1)  # QuasiStatic

    i_n = system.istart + system.i
    dx = file["/event_driven/dx"][...]

    if "branch" in file:
        system.trigger(p=file["/output/p"][step], eps=dx, direction=1)
    else:
        system.eventDrivenStep(dx, file["/event_driven/kick"][step])

    R = []
    T = []
    S = []

    while True:

        if np.any(system.i > system.y.shape[1] - system.nbuffer):
            system.chunk_rshift()

        i_t = system.istart + system.i
        niter = system.timeStepsUntilEvent()
        assert np.all(np.logical_and(system.i > 10, system.i < system.y.shape[1] - 10))
        i = system.istart + system.i
        t = system.t

        for r in np.argwhere(i != i_t).ravel():
            R += [r]
            T += [t * np.ones(r.shape)]
            S += [(i - i_t)[r]]

        i_t = np.copy(i)

        if np.sum(i - i_n) >= Smax:
            break

        if niter == 0:
            break

    ret = dict(r=np.array(R).ravel(), t=np.array(T).ravel(), S=np.array(S).ravel())

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    tools.check_docstring(doc, ret, ":return:")

    return ret


def cli_run(cli_args=None):
    """
    Rerun increment and store basic event info (position and time).
    Tip: truncate when (known) S is reached to not waste time on final stage of energy minimisation.
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
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output file")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-s", "--step", required=True, type=int, help="Step number")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    tools._check_overwrite_file(args.output, args.force)

    with h5py.File(args.file, "r") as file:
        system = QuasiStatic.System(file)
        ret = runinc_event_basic(system, file, args.step, args.smax)

    with h5py.File(args.output, "w") as file:
        file["r"] = ret["r"]
        file["t"] = ret["t"]
        file["S"] = ret["S"]

        meta = QuasiStatic.create_check_meta(file, f"/meta/{progname}", dev=args.develop)
        meta.attrs["file"] = args.file
        meta.attrs["step"] = args.step
        meta.attrs["Smax"] = args.smax if args.smax else sys.maxsize

    if cli_args is not None:
        return ret


def cli_basic_output(cli_args=None):
    """
    Collect basis information from :py:func:`cli_run` and combine in a single output file.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    progname = entry_points[funcname]
    output = file_defaults[funcname]

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Files to read")

    args = tools._parse(parser, cli_args)
    assert len(args.files) > 0
    assert all([os.path.isfile(file) for file in args.files])
    tools._check_overwrite_file(args.output, args.force)

    # collecting data

    data = dict(
        t=[],
        A=[],
        S=[],
        file=[],
        step=[],
        Smax=[],
        version=[],
        dependencies=[],
    )

    executable = entry_points["cli_run"]

    for filepath in tqdm.tqdm(args.files):
        with h5py.File(filepath, "r") as file:
            meta = file[f"/meta/{executable}"]
            data["t"].append(file["t"][...][-1] - file["t"][...][0])
            data["S"].append(np.sum(file["S"][...]))
            data["A"].append(np.unique(file["r"][...]).size)
            data["file"].append(meta.attrs["file"])
            data["step"].append(meta.attrs["step"])
            data["Smax"].append(meta.attrs["Smax"])
            data["version"].append(meta.attrs["version"])
            data["dependencies"].append(meta.attrs["dependencies"])

    # sorting simulation-id and then increment

    sorter = np.lexsort((data["step"], data["file"]))
    for key in data:
        data[key] = [data[key][i] for i in sorter]

    # store (compress where possible)

    with h5py.File(args.output, "w") as file:

        for key in ["t", "A", "S", "step"]:
            file[key] = data[key]

        prefix = os.path.dirname(os.path.commonprefix(data["file"]))
        if data["file"][0].removeprefix(prefix)[0] == "/":
            prefix += "/"
        data["file"] = [i.removeprefix(prefix) for i in data["file"]]
        file["/file/prefix"] = prefix
        tools.h5py_save_unique(data["file"], file, "/file", asstr=True)
        tools.h5py_save_unique(data["version"], file, "/version", asstr=True)
        tools.h5py_save_unique(
            [";".join(i) for i in data["dependencies"]], file, "/dependencies", split=";"
        )

        QuasiStatic.create_check_meta(file, f"/meta/{progname}", dev=args.develop)