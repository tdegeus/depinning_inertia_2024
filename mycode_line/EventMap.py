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
import GooseFEM
import GooseHDF5 as g5
import h5py
import numpy as np
import tqdm
import XDMFWrite_h5py as xh

from . import QuasiStatic
from . import tools
from ._version import version


entry_points = dict(
    cli_run="EventMap_run",
    cli_paraview="EventMap_Paraview",
    cli_plot="EventMap_Plot",
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


def cli_run(cli_args=None):
    """
    Rerun a quasistatic step (loaded using QuasiStatic, or triggered using Trigger)
    and store basic event info as follows::

        r: Position of yielding event (block index).
        t: Time of each yielding event (real units).
        S: Size (signed) of the yielding event.

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
    parser.add_argument("--avalanche", action="store_true", help="Truncated once A == N")
    parser.add_argument("-x", action="store_true", help="Store x")
    parser.add_argument("-s", action="store_true", help="Store S")
    parser.add_argument("--smax", type=int, help="Truncate at a maximum total S")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output file")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("--step", required=True, type=int, help="Step number")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    tools._check_overwrite_file(args.output, args.force)
    assert args.x or args.s

    with h5py.File(args.file) as file, h5py.File(args.output, "w") as output:

        system = QuasiStatic.allocate_system(file)

        if args.smax is None:
            args.smax = sys.maxsize

        if "Trigger" in file:
            root = file[f"/Trigger/branches/{args.step:d}"]
            system.restore_quasistatic_step(root, 0)
        else:
            root = file["QuasiStatic"]
            system.restore_quasistatic_step(root, args.step - 1)

        meta = QuasiStatic.create_check_meta(output, f"/meta/{progname}", dev=args.develop)
        meta.attrs["file"] = args.file
        meta.attrs["step"] = args.step
        meta.attrs["Smax"] = args.smax if args.smax else sys.maxsize

        output["x0"] = system.x[system.organisation]
        output["t0"] = system.t
        output["organisation"] = system.organisation

        tref = system.t
        i_n = system.i
        iter = 0
        dx = file["/param/xyield/dx"][...]
        avalanche = True

        if "Trigger" in file:
            system.trigger(p=root["p"][1], eps=dx, direction=1)
        else:
            system.eventDrivenStep(dx, root["kick"][args.step])

        with g5.ExtendableList(output, "r", np.uint64) as dset_r, g5.ExtendableList(
            output, "t", np.float64
        ) as dset_t, g5.ExtendableList(output, "dx", np.float64) as dset_dx, g5.ExtendableList(
            output, "ds", np.int64
        ) as dset_ds:

            while True:

                iter += 1
                i_t = np.copy(system.i)
                x_t = np.copy(system.x)
                ret = system.timeStepsUntilEvent()
                i = system.i
                x = system.x
                t = system.t

                for r in np.argwhere(i != i_t).ravel():
                    dset_r.append(r)
                    dset_t.append(t - tref)
                    if args.s:
                        dset_ds.append(i[r] - i_t[r])
                    if args.x:
                        dset_dx.append(x[r] - x_t[r])

                i_t = np.copy(i)
                x_t = np.copy(x)

                if np.sum(i - i_n) >= args.smax:
                    break

                if ret == 0:
                    break

                if iter % 2000 == 0:
                    file.flush()

                if avalanche:
                    if np.sum(i != i_n) == system.N:
                        output["t_A=N"] = t - tref
                        avalanche = False
                        if args.avalanche:
                            break


def cli_paraview(cli_args=None):
    """
    Convert :py:func:`cli_run` output to be viewed in Paraview.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))

    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, required=True, help="Appended xdmf/h5py")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("--bins", type=int, default=1000, help="Number of time steps to write")
    parser.add_argument("file", type=str, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    tools._check_overwrite_file(args.output, args.force)

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    tools._check_overwrite_file(f"{args.output}.h5", args.force)
    tools._check_overwrite_file(f"{args.output}.xdmf", args.force)

    with h5py.File(args.file) as file, h5py.File(f"{args.output}.h5", "w") as out, xh.TimeSeries(
        f"{args.output}.xdmf"
    ) as xdmf:

        x0 = file["x0"][...]
        t = file["t"][...]
        r = file["r"][...]
        ds = file["ds"][...]
        dx = file["dx"][...]
        organisation = file["organisation"][...]

        assert np.all(organisation.ravel() == np.arange(x0.size))
        assert x0.ndim == 2

        mesh = GooseFEM.Mesh.Quad4.Regular(x0.shape[0] - 1, x0.shape[1] - 1)
        coor = xh.as3d(mesh.coor())
        coor[:, 2] = (x0 - np.mean(x0)).ravel()

        x0 = x0.ravel()

        out["coor"] = coor
        out["conn"] = mesh.conn()

        X = np.zeros(coor.shape[0], dtype=np.float64)
        S = np.zeros(coor.shape[0], dtype=np.int64)

        args.bins = min(args.bins, np.unique(t).size)

        if "t_A=N" in file:
            n = int(args.bins / 2)
            ta = np.linspace(0, file["t_A=N"][...], n + 1)
            tb = np.linspace(file["t_A=N"][...], np.max(t), n + 1)[1:]
            tsave = np.hstack((ta, tb))
        else:
            tsave = np.linspace(0, np.max(t), args.bins + 1)

        for ibin in tqdm.tqdm(range(1, args.bins + 1)):

            keep = np.logical_and(t >= tsave[ibin - 1], t < tsave[ibin])
            np.add.at(X, r[keep], dx[keep])
            np.add.at(S, r[keep], ds[keep])

            disp = np.zeros_like(coor)
            disp[:, -1] = X

            out[f"/S/{ibin:d}"] = S
            out[f"/disp/{ibin:d}"] = disp

            xdmf += xh.TimeStep(time=tsave[ibin])
            xdmf += xh.Unstructured(out["coor"], out["conn"], xh.ElementType.Quadrilateral)
            xdmf += xh.Attribute(out[f"/disp/{ibin:d}"], xh.AttributeCenter.Node, name="dx")
            xdmf += xh.Attribute(out[f"/S/{ibin:d}"], xh.AttributeCenter.Node, name="S")


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

    parser.add_argument("-o", "--output", type=str, help="Store figure.")
    parser.add_argument("file", type=str, help="Event map")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)

    with h5py.File(args.file) as file:

        r = file["r"][...]
        t = file["t"][...]
        ds = file["ds"][...]

    fig, ax = plt.subplots()

    ax.plot(r[ds < 0], t[ds < 0], ".", color="b", rasterized=True, markersize=1)
    ax.plot(r[ds > 0], t[ds > 0], ".", color="k", rasterized=True, markersize=1)

    ax.set_xlabel("$r$")
    ax.set_ylabel("$t$")

    if args.output is not None:
        fig.savefig(args.output)
    else:
        plt.show()

    plt.close(fig)


def cli_basic_output(cli_args=None):
    """
    Collect basic information from :py:func:`cli_run` and combine in a single output file.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    progname = entry_points[funcname]
    output = file_defaults[funcname]

    # developer options
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)

    # output file
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")

    # input files
    parser.add_argument(
        "files", nargs="*", type=str, help=f"Files to read (generate by {entry_points['cli_run']})"
    )

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
            data["S"].append(np.sum(file["ds"][...]))
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
