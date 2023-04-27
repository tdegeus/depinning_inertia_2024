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
from . import tag
from . import tools
from ._version import version

basename = os.path.splitext(os.path.basename(__file__))[0]

entry_points = dict(
    cli_ensemblepack="Flow_EnsemblePack",
    cli_generate="Flow_Generate",
    cli_plot="Flow_Plot",
    cli_run="Flow_Run",
)


file_defaults = dict(
    cli_ensemblepack="Flow_EnsemblePack.h5",
)

data_version = "2.1"
assert tag.greater_equal(data_version, QuasiStatic.data_version)


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
        if key in ["v_frame", "jump"]:
            info[key] = float(info[key])
        else:
            info[key] = int(info[key])

    return info


def ensemble_average(file: h5py.File | dict, skip: int = None):
    """
    Ensemble average from a file written by :py:func:`cli_enembleinfo`.

    :param file: Ensemble info (opened HDF5 archive).
    :param skip: Skip the first ``skip`` entries.
    """

    data = {
        "v": defaultdict(list),
        "f_frame": defaultdict(list),
        "f_potential": defaultdict(list),
    }

    root = file["/full"]

    for name in root:
        v_frame = float(root[name]["v_frame"][...])
        v = root[name]["v"][...]
        f_frame = root[name]["f_frame"][...]
        f_potential = root[name]["f_potential"][...]

        if skip is None:
            s = int(v.size / 2)
        else:
            s = skip

        data["v"][v_frame] += list(v[s:])
        data["f_frame"][v_frame] += list(f_frame[s:])
        data["f_potential"][v_frame] += list(f_potential[s:])

    mean = {}
    std = {}
    v_frame = np.array(sorted(data["v"].keys()))

    for field in data:
        mean[field] = np.array([np.mean(data[field][v]) for v in v_frame])
        std[field] = np.array([np.std(data[field][v]) for v in v_frame])

    return v_frame, mean, std, data


def cli_ensemblepack(cli_args=None):
    """
    Extract output all from a set of files run with :py:func:`cli_run`.
    After this the run-files can be deleted (only destroys the possibility to continue the run).
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

    parser.add_argument("-i", "--inplace", action="store_true", help="Update output file inplace")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("files", nargs="*", type=str, help="Input files")

    args = tools._parse(parser, cli_args)
    assert all([os.path.isfile(file) for file in args.files])
    if args.inplace:
        assert os.path.isfile(args.output)
        mode = "a"
    else:
        tools._check_overwrite_file(args.output, args.force)
        mode = "w"

    with h5py.File(args.output, mode) as output:
        QuasiStatic.create_check_meta(output, f"/meta/{progname}", dev=args.develop)
        if "full" not in output:
            output.create_group("full")
        if "param" in output:
            assert QuasiStatic._get_data_version(output) == data_version

        for filepath in args.files:
            fname = os.path.relpath(filepath, os.path.dirname(args.output))
            fname = fname.replace("/", "_")

        for i, filepath in enumerate(tqdm.tqdm(args.files)):
            fname = os.path.relpath(filepath, os.path.dirname(args.output))
            fname = fname.replace("/", "_")
            with h5py.File(filepath) as file:
                if i == 0 and "param" not in output:
                    g5.copy(file, output, "/param")

                if i == 0:
                    norm = QuasiStatic.Normalisation(file).asdict()
                else:
                    QuasiStatic._check_normalisation(norm, QuasiStatic.Normalisation(file).asdict())

                assert QuasiStatic._get_data_version(file) == data_version

                root = file["/Flow/output"]
                if fname in output["full"]:
                    out = output[f"/full/{fname}"]
                    assert np.allclose(out["u_frame"], root["u_frame"][...])
                    assert np.allclose(out["f_frame"], root["f_frame"][...])
                    assert np.allclose(out["f_potential"], root["f_potential"][...])
                    assert np.allclose(out["u"], root["u"][...])
                    assert np.allclose(out["v"], root["v"][...])
                    assert np.allclose(out["v_frame"], file["/Flow/param/v_frame"][...])
                    assert out["realisation"]["seed"][...] == file["realisation"]["seed"][...]
                else:
                    out = output.create_group(f"/full/{fname}")
                    out["u_frame"] = root["u_frame"][...]
                    out["f_frame"] = root["f_frame"][...]
                    out["f_potential"] = root["f_potential"][...]
                    out["u"] = root["u"][...]
                    out["v"] = root["v"][...]
                    out["v_frame"] = file["/Flow/param/v_frame"][...]
                    g5.copy(file, output, "/realisation", root=f"/full/{fname}")


def cli_generate(cli_args=None):
    """
    Generate IO files.
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

    QuasiStatic._generate_cli_options(parser)

    parser.add_argument(
        "--output",
        type=lambda arg: float(arg),
        default=50,
        help="delta(u_frame) to leave between output steps",
    )
    parser.add_argument(
        "--restart",
        type=lambda arg: int(float(arg)),
        default=200,
        help="Snapshot for restart every n output steps.",
    )
    parser.add_argument("--v-frame", type=float, required=True, help="Driving rate.")
    args = tools._parse(parser, cli_args)

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    opts = QuasiStatic._generate_parse(args)

    n = args.size if args.shape is None else np.prod(args.shape)
    assert not any(
        [(outdir / f"id={i:04d}.h5").exists() for i in range(args.start, args.start + args.nsim)]
    )
    files = []
    for i in range(args.start, args.start + args.nsim):
        files += [f"id={i:04d}.h5"]
        seed = i * n
        with h5py.File(outdir / files[-1], "w") as file:
            QuasiStatic.generate(file=file, seed=seed, **opts)
            file["/param/data_version"][...] = data_version
            dt = file["/param/dt"][...]
            output = int(args.output / (args.v_frame * dt))
            file["/Flow/param/v_frame"] = args.v_frame
            file["/Flow/output/interval"] = output
            file["/Flow/restart/interval"] = output * args.restart

    executable = f'{entry_points["cli_run"]} --nstep {args.nstep:d}'
    commands = [f"{executable} {file}" for file in files]
    shelephant.yaml.dump(outdir / "commands_run.yaml", commands, force=True)


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
        "--init-force",
        type=float,
        default=0.5,
        help="At initialisation: move the load frame at once to this force.",
    )
    parser.add_argument(
        "-n", "--nstep", type=lambda arg: int(float(arg)), default=1000, help="#output steps to run"
    )
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)

    with h5py.File(args.file, "a") as file:
        assert QuasiStatic._get_data_version(file) == data_version
        root = file["/Flow/output"]
        res = file["/Flow/restart"]
        output = root["interval"][...]
        restart = file["/Flow/restart/interval"][...]
        v_frame = file["/Flow/param/v_frame"][...]
        assert restart % output == 0

        system = QuasiStatic.allocate_system(file)
        system.inc = 0
        QuasiStatic.create_check_meta(file, f"/meta/{progname}", dev=args.develop)

        if "u" in res:
            system.inc = res["inc"][...]
            i = int(system.inc / output)
            system.chunk.restore(
                state=res["state"][...], value=res["value"][...], index=res["index"][...]
            )
            system.u_frame = root["u_frame"][i]
            system.u = res["u"][...]
            system.v = res["v"][...]
            system.a = res["a"][...]
            print(f"Restarting at u_frame = {system.u_frame:.1f}")
            assert np.isclose(root["f_frame"][i], np.mean(system.f_frame))
            assert np.isclose(root["f_potential"][i], np.mean(system.f_potential))
            assert np.isclose(root["u"][i], np.mean(system.u))
            assert np.isclose(root["v"][i], np.mean(system.v))
        else:
            # start the system with a bit of advance
            system.u_frame = (args.init_force - np.mean(system.f_frame)) / system.k_frame
            system.v = v_frame * np.ones_like(system.v)
            # create/check output
            if "f_frame" not in root:
                fpot = np.mean(system.f_potential)
                root.create_dataset("u_frame", data=[system.u_frame], maxshape=(None,))
                root.create_dataset("f_frame", data=[np.mean(system.f_frame)], maxshape=(None,))
                root.create_dataset("f_potential", data=[fpot], maxshape=(None,))
                root.create_dataset("u", data=[np.mean(system.u)], maxshape=(None,))
                root.create_dataset("v", data=[np.mean(system.v)], maxshape=(None,))
                root.create_dataset("inc", data=[system.inc], maxshape=(None,))
            else:
                assert np.isclose(root["u_frame"][0], np.mean(system.u_frame))
                assert np.isclose(root["f_frame"][0], np.mean(system.f_frame))
                assert np.isclose(root["f_potential"][0], np.mean(system.f_potential))
                assert np.isclose(root["u"][0], np.mean(system.u))
                assert np.isclose(root["v"][0], np.mean(system.v))
                assert root["inc"][0] == system.inc

        for _ in tqdm.tqdm(range(args.nstep), desc=args.file):
            system.flowSteps(output, v_frame)

            if restart > 0:
                if system.inc % restart == 0:
                    storage.dump_overwrite(res, "inc", system.inc)
                    storage.dump_overwrite(res, "u", system.u)
                    storage.dump_overwrite(res, "v", system.v)
                    storage.dump_overwrite(res, "a", system.a)
                    storage.dump_overwrite(res, "state", system.chunk.state_at(system.chunk.start))
                    storage.dump_overwrite(res, "index", system.chunk.start)
                    storage.dump_overwrite(res, "value", system.chunk.data[:, 0])
                    file.flush()

            if output > 0:
                if system.inc % output == 0:
                    i = int(system.inc / output)
                    for key in ["inc", "u_frame", "f_frame", "f_potential", "u", "v"]:
                        if root[key].size <= i:
                            root[key].resize((i + 1,))
                    root["inc"][i] = system.inc
                    root["u_frame"][i] = system.u_frame
                    root["f_frame"][i] = np.mean(system.f_frame)
                    root["f_potential"][i] = np.mean(system.f_potential)
                    root["u"][i] = np.mean(system.u)
                    root["v"][i] = np.mean(system.v)
                    file.flush()


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

    parser.add_argument("-m", "--marker", type=str, help="Marker.")
    parser.add_argument("-o", "--output", type=str, help="Store figure.")
    parser.add_argument("-p", "--path", type=str, help="'/full/{path}' (EnsembleInfo).")
    parser.add_argument("file", type=str, help="Simulation file / EnsembleInfo")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)

    with h5py.File(args.file) as file:
        if "full" in file:
            v_frame, mean, std, _ = ensemble_average(file)
            ensemble = {
                "x": v_frame[...],
                "y": mean["f_frame"][...],
                "xerr": std["v"][...],
                "yerr": std["f_frame"][...],
            }
            root = file["full"][args.path]
            v_frame = root["v_frame"][...]
        else:
            ensemble = None
            root = file["/Flow/output"]
            v_frame = file["/Flow/param/v_frame"][...]

        u_frame = root["u_frame"][...]
        f_frame = root["f_frame"][...]
        f_potential = root["f_potential"][...]
        v = root["v"][...]

    opts = {}
    if args.marker is not None:
        opts["marker"] = args.marker

    fig, axes = gplt.subplots(ncols=2 if ensemble is None else 3)

    ax = axes[0]
    ax.plot(u_frame, f_frame, label=r"$f_\text{frame}$", c="k", **opts)
    ax.plot(u_frame, -f_potential, label=r"$f_\text{potential}$", c="r", **opts)
    ax.set_xlabel(r"$u_\text{frame}$")
    ax.set_ylabel(r"$f$")
    ax.legend()

    ax = axes[1]
    ax.plot(u_frame, v / v_frame, c="k", **opts)
    ax.set_xlabel(r"$u_\text{frame}$")
    ax.set_ylabel(r"$v / v_\text{frame}$")

    if ensemble is not None:
        ax = axes[2]
        ax.errorbar(**ensemble, c="k", ls="none", marker="o", lw=1)
        ax.set_xlabel(r"$v_\text{frame}$")
        ax.set_ylabel(r"$f_\text{frame}$")

    if args.output is not None:
        fig.savefig(args.output)
    else:
        plt.show()
    plt.close(fig)
