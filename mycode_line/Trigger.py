"""
??
"""
from __future__ import annotations

import argparse
import inspect
import itertools
import os
import pathlib
import textwrap

import FrictionQPotSpringBlock  # noqa: F401
import GooseHDF5
import h5py
import numpy as np
import shelephant
import tqdm

from . import Dynamics
from . import EventMap
from . import QuasiStatic
from . import storage
from . import tools
from ._version import version

basename = os.path.splitext(os.path.basename(__file__))[0]

entry_points = dict(
    cli_run="Trigger_Run",
    cli_merge="Trigger_Merge",
    cli_generate="Trigger_Generate",
    cli_ensembleinfo="Trigger_EnsembleInfo",
    cli_job_rerun="Trigger_Job_Rerun",
)

file_defaults = dict(
    cli_ensembleinfo="Trigger_EnsembleInfo.h5",
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
    Trigger all branches stored in a file generated by :py:func:`cli_generate`.
    An option is available to rerun a previous trigger, just to check.
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
    parser.add_argument("--check", type=int, action="append", help="Rerun and check branch(es)")
    parser.add_argument("file", type=str, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    basename = os.path.basename(args.file)

    with h5py.File(args.file, "a" if args.check is None else "r") as file:

        QuasiStatic.create_check_meta(file, f"/meta/{progname}", dev=args.develop)

        branches = np.arange(file["/Trigger/step"].size)

        if args.check is not None:
            assert np.all(np.array(args.check) < branches.size)
            branches = [i for i in args.check]

        dx = file["/param/xyield/dx"][...]
        system = QuasiStatic.allocate_system(file)
        pbar = tqdm.tqdm(branches, desc=f"{basename}: branch = {-1:8d}, p = {-1:8d}, A = {-1:8d}")

        for ibranch in pbar:

            root = file[f"/Trigger/branches/{ibranch:d}"]

            if args.check is None:
                if len(root["completed"]) >= 2:
                    if root["completed"][1]:
                        pbar.set_description(
                            f"{basename}: branch = {ibranch:8d}, p = {root['p'][1]:8d}, A = {root['A'][1]:8d}"
                        )
                        pbar.refresh()
                        continue
            else:
                assert len(root["completed"]) >= 2

            if root["p"][1] > 0 or args.check is not None:
                try_p = np.array([root["p"][1]])
                assert np.all(np.logical_and(try_p >= 0, try_p < system.N))
            else:
                try_p = root["try_p"][1] + np.arange(system.N)
                try_p = np.where(try_p < system.N, try_p, try_p - system.N)

            fastload = False
            if os.path.exists(QuasiStatic.filename2fastload(args.file)):
                fastload = (
                    QuasiStatic.filename2fastload(args.file),
                    f"/QuastiStatic/{file['/Trigger/step'][ibranch]:d}",
                )

            for p in try_p:

                system.restore_quasistatic_step(root, 0, fastload)
                inc = system.inc
                i_n = system.i

                system.trigger(p=p, eps=dx, direction=1)

                while True:
                    ret = system.minimise(time_activity=True)
                    if ret == 0:
                        break
                    system.chunk.align(system.x)

                A = np.sum(system.i != i_n)

                if args.check is not None:
                    assert A >= 0

                if A > 1:
                    break

            pbar.set_description(f"{basename}: branch = {ibranch:8d}, p = {p:8d}, A = {A:8d}")
            pbar.refresh()

            S = np.sum(system.i - i_n)
            T = system.quasistaticActivityLast() - inc
            f_frame = np.mean(system.f_frame)

            if args.check is not None:

                assert root["completed"][1]
                assert root["S"][1] == S
                assert root["A"][1] == A
                assert root["T"][1] == T
                assert root["inc"][1] == system.inc
                assert np.isclose(root["f_frame"][1], f_frame)
                assert np.allclose(root["x"]["1"][...], system.x)

            else:

                root["x"]["1"] = system.x
                storage.dset_extend1d(root, "inc", 1, system.inc)
                storage.dset_extend1d(root, "x_frame", 1, system.x_frame)
                storage.dset_extend1d(root, "completed", 1, True)
                storage.dset_extend1d(root, "p", 1, p)
                storage.dset_extend1d(root, "S", 1, S)
                storage.dset_extend1d(root, "A", 1, A)
                storage.dset_extend1d(root, "T", 1, T)
                storage.dset_extend1d(root, "f_frame", 1, f_frame)

                # prepare the next push
                storage.dset_extend1d(root, "p", 2, -1)
                storage.dset_extend1d(root, "try_p", 2, root["try_p"][1])

                file.flush()


def _to_ranges(mylist):
    mylist = sorted(set(mylist))
    for _, group in itertools.groupby(enumerate(mylist), lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]


def _to_str_ranges(mylist):
    groups = list(_to_ranges(mylist))
    for i in range(len(groups)):
        if groups[i][0] == groups[i][1]:
            groups[i] = str(groups[i][0])
        else:
            groups[i] = str(groups[i][0]) + "-" + str(groups[i][1])
    return groups


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

    fmt = "{:" + str(max(len(i) for i in args.files)) + "s}"
    pbar = tqdm.tqdm([os.path.normpath(i) for i in args.files])
    pbar.set_description(fmt.format(""))

    with h5py.File(args.output, "w") as output:

        QuasiStatic.create_check_meta(output, f"/meta/{progname}", dev=args.develop)

        S = []
        A = []
        T = []
        p = []
        f_frame = []
        f_frame_0 = []
        step = []
        step_c = []
        branch = []
        source = []

        for i, (filename, filepath) in enumerate(zip(pbar, args.files)):

            pbar.set_description(fmt.format(filename), refresh=True)

            with h5py.File(filepath) as file:

                ignore = []

                for ibranch in np.arange(file["/Trigger/step"].size):

                    root = file[f"/Trigger/branches/{ibranch:d}"]

                    if len(root["completed"]) < 2:
                        ignore.append(ibranch)
                        continue

                    if not root["completed"][1]:
                        ignore.append(ibranch)
                        continue

                    S.append(root["S"][1])
                    A.append(root["A"][1])
                    T.append(root["T"][1])
                    p.append(root["p"][1])
                    f_frame.append(root["f_frame"][1])
                    f_frame_0.append(root["f_frame"][0])
                    step.append(file["/Trigger/step"][ibranch])
                    step_c.append(file["/Trigger/step_c"][ibranch])
                    branch.append(ibranch)
                    source.append(filename)

                if i == 0:
                    output["N"] = file["/param/xyield/initseq"].size

                if len(ignore) > 0:
                    print(f"{filepath} ignoring: " + ", ".join(_to_str_ranges(ignore)))

        assert len(source) > 0

        output["S"] = S
        output["A"] = A
        output["T"] = T
        output["p"] = p
        output["f_frame"] = f_frame
        output["f_frame_0"] = f_frame_0
        output["step"] = step
        output["step_c"] = step_c
        output["branch"] = branch
        tools.h5py_save_unique(data=source, file=output, path="source", asstr=True)

        output["S"].attrs["desc"] = "Slip: total number of yield events"
        output["A"].attrs["desc"] = "Spatial extension: total number of yielded particles"
        output["T"].attrs["desc"] = "Duration: time between first and last event"
        output["p"].attrs["desc"] = "Index of the particle that was triggered"
        output["f_frame"].attrs["desc"] = "Frame force (after event)"
        output["f_frame_0"].attrs["desc"] = "Frame force (before triggering)"
        output["step"].attrs["desc"] = "Step at triggering"
        output["step_c"].attrs["desc"] = "Step of last system-spanning event"
        output["branch"].attrs["desc"] = "The branch in the source file"
        output["source"].attrs["desc"] = "Source file (restore by ``value[index]``)"


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
    Branch simulations:

    -   (default) After each system-spanning event.
    -   (--delta-f) At a fixed force increment (of the frame force) after a system-spanning event.

    Per realisation in the output of :py:func:`QuasiStatic.cli_ensembleinfo` create one file
    with a branch ``ibranch`` at the relevant state after each system-spanning event
    (``n`` in total)::

        # Note:
        # -     Fields marked (*) are extendible, with initial size = 1
        # -     Fields marked (**) are extendible, with initial size = 2
        #       They control trigger `i`, and have a size that is 'one ahead'
        # -     Fields marked (***) are reconstructable and there only for fast post-processing
        # -     Unknown items are ``-1`` (or ``0`` for sizes of durations)

        # O.    Meta-data

        /Trigger/step        # per branch, the last quasistatic step it was based on [n]
        /Trigger/step_c      # per branch, the last system-spanning step it was based on [n]
        /Trigger/loaded      # per branch, true is elastic loading was applied to ``step`` [n]

        # I.    Definition of the state

        /Trigger/branches/{ibranch:d}/x/0         # particle positions [N]
        /Trigger/branches/{ibranch:d}/x_frame     # frame position per step (*)
        /Trigger/branches/{ibranch:d}/inc         # increment per step (*)

        # III.  Trigger control parameters

        /Trigger/branches/{ibranch:d}/try_p       # particle to try to trigger first (**)
        /Trigger/branches/{ibranch:d}/p           # triggered particle (**)

        # IV.   Basic output that cannot be reconstructed

        /Trigger/branches/{ibranch:d}/completed   # check that the dynamics finished (*)
        /Trigger/branches/{ibranch:d}/T           # duration of the event (*)

        # V.    Basic output that can be reconstructed from "x"/"x_frame"

        /Trigger/branches/{ibranch:d}/S           # number of times that blocks yielded (***)
        /Trigger/branches/{ibranch:d}/A           # number of blocks that yielded (***)
        /Trigger/branches/{ibranch:d}/f_frame     # frame force (***)
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
    parser.add_argument("ensembleinfo", type=str, help="EnsembleInfo (read-only)")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.ensembleinfo)

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    basedir = pathlib.Path(args.ensembleinfo).parent

    with h5py.File(args.ensembleinfo) as info:

        N = info["/normalisation/N"][...]
        consider = sorted(info["full"])
        files = []
        for filename in consider:
            if np.sum(info["full"][filename]["A"][...] == N) > 0:
                files += [filename]

        assert np.all([(basedir / file).exists() for file in files])
        assert not np.any([(outdir / file).exists() for file in files])
        assert not np.any(
            [os.path.exists(QuasiStatic.filename2fastload(outdir / file)) for file in files]
        )

        executable = entry_points["cli_run"]
        commands = [f"{executable} {file}" for file in files]
        shelephant.yaml.dump(outdir / "commands.yaml", commands)

        for filename in tqdm.tqdm(files):

            with h5py.File(basedir / filename) as source, h5py.File(outdir / filename, "w") as dest:

                path = pathlib.Path(QuasiStatic.filename2fastload(source.filename))
                if path.exists():
                    dfile = pathlib.Path(QuasiStatic.filename2fastload(dest.filename))
                    dfile.symlink_to(os.path.relpath(path, dfile.parent))

                GooseHDF5.copy(source, dest, ["/param", "/meta", "/realisation"])

                system = QuasiStatic.allocate_system(source)

                A = info["full"][filename]["A"][...]
                step = info["full"][filename]["step"][...].astype(np.int64)
                kick = info["full"][filename]["kick"][...]
                x_frame = info["full"][filename]["x_frame"][...]
                f_frame = info["full"][filename]["f_frame"][...]
                steadystate = QuasiStatic.steadystate(x_frame, f_frame, kick, A, N)
                assert np.all(A[~kick] == 0)
                systemspanning = step[np.logical_and(A == N, step > steadystate)]

                QuasiStatic.create_check_meta(dest, f"/meta/{progname}", dev=args.develop)

                meta = dest.create_group("/Trigger/branches")
                meta.attrs["try_p"] = "First particle to try to trigger [ntrigger]"
                meta.attrs["p"] = "Actual triggered particle [ntrigger]"
                meta.attrs["A"] = "Number of yielding particles during event [ntrigger]"
                meta.attrs["S"] = "Total number of yielding events during event [ntrigger]"
                meta.attrs["T"] = "Event duration [ntrigger]"
                meta.attrs["completed"] = "True if trigger was successfully completed [ntrigger]"
                meta.attrs["f_frame"] = "Frame force (after minimisation) [ntrigger]"

                key = "/Trigger/step"
                dset = dest.create_dataset(key, shape=(0,), maxshape=(None,), dtype=np.int64)
                dset.attrs["desc"] = "Quasi-static-load-step on which configuration is based"

                key = "/Trigger/step_c"
                dset = dest.create_dataset(key, shape=(0,), maxshape=(None,), dtype=np.int64)
                dset.attrs["desc"] = "Quasi-static-load-step of last system-spanning event"

                key = "/Trigger/loaded"
                dset = dest.create_dataset(key, shape=(0,), maxshape=(None,), dtype=bool)
                dset.attrs["desc"] = "True if 'elastic' loading was applied to 'step'"

                ibranch = 0

                for start, stop in zip(tqdm.tqdm(systemspanning[:-1]), systemspanning[1:]):

                    if args.delta_f is None:
                        s, f, load = start, f_frame[start], False
                    elif f_frame[start] + args.delta_f > f_frame[stop - 1]:
                        continue
                    else:
                        s, f, load = _get_force_increment(
                            step=step[start:stop],
                            force=f_frame[start:stop],
                            kick=kick[start:stop],
                            target_force=f_frame[start] + args.delta_f,
                        )

                    qsroot = source["QuasiStatic"]
                    x = qsroot["x"][str(s)][...]
                    x_frame = qsroot["x_frame"][s]
                    inc = qsroot["inc"][s]

                    if load:
                        system.restore_quasistatic_step(qsroot, s)
                        system.advanceToFixedForce(f)
                        x = system.x
                        x_frame = system.x_frame
                        assert np.isclose(np.mean(system.f_frame), f)

                    storage.dset_extend1d(dest, "/Trigger/step", ibranch, s)
                    storage.dset_extend1d(dest, "/Trigger/step_c", ibranch, start)
                    storage.dset_extend1d(dest, "/Trigger/loaded", ibranch, load)

                    root = dest.create_group(f"/Trigger/branches/{ibranch:d}")

                    root.create_group("x").create_dataset("0", data=x)
                    root.create_dataset("inc", data=[inc], maxshape=(None,), dtype=np.uint64)
                    root.create_dataset(
                        "x_frame", data=[x_frame], maxshape=(None,), dtype=np.float64
                    )

                    root.create_dataset("try_p", data=[-1, 0], maxshape=(None,), dtype=np.int64)
                    root.create_dataset("p", data=[-1, -1], maxshape=(None,), dtype=np.int64)
                    root.create_dataset("completed", data=[True], maxshape=(None,), dtype=bool)

                    root.create_dataset("f_frame", data=[f], maxshape=(None,), dtype=np.float64)
                    root.create_dataset("A", data=[0], maxshape=(None,), dtype=np.int64)
                    root.create_dataset("S", data=[0], maxshape=(None,), dtype=np.int64)
                    root.create_dataset("T", data=[0], maxshape=(None,), dtype=np.int64)

                    dest.flush()
                    ibranch += 1


def cli_merge(cli_args=None):
    """
    Merge an previous run "push" file (``source``) into a "push" file (``destination``).
    Only pushes that are not present in the ``destination`` file are added.

    There are assertions on:

    -   The parameters.
    -   The position of the frame and of the particles before triggering.
    -   The particle tried for triggering.
    -   the increment (a.k.a. time).
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

    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("source", type=str, help="File to merge")
    parser.add_argument("destination", type=str, help="File where 'source' is merged into")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.source)
    assert os.path.isfile(args.destination)

    with h5py.File(args.source) as src, h5py.File(args.destination, "a") as dest:

        assert "Trigger" in src
        assert "Trigger" in dest
        assert src["/Trigger/step"].size <= dest["/Trigger/step"].size

        test = GooseHDF5.compare(src, dest, GooseHDF5.getdatapaths(src, root="/param"))

        for path in ["/param/xyield/nchunk"]:
            for key in ["!=", "->", "<-"]:
                if path in test[key]:
                    test[key].remove(path)

        assert len(test["!="]) == 0
        assert len(test["->"]) == 0
        assert len(test["<-"]) == 0
        assert len(test["=="]) > 0

        branches = np.arange(src["/Trigger/step"].size)

        for ibranch in tqdm.tqdm(branches):

            sroot = src[f"/Trigger/branches/{ibranch:d}"]
            droot = dest[f"/Trigger/branches/{ibranch:d}"]

            assert sroot["inc"][0] == droot["inc"][0]
            assert sroot["try_p"][1] == droot["try_p"][1]
            assert np.isclose(sroot["x_frame"][0], droot["x_frame"][0])
            assert np.allclose(sroot["x"]["0"][...], droot["x"]["0"][...])

        if f"/meta/{entry_points['cli_run']}" not in dest:
            GooseHDF5.copy(src, dest, f"/meta/{entry_points['cli_run']}")

        for ibranch in tqdm.tqdm(branches):

            sroot = src[f"/Trigger/branches/{ibranch:d}"]
            droot = dest[f"/Trigger/branches/{ibranch:d}"]

            if not sroot["completed"][1]:
                continue

            src.copy(sroot["x"]["1"], droot["x"], "1")
            storage.dset_extend1d(droot, "inc", 1, sroot["inc"][1])
            storage.dset_extend1d(droot, "x_frame", 1, sroot["x_frame"][1])
            storage.dset_extend1d(droot, "completed", 1, sroot["completed"][1])
            storage.dset_extend1d(droot, "p", 1, sroot["p"][1])
            storage.dset_extend1d(droot, "S", 1, sroot["S"][1])
            storage.dset_extend1d(droot, "A", 1, sroot["A"][1])
            storage.dset_extend1d(droot, "T", 1, sroot["T"][1])
            storage.dset_extend1d(droot, "f_frame", 1, sroot["f_frame"][1])
            storage.dset_extend1d(droot, "p", 1, sroot["p"][1])
            storage.dset_extend1d(droot, "p", 2, sroot["p"][2])
            storage.dset_extend1d(droot, "try_p", 2, sroot["try_p"][2])
            dest.flush()


def cli_job_rerun(cli_args=None):
    """
    Create jobs to get event maps, from:

    -   ``--largest-avalanches=n``: the (maximum) ``n`` largest avalanches.
    -   ``--system-spanning=n``: (maximum) ``n`` avalanches.

    In the case of triggering the events are selected based on the distance between the force
    and the mean of the selected bin.
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

    parser.add_argument("--eventmap", action="store_true", help="Create EventMap jobs")
    parser.add_argument("--dynamics", action="store_true", help="Create Dynamics jobs")
    parser.add_argument("--largest-avalanches", type=int, help="n largest avalanches")
    parser.add_argument("--system-spanning", type=int, help="n system spanning events")
    parser.add_argument("--bin", type=int, help="Force bin to choose (Trigger only)")
    parser.add_argument("--bins", type=int, help="Number of force bins (Trigger only)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output YAML file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-w", "--time", type=str, default="24h", help="Walltime")
    parser.add_argument("info", type=str, help="Trigger.EnsembleInfo (read-only)")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.info)
    sourcedir = pathlib.Path(args.info).resolve().parent
    outdir = pathlib.Path(args.output).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.info) as file:

        if f"/meta/{entry_points['cli_ensembleinfo']}" in file:

            N = file["N"][...]
            S = file["S"][...]
            A = file["A"][...]
            step = file["branch"][...]
            source = np.array(tools.h5py_read_unique(file, "/source", asstr=True))
            f = file["f_frame_0"][...]
            trigger = True

            bin_edges = np.linspace(f.min(), f.max(), args.bins + 1)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            bin_idx = np.digitize(f, bin_edges) - 1
            target = bin_centers[args.bin]

            S = S[bin_idx == args.bin]
            A = A[bin_idx == args.bin]
            step = step[bin_idx == args.bin]
            source = source[bin_idx == args.bin]
            f = f[bin_idx == args.bin]

            sorter = np.argsort(np.abs(f - target))

            S = S[sorter]
            A = A[sorter]
            step = step[sorter]
            source = source[sorter]

        elif f"/meta/{QuasiStatic.entry_points['cli_ensembleinfo']}" in file or "avalanche" in file:

            N = file["/normalisation/N"][...]
            S = file["/avalanche/S"][...]
            A = file["/avalanche/A"][...]
            step = file["/avalanche/step"][...]
            source = np.array(file["/files"].asstr()[...])[file["/avalanche/file"]]
            trigger = False

        else:

            raise ValueError("Not a Trigger.EnsembleInfo or QuasiStatic.EnsembleInfo file")

    if args.largest_avalanches is not None:

        assert args.system_spanning is None
        n = args.largest_avalanches
        assert n > 0

        keep = A < N
        S = S[keep]
        A = A[keep]
        step = step[keep]
        source = source[keep]

        sorter = np.argsort(A)[::-1]
        S = S[sorter][:n]
        A = A[sorter][:n]
        step = step[sorter][:n]
        source = source[sorter][:n]

    elif args.system_spanning is not None:

        assert args.largest_avalanches is None
        n = args.system_spanning
        assert n > 0

        keep = A == N
        S = S[keep][:n]
        A = A[keep][:n]
        step = step[keep][:n]
        source = source[keep][:n]

    if args.eventmap:
        assert not args.dynamics
        excecutable = EventMap.entry_points["cli_run"]
    elif args.dynamics:
        excecutable = Dynamics.entry_points["cli_run"]
    else:
        raise ValueError("No job type specified")

    commands = []

    for i in range(A.size):

        src = os.path.relpath(sourcedir / source[i], outdir)

        if trigger:
            out = f"{os.path.splitext(source[i])[0].replace('/', '_')}_branch={step[i]}.h5"
            opts = f"--output={out} --branch={step[i]} --step=1"
        else:
            out = f"{os.path.splitext(source[i])[0].replace('/', '_')}_step={step[i]}.h5"
            opts = f"--output={out} --step={step[i]}"

        if args.eventmap:
            opts += f" --smax={S[i]}"

        commands.append(f"{excecutable} {opts} {src}")

    shelephant.yaml.dump(args.output, commands)
