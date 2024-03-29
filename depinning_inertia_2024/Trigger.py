"""
Branch quasistatic simulations at different forces and trigger events.
See :py:mod:`depinning_inertia_2024.QuasiStatic`.
"""

from __future__ import annotations

import argparse
import itertools
import os
import pathlib
import shutil
import tempfile
import textwrap

import FrictionQPotSpringBlock  # noqa: F401
import GooseFEM
import GooseHDF5
import h5py
import numpy as np
import shelephant
import tqdm
import XDMFWrite_h5py as xh

from . import QuasiStatic
from . import storage
from . import tag
from . import tools
from ._version import version

basename = os.path.splitext(os.path.basename(__file__))[0]


file_defaults = dict(
    EnsembleInfo="Trigger_EnsembleInfo.h5",
)

data_version = "2.1"
assert tag.greater_equal(data_version, QuasiStatic.data_version)


def _update_data_version(file: h5py.File) -> None:
    """
    Update data version in file
    """

    ver = QuasiStatic._get_data_version(file)

    if ver == data_version:
        return False

    if "/param/data_version" not in file:
        file["/param/data_version"] = data_version
    else:
        file["/param/data_version"][...] = data_version

    if tag.less(ver, "2.1"):
        if "/Trigger/branches" in file:
            for ibranch in np.arange(file["/Trigger/step"].size):
                root = file[f"/Trigger/branches/{ibranch:d}"]
                data = [False] * root["completed"].size
                root.create_dataset("truncated", data=data, maxshape=(None,), dtype=bool)

    return True


def _UpdateData_parser():
    parser = argparse.ArgumentParser(
        formatter_class=QuasiStatic.MyFmt, description=textwrap.dedent(UpdateData.__doc__)
    )
    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("--no-bak", action="store_true", help="Do not backup before modifying")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Simulation files")
    return parser


def UpdateData(cli_args=None):
    """
    Update the data from any version to the current version.
    """
    parser = _UpdateData_parser()
    args = tools._parse(parser, cli_args)

    assert all([os.path.isfile(f) for f in args.files])
    files = []
    for filename in tqdm.tqdm(args.files, desc="Reading version"):
        with h5py.File(filename) as file:
            if QuasiStatic._get_data_version(file) != data_version:
                files.append(filename)

    if len(files) == 0:
        return

    if not args.no_bak:
        assert not any([os.path.isfile(f + ".bak") for f in files])
        for filename in files:
            shutil.copy2(filename, filename + ".bak")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for filename in tqdm.tqdm(files, desc="Updating data"):
            shutil.copy2(filename, tmp / "my.h5")
            with h5py.File(tmp / "my.h5", "a") as file:
                _update_data_version(file)
            shutil.copy2(tmp / "my.h5", filename)


def _CheckData_parser():
    return QuasiStatic._CheckData_parser()


def CheckData(cli_args=None):
    """
    Check the data file for data version.
    Prints the files that have failed. No output is written if all files are ok.
    """
    return QuasiStatic.CheckData(cli_args, data_version)


def _FilterCompleted_parser():
    parser = argparse.ArgumentParser(
        formatter_class=QuasiStatic.MyFmt, description=textwrap.dedent(FilterCompleted.__doc__)
    )
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("--sep", type=str, help="Separator in output", default=" ")
    parser.add_argument("files", nargs="*", type=str, help="Simulation files")
    return parser


def FilterCompleted(cli_args=None):
    """
    Filter completed files from list of files.
    Printed are just the files that are not completed.
    """
    parser = _FilterCompleted_parser()
    args = tools._parse(parser, cli_args)
    assert np.all([os.path.isfile(f) for f in args.files])
    ret = []

    for filepath in args.files:
        with h5py.File(filepath) as file:
            branches = np.arange(file["/Trigger/step"].size)
            completed = file["/Trigger/step"].size == 0

            for ibranch in branches:
                root = file[f"/Trigger/branches/{ibranch:d}"]

                if len(root["completed"]) >= 2:
                    if root["completed"][1]:
                        completed = True
                        break

            if not completed:
                ret.append(filepath)

    print(args.sep.join(ret))


def _Run_parser():
    parser = argparse.ArgumentParser(
        formatter_class=QuasiStatic.MyFmt, description=textwrap.dedent(Run.__doc__)
    )
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("--check", type=int, action="append", help="Rerun and check branch(es)")
    parser.add_argument(
        "--truncate-system-spanning", action="store_true", help="Truncate as soon as A == N"
    )
    parser.add_argument("file", type=str, help="Simulation file")
    return parser


def Run(cli_args=None):
    """
    Trigger all branches stored in a file.
    The file has to be generated by :py:func:`depinning_inertia_2024.Trigger.Generate`.
    An option is available to rerun a previous trigger, just to check.
    """
    parser = _Run_parser()
    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    basename = os.path.basename(args.file)

    with h5py.File(args.file, "a" if args.check is None else "r") as file:
        QuasiStatic.create_check_meta(file, "/meta/Trigger_Run", dev=args.develop)
        _update_data_version(file)
        branches = np.arange(file["/Trigger/step"].size)

        if args.check is not None:
            assert np.all(np.array(args.check) < branches.size)
            branches = [i for i in args.check]

        du = file["/param/potentials/du"][...]
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
                assert np.all(np.logical_and(try_p >= 0, try_p < system.size))
            else:
                try_p = root["try_p"][1] + np.arange(system.size)
                try_p = np.where(try_p < system.size, try_p, try_p - system.size)

            fastload = False
            if os.path.exists(QuasiStatic.filename2fastload(args.file)):
                fastload = (
                    QuasiStatic.filename2fastload(args.file),
                    f"/QuastiStatic/{file['/Trigger/step'][ibranch]:d}",
                )

            for p in try_p:
                system.restore_quasistatic_step(root, 0, fastload)
                inc = system.inc
                i_n = np.copy(system.chunk.index_at_align)

                system.trigger(p=p, eps=du, direction=1)

                if args.truncate_system_spanning:
                    ret = system.minimise_truncate(i_n=i_n, A_truncate=system.size)
                else:
                    ret = system.minimise(time_activity=True)

                A = np.sum(system.chunk.index_at_align != i_n)

                if args.check is not None:
                    assert A >= 0

                if A > 1:
                    break

            pbar.set_description(f"{basename}: branch = {ibranch:8d}, p = {p:8d}, A = {A:8d}")
            pbar.refresh()

            S = np.sum(system.chunk.index_at_align - i_n)
            T = system.quasistaticActivityLast - inc
            f_frame = np.mean(system.f_frame)

            if args.check is not None:
                assert root["completed"][1]
                assert root["truncated"][1] == (ret > 0)
                assert root["S"][1] == S
                assert root["A"][1] == A
                assert root["T"][1] == T
                assert root["inc"][1] == system.inc
                assert np.isclose(root["f_frame"][1], f_frame)
                assert np.allclose(root["u"]["1"][...], system.u)

            else:
                root["u"]["1"] = system.u
                storage.dset_extend1d(root, "inc", 1, system.inc)
                storage.dset_extend1d(root, "u_frame", 1, system.u_frame)
                storage.dset_extend1d(root, "completed", 1, True)
                storage.dset_extend1d(root, "truncated", 1, ret > 0)
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


def _EnsembleInfo_parser():
    parser = argparse.ArgumentParser(
        formatter_class=QuasiStatic.MyFmt, description=textwrap.dedent(EnsembleInfo.__doc__)
    )
    output = file_defaults["EnsembleInfo"]
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Files to read")
    return parser


def EnsembleInfo(cli_args=None):
    """
    Read information (avalanche size, force) of an ensemble.
    See :py:func:`depinning_inertia_2024.Trigger.basic_output`.
    Store into a single output file.
    """
    parser = _EnsembleInfo_parser()
    args = tools._parse(parser, cli_args)
    assert len(args.files) > 0
    assert all([os.path.isfile(file) for file in args.files])
    tools._check_overwrite_file(args.output, args.force)

    fmt = "{:" + str(max(len(i) for i in args.files)) + "s}"
    pbar = tqdm.tqdm([os.path.normpath(i) for i in args.files])
    pbar.set_description(fmt.format(""))

    with h5py.File(args.output, "w") as output:
        QuasiStatic.create_check_meta(output, "/meta/Trigger_EnsembleInfo", dev=args.develop)

        S = []
        A = []
        T = []
        truncated = []
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
                ver = QuasiStatic._get_data_version(file)
                ignore = []

                for ibranch in np.arange(file["/Trigger/step"].size):
                    root = file[f"/Trigger/branches/{ibranch:d}"]

                    if "completed" not in root:
                        ignore.append(ibranch)
                        continue

                    if len(root["completed"]) < 2:
                        ignore.append(ibranch)
                        continue

                    if not root["completed"][1]:
                        ignore.append(ibranch)
                        continue

                    if tag.less(ver, "2.1"):
                        truncated.append(root["truncated"][1])
                    else:
                        truncated.append(False)

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
                    GooseHDF5.copy(file, output, "/param")
                    norm = QuasiStatic.Normalisation(file).asdict()
                    for key, value in norm.items():
                        output[f"/normalisation/{key}"] = value

                if len(ignore) > 0:
                    print(f"{filepath} ignoring: " + ", ".join(_to_str_ranges(ignore)))

        assert len(source) > 0

        output["S"] = S
        output["A"] = A
        output["T"] = T
        output["p"] = p
        output["truncated"] = truncated
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
        output["truncated"].attrs["desc"] = "True if the trigger was truncated at A == N"
        output["f_frame"].attrs["desc"] = "Frame force (after event)"
        output["f_frame_0"].attrs["desc"] = "Frame force (before triggering)"
        output["step"].attrs["desc"] = "Step at triggering"
        output["step_c"].attrs["desc"] = "Step of last system spanning event"
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


def _Generate_parser():
    parser = argparse.ArgumentParser(
        formatter_class=QuasiStatic.MyFmt, description=textwrap.dedent(Generate.__doc__)
    )
    parser.add_argument("--delta-f", type=float, help="Advance to fixed force")
    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-o", "--outdir", type=str, default=".", help="Output directory")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument(
        "ensembleinfo",
        type=str,
        help="EnsembleInfo from :py:func:depinning_inertia_2024.QuasiStatic.EnsembleInfo` (read-only)",
    )
    return parser


def Generate(cli_args=None):
    """
    Branch simulations at different forces.
    The simulations are the result of :py:func:`depinning_inertia_2024.QuasiStatic.Run`.

    The following options are available:

    -   (default) After each system spanning event.
    -   (--delta-f) At a fixed force increment (of the frame force) after a system spanning event.

    Per realisation in the output of :py:func:`depinning_inertia_2024.QuasiStatic.EnsembleInfo`
    create one file with a branch ``ibranch`` at the relevant state after each system spanning event
    (``n`` in total)::

        # Note:
        # -     Fields marked (*) are extendible, with initial size = 1
        # -     Fields marked (**) are extendible, with initial size = 2
        #       They control trigger `i`, and have a size that is 'one ahead'
        # -     Fields marked (***) are reconstructable and there only for fast post-processing
        # -     Unknown items are ``-1`` (or ``0`` for sizes of durations)

        # O.    Meta-data

        /Trigger/step        # per branch, the last quasistatic step it was based on [n]
        /Trigger/step_c      # per branch, the last system spanning step it was based on [n]
        /Trigger/loaded      # per branch, true is elastic loading was applied to ``step`` [n]

        # I.    Definition of the state

        /Trigger/branches/{ibranch:d}/u/0         # particle positions [shape]
        /Trigger/branches/{ibranch:d}/u_frame     # frame position per step (*)
        /Trigger/branches/{ibranch:d}/inc         # increment per step (*)

        # III.  Trigger control parameters

        /Trigger/branches/{ibranch:d}/try_p       # particle to try to trigger first (**)
        /Trigger/branches/{ibranch:d}/p           # triggered particle (**)

        # IV.   Basic output that cannot be reconstructed

        /Trigger/branches/{ibranch:d}/completed   # check that the dynamics finished (*)
        /Trigger/branches/{ibranch:d}/truncated   # if event was stopped at A == N (*)
        /Trigger/branches/{ibranch:d}/T           # duration of the event (*)

        # V.    Basic output that can be reconstructed from "u"/"u_frame"

        /Trigger/branches/{ibranch:d}/S           # number of times that blocks yielded (***)
        /Trigger/branches/{ibranch:d}/A           # number of blocks that yielded (***)
        /Trigger/branches/{ibranch:d}/f_frame     # frame force (***)
    """
    parser = _Generate_parser()
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

        executable = "Trigger_Run"
        commands = [f"{executable} {file}" for file in files]
        shelephant.yaml.dump(outdir / "commands_run.yaml", commands, force=True)

        for filename in tqdm.tqdm(files):
            with h5py.File(basedir / filename) as source, h5py.File(outdir / filename, "w") as dest:
                path = pathlib.Path(QuasiStatic.filename2fastload(source.filename))
                if path.exists():
                    dfile = pathlib.Path(QuasiStatic.filename2fastload(dest.filename))
                    dfile.symlink_to(os.path.relpath(path, dfile.parent))

                GooseHDF5.copy(source, dest, ["/param", "/meta", "/realisation"])
                dest["/param/data_version"][...] = data_version

                system = QuasiStatic.allocate_system(source)

                A = info["full"][filename]["A"][...]
                step = info["full"][filename]["step"][...].astype(np.int64)
                kick = info["full"][filename]["kick"][...]
                u_frame = info["full"][filename]["u_frame"][...]
                f_frame = info["full"][filename]["f_frame"][...]
                steadystate = QuasiStatic.steadystate(u_frame, f_frame, kick, A, N)
                assert np.all(A[~kick] == 0)
                systemspanning = step[np.logical_and(A == N, step > steadystate)]

                QuasiStatic.create_check_meta(dest, "/meta/Trigger_Generate", dev=args.develop)

                meta = dest.create_group("/Trigger/branches")
                meta.attrs["try_p"] = "First particle to try to trigger [ntrigger]"
                meta.attrs["p"] = "Actual triggered particle [ntrigger]"
                meta.attrs["A"] = "Number of yielding particles during event [ntrigger]"
                meta.attrs["S"] = "Total number of yielding events during event [ntrigger]"
                meta.attrs["T"] = "Event duration [ntrigger]"
                meta.attrs["completed"] = "True if trigger was successfully completed [ntrigger]"
                meta.attrs["truncated"] = "True if trigger was truncated at A == N [ntrigger]"
                meta.attrs["f_frame"] = "Frame force (after minimisation) [ntrigger]"

                key = "/Trigger/step"
                dset = dest.create_dataset(key, shape=(0,), maxshape=(None,), dtype=np.int64)
                dset.attrs["desc"] = "QuasiStatic step on which configuration is based"

                key = "/Trigger/step_c"
                dset = dest.create_dataset(key, shape=(0,), maxshape=(None,), dtype=np.int64)
                dset.attrs["desc"] = "QuasiStatic step of last system spanning event"

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
                    u = qsroot["u"][str(s)][...]
                    u_frame = qsroot["u_frame"][s]
                    inc = qsroot["inc"][s]

                    if load:
                        system.restore_quasistatic_step(qsroot, s)
                        system.advanceToFixedForce(f)
                        u = system.u
                        u_frame = system.u_frame
                        assert np.isclose(np.mean(system.f_frame), f)

                    storage.dset_extend1d(dest, "/Trigger/step", ibranch, s)
                    storage.dset_extend1d(dest, "/Trigger/step_c", ibranch, start)
                    storage.dset_extend1d(dest, "/Trigger/loaded", ibranch, load)

                    root = dest.create_group(f"/Trigger/branches/{ibranch:d}")

                    root.create_group("u").create_dataset("0", data=u)
                    root.create_dataset("inc", data=[inc], maxshape=(None,), dtype=np.uint64)
                    root.create_dataset(
                        "u_frame", data=[u_frame], maxshape=(None,), dtype=np.float64
                    )

                    root.create_dataset("try_p", data=[-1, 0], maxshape=(None,), dtype=np.int64)
                    root.create_dataset("p", data=[-1, -1], maxshape=(None,), dtype=np.int64)
                    root.create_dataset("completed", data=[True], maxshape=(None,), dtype=bool)
                    root.create_dataset("truncated", data=[False], maxshape=(None,), dtype=bool)

                    root.create_dataset("f_frame", data=[f], maxshape=(None,), dtype=np.float64)
                    root.create_dataset("A", data=[0], maxshape=(None,), dtype=np.int64)
                    root.create_dataset("S", data=[0], maxshape=(None,), dtype=np.int64)
                    root.create_dataset("T", data=[0], maxshape=(None,), dtype=np.int64)

                    dest.flush()
                    ibranch += 1


def _Merge_parser():
    parser = argparse.ArgumentParser(
        formatter_class=QuasiStatic.MyFmt, description=textwrap.dedent(Merge.__doc__)
    )
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("source", type=str, help="File to merge")
    parser.add_argument("destination", type=str, help="File where 'source' is merged into")
    return parser


def Merge(cli_args=None):
    """
    Merge a previous run "push" file (``source``) into a "push" file (``destination``).
    Only pushes that are not present in the ``destination`` file are added.

    There are assertions on:

    -   The parameters.
    -   The position of the frame and of the particles before triggering.
    -   The particle tried for triggering.
    -   the increment (a.k.a. time).
    """
    parser = _Merge_parser()
    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.source)
    assert os.path.isfile(args.destination)

    with h5py.File(args.source) as src, h5py.File(args.destination, "a") as dest:
        # catching old bug
        if src["/Trigger/step"].size == 1 and "0" not in src["/Trigger/branches"]:
            return

        assert "Trigger" in src
        assert "Trigger" in dest
        assert src["/Trigger/step"].size <= dest["/Trigger/step"].size

        test = GooseHDF5.compare(src, dest, GooseHDF5.getdatapaths(src, root="/param"))

        assert len(test["!="]) == 0
        assert len(test["->"]) == 0
        assert len(test["<-"]) == 0
        assert len(test["=="]) > 0

        branches = np.arange(src["/Trigger/step"].size)

        for ibranch in branches:
            sroot = src[f"/Trigger/branches/{ibranch:d}"]
            droot = dest[f"/Trigger/branches/{ibranch:d}"]

            assert sroot["inc"][0] == droot["inc"][0]
            assert sroot["try_p"][1] == droot["try_p"][1]
            assert np.isclose(sroot["u_frame"][0], droot["u_frame"][0])
            assert np.allclose(sroot["u"]["0"][...], droot["u"]["0"][...])

        if "/meta/Trigger_Run" not in dest:
            GooseHDF5.copy(src, dest, "/meta/Trigger_Run")

        for ibranch in branches:
            sroot = src[f"/Trigger/branches/{ibranch:d}"]
            droot = dest[f"/Trigger/branches/{ibranch:d}"]

            if sroot["completed"].size < 2:
                continue

            if not sroot["completed"][1]:
                continue

            if droot["completed"].size >= 2:
                test = GooseHDF5.compare(
                    src, dest, GooseHDF5.getdatapaths(src, root=f"/Trigger/branches/{ibranch:d}")
                )
                assert len(test["!="]) == 0
                assert len(test["->"]) == 0
                assert len(test["<-"]) == 0
                assert len(test["=="]) > 0
                continue

            src.copy(sroot["u"]["1"], droot["u"], "1")
            storage.dset_extend1d(droot, "inc", 1, sroot["inc"][1])
            storage.dset_extend1d(droot, "u_frame", 1, sroot["u_frame"][1])
            storage.dset_extend1d(droot, "completed", 1, sroot["completed"][1])
            storage.dset_extend1d(droot, "truncated", 1, sroot["truncated"][1])
            storage.dset_extend1d(droot, "p", 1, sroot["p"][1])
            storage.dset_extend1d(droot, "S", 1, sroot["S"][1])
            storage.dset_extend1d(droot, "A", 1, sroot["A"][1])
            storage.dset_extend1d(droot, "T", 1, sroot["T"][1])
            storage.dset_extend1d(droot, "f_frame", 1, sroot["f_frame"][1])
            storage.dset_extend1d(droot, "p", 1, sroot["p"][1])
            storage.dset_extend1d(droot, "p", 2, sroot["p"][2])
            storage.dset_extend1d(droot, "try_p", 2, sroot["try_p"][2])
            dest.flush()


def _MergeBatch_parser():
    parser = argparse.ArgumentParser(
        formatter_class=QuasiStatic.MyFmt, description=textwrap.dedent(MergeBatch.__doc__)
    )
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-o", "--output", required=True, type=str, help="Destination directory")
    parser.add_argument("files", nargs="*", type=str, help="Files to extract data from")
    return parser


def MergeBatch(cli_args=None):
    """
    Merge a a batch of files.
    This is a wrapper around :py:func:`depinning_inertia_2024.Trigger.Merge`.
    The usage is::

        :py:func:`depinning_inertia_2024.Trigger.MergeBatch` -o delta-f=x /path/to/old/delta-f=x/id*h5
    """
    parser = _MergeBatch_parser()
    args = tools._parse(parser, cli_args)
    dest = pathlib.Path(args.output)
    assert dest.is_dir()
    assert all([os.path.isfile(i) for i in args.files])
    destinations = [str(dest / pathlib.Path(i).name) for i in args.files]
    assert all([os.path.isfile(i) for i in destinations])

    pbar = tqdm.tqdm(args.files)

    for src, dest in zip(pbar, destinations):
        pbar.set_description(src)
        pbar.refresh()
        Merge([src, dest])


def _JobRerun_parser():
    parser = argparse.ArgumentParser(
        formatter_class=QuasiStatic.MyFmt, description=textwrap.dedent(JobRerun.__doc__)
    )
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
    return parser


def JobRerun(cli_args=None):
    """
    Create jobs to get event maps.
    The following options are available:

    -   ``--largest-avalanches=n``: the (maximum) ``n`` largest avalanches.
    -   ``--system-spanning=n``: (maximum) ``n`` avalanches.

    In the case of triggering the events are selected based on the distance between the force
    and the mean of the selected bin.
    """
    parser = _JobRerun_parser()
    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.info)
    sourcedir = pathlib.Path(args.info).resolve().parent
    outdir = pathlib.Path(args.output).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.info) as file:
        if "/meta/Trigger_EnsembleInfo" in file:
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

        elif "/meta/QuasiStatic_EnsembleInfo" in file or "avalanche" in file:
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
        executable = "EventMap_Run"
    elif args.dynamics:
        executable = "Dynamics_Run"
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

        commands.append(f"{executable} {opts} {src}")

    shelephant.yaml.dump(args.output, commands, force=True)


def _Paraview_parser():
    parser = argparse.ArgumentParser(
        formatter_class=QuasiStatic.MyFmt, description=textwrap.dedent(Paraview.__doc__)
    )
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, required=True, help="Appended xdmf/h5py")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")
    return parser


def Paraview(cli_args=None):
    """
    Write states to be viewed in Paraview.
    """
    parser = _Paraview_parser()
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
        for i in range(file["/Trigger/step"].size):
            if f"/Trigger/branches/{i}/u/1" not in file:
                break

            u = file[f"/Trigger/branches/{i}/u/0"][...]
            u1 = file[f"/Trigger/branches/{i}/u/1"][...]

            if u.ndim == 1:
                u = u.reshape(-1, 1)
                u1 = u1.reshape(-1, 1)

            if i == 0:
                mesh = GooseFEM.Mesh.Quad4.Regular(u.shape[0] - 1, u.shape[1] - 1)
                coor = xh.as3d(mesh.coor())
                out["coor"] = coor
                out["conn"] = mesh.conn()
                disp = np.zeros_like(coor)

            disp[:, -1] = (u - np.mean(u)).ravel()
            out[f"/disp/{2 * i}"] = disp
            out[f"/S/{2 * i}"] = np.zeros(u.size, dtype=np.int64)

            xdmf += xh.TimeStep(time=2 * i)
            xdmf += xh.Unstructured(out["coor"], out["conn"], xh.ElementType.Quadrilateral)
            xdmf += xh.Attribute(out[f"/disp/{2 * i}"], xh.AttributeCenter.Node, name="du")
            xdmf += xh.Attribute(out[f"/S/{2 * i}"], xh.AttributeCenter.Node, name="S")

            disp[:, -1] = (u1 - np.mean(u1)).ravel()
            out[f"/disp/{2 * i + 1}"] = disp
            out[f"/S/{2 * i + 1}"] = (u1 - u).ravel()

            xdmf += xh.TimeStep(time=2 * i + 1)
            xdmf += xh.Unstructured(out["coor"], out["conn"], xh.ElementType.Quadrilateral)
            xdmf += xh.Attribute(out[f"/disp/{2 * i + 1}"], xh.AttributeCenter.Node, name="du")
            xdmf += xh.Attribute(out[f"/S/{2 * i + 1}"], xh.AttributeCenter.Node, name="S")
