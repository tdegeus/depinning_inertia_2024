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

from . import QuasiStatic
from . import storage
from . import tools
from ._version import version

entry_points = dict(
    cli_average_systemspanning="Dynamics_AverageSystemSpanning",
    cli_run="Dynamics_Run",
)

file_defaults = dict(
    cli_average_systemspanning="Dynamics_AverageSystemSpanning.h5",
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
    Rerun an event and store output at different increments that are selected at:
    *   Given event sizes "A" unit the event is system spanning (``--A-step`` controls interval).
    *   Given time-steps if no longer checking at "A" (interval controlled by ``--t-step``).

    Customisation:
    *   ``--t-step=0``: Break simulation when ``A = N``.
    *   ``--A-step=0``: Store at fixed time intervals from the beginning.

    Storage:
    *   An exact copy of the input file.
    *   The position of all particles ("/Dynamics/x/{iiter:d}").
    *   Metadata:
        - "/Dynamics/inc": Increment number (-> time).
        - "/Dynamics/A": Actual number of blocks that yielded at least once.
        - "/Dynamics/stored": The stored "iiter".
        - "/Dynamics/sync-A": List of "iiter" stored due to given "A".
        - "/Dynamics/sync-t": List of "iiter" stored due to given "inc" after checking for "A".
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

    # output selection
    parser.add_argument("--A-step", type=int, default=1, help="Control sync-A storage")
    parser.add_argument("--t-step", type=int, default=500, help="Control sync-t storage")

    # input selection
    parser.add_argument("--step", type=int, help="Quasistatic step to run")
    parser.add_argument("--branch", type=int, help="Select branch to run (first trigger)")

    # output file
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file")

    # input files
    parser.add_argument("file", type=str, help="Simulation from which to run (read-only)")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    assert os.path.abspath(args.file) != os.path.abspath(args.output)
    tools._check_overwrite_file(args.output, args.force)
    assert args.A_step > 0 or args.t_step > 0

    # basic assertions
    with h5py.File(args.file) as src:
        if args.branch is not None:
            assert f"/Trigger/branches/{args.branch:d}/x/{args.step - 1:d}" in src
        else:
            assert f"/QuasiStatic/x/{args.step - 1:d}" in src

    with h5py.File(args.output, "w") as file:

        # copy from input

        with h5py.File(args.file) as src:

            GooseHDF5.copy(src, file, ["/param", "/meta"])
            root = file.create_group("Dynamics")
            root.create_group("x")

            meta = QuasiStatic.create_check_meta(file, f"/meta/{progname}", dev=args.develop)
            meta.attrs["file"] = os.path.basename(args.file)
            meta.attrs["A-step"] = args.A_step
            meta.attrs["t-step"] = args.t_step
            meta.attrs["step"] = args.step

            if args.branch is not None:
                sroot = src[f"/Trigger/branches/{args.branch:d}"]
                p = sroot["p"][args.step]
                meta.attrs["branch"] = args.branch
                meta.attrs["p"] = p
            else:
                sroot = src["/QuasiStatic"]
                kick = sroot["kick"][args.step]

            root.create_dataset("x_frame", data=[sroot["x_frame"][args.step - 1]], maxshape=(None,))
            root.create_dataset("inc", data=[sroot["inc"][args.step - 1]], maxshape=(None,))

            # ensure a chunk that will be big enough
            system = QuasiStatic.System(file)
            system.restore_quasistatic_step(sroot, args.step)
            i_n = system.istart + system.i
            system.restore_quasistatic_step(sroot, args.step - 1)
            nchunk = np.max(i_n - system.istart + system.i)
            file["/param/xyield/nchunk"][...] = int(
                max(1.5 * nchunk, file["/param/xyield/nchunk"][...])
            )

            # estimate number of steps
            maxinc = sroot["inc"][args.step] - sroot["inc"][args.step - 1]

        # storage preparation

        storage.create_extendible(
            root, "A", np.uint64, desc='Size "A" of each stored item'
        )
        storage.create_extendible(
            root, "sync-A", np.uint64, desc="Items stored due to sync-A"
        )
        storage.create_extendible(
            root, "sync-t", np.uint64, desc="Items stored due to sync-t"
        )

        # rerun dynamics and store every other time

        pbar = tqdm.tqdm(total=maxinc)
        pbar.set_description(args.output)

        dx = file["/param/xyield/dx"][...]
        i_n = system.istart + system.i
        i = np.copy(i_n)
        N = system.N

        A = 0  # maximal A encountered so far
        A_next = 0  # next A at which to write output
        A_istore = 0  # storage index for sync-A storage
        t_istore = 0  # storage index for sync-t storage
        A_check = args.A_step > 0  # switch to store sync-A
        trigger = True  # signal if trigger is needed
        store = True  # signal if storage is needed
        iiter = 0  # total number of elapsed iterations
        istore = 0  # index of the number of storage steps
        stop = False
        last_stored_iiter = -1  # last written increment

        while True:

            if store:

                if iiter != last_stored_iiter:
                    root["x"][str(istore)] = system.x
                    storage.dset_extend1d(root, "x_frame", istore, system.x_frame)
                    storage.dset_extend1d(root, "inc", istore, system.inc)
                    storage.dset_extend1d(root, "A", istore, np.sum(np.not_equal(i, i_n)))
                    file.flush()
                    istore += 1
                    last_stored_iiter = iiter
                store = False
                pbar.n = iiter
                pbar.refresh()

            if stop:

                break

            if trigger:

                trigger = False
                if kick is None:
                    system.trigger(p=p, eps=dx, direction=1)
                else:
                    system.eventDrivenStep(dx, kick)

            if A_check:

                niter = system.timeStepsUntilEvent()
                iiter += niter
                stop = niter == 0
                i = np.copy(system.istart + system.i)
                a = np.sum(np.not_equal(i, i_n))
                A = max(A, a)

                if (A >= A_next and A % args.A_step == 0) or A == N:
                    storage.dset_extend1d(root, "sync-A", A_istore, istore)
                    A_istore += 1
                    store = True
                    A_next += args.A_step

                if A == N:
                    storage.dset_extend1d(root, "sync-t", t_istore, istore)
                    t_istore += 1
                    store = True
                    A_check = False
                    if args.t_step == 0:
                        stop = True

            else:

                inc_n = system.inc
                ret = system.minimise(max_iter=args.t_step, max_iter_is_error=False, nmargin=5)
                assert ret >= 0
                iiter += system.inc - inc_n
                stop = ret == 0
                storage.dset_extend1d(root, "sync-t", t_istore, istore)
                t_istore += 1
                store = True

        meta.attrs["completed"] = 1


class AlignedAverage(enstat.static):
    """
    Support class for :py:func:`cli_average_systemspanning`.
    This class writes on item at a time using :py:func:`BasicAverage.subsample`.
    """

    def __init__(self, shape):
        """
        :param shape: Shape of the averaged field: ``[nitem, N]``.
        """
        assert len(shape) == 2
        super().__init__(shape=shape)

    def subsample(self, index, data, roll, broken=None):
        """
        :param index: Index of the item to add to the average.
        :param data: Data to add to the average.
        :param roll: Roll to apply to align the data.
        :param broken: Array with per weak element whether the element is broken.
        """
        assert data.ndim == 1
        data = np.roll(data, roll)

        if broken is None:
            self.first[index, :] += data
            self.second[index, :] += data**2
            self.norm[index, :] += 1
        else:
            incl = np.roll(broken, roll)
            self.first[index, incl] += data[incl]
            self.second[index, incl] += data[incl] ** 2
            self.norm[index, incl] += 1


def cli_average_systemspanning(cli_args=None):
    """
    Compute averages from output of :py:func:`cli_run`:

    -   'Simple' averages (macroscopic, on moving particles):

        *   For bins of time compared to the time when the event is system-spanning.
        *   For fixed ``A``.

    -   'Aligned' averages (for different element rows), for fixed A.
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

    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("files", nargs="*", type=str, help="See " + entry_points["cli_run"])

    args = tools._parse(parser, cli_args)
    assert len(args.files) > 0
    assert all([os.path.isfile(file) for file in args.files])
    tools._check_overwrite_file(args.output, args.force)
    QuasiStatic.create_check_meta(dev=args.develop)

    # get duration of each event and allocate binning on duration since system spanning

    t_start = []
    t_end = []

    for ifile, filepath in enumerate(args.files):

        with h5py.File(filepath, "r") as file:

            if ifile == 0:

                t_step = file[f"/meta/{entry_points['cli_run']}"].attrs["t-step"]
                norm = QuasiStatic.normalisation(file)
                norm.pop("seed")
                N = norm["N"]
                dt = norm["dt"]

            else:

                assert t_step == file[f"/meta/{entry_points['cli_run']}"].attrs["t-step"]
                n = QuasiStatic.normalisation(file)
                for key in norm:
                    assert norm[key] == n[key]

            t = file["/Dynamics/inc"][...].astype(float)
            A = file["/Dynamics/A"][...]
            assert np.sum(A == N) > 0
            t = np.sort(t) - np.min(t[A == N])
            t_start.append(t[0])
            t_end.append(t[-1])

    t_bin = np.arange(np.min(t_start), np.max(t_end) + 3 * t_step, t_step)
    t_bin = t_bin - np.min(t_bin[t_bin > 0]) - 0.5 * t_step
    t_mid = 0.5 * (t_bin[1:] + t_bin[:-1])

    # allocate averages

    def allocate(n):
        return dict(
            delta_t=enstat.static(shape=n),
            f_potential=enstat.static(shape=n),
            f_frame=enstat.static(shape=n),
            f_neighbours=enstat.static(shape=n),
            dx=enstat.static(shape=n),
            S=enstat.static(shape=n),
            f_potential_moving=enstat.static(shape=n),
            f_frame_moving=enstat.static(shape=n),
            f_neighbours_moving=enstat.static(shape=n),
            dx_moving=enstat.static(shape=n),
        )

    synct = allocate(t_bin.size - 1)
    syncA = allocate(N + 1)

    for title in ["align", "align_moving"]:
        syncA[title] = dict(
            f_potential=AlignedAverage(shape=(N + 1, N)),
            f_frame=AlignedAverage(shape=(N + 1, N)),
            f_neighbours=AlignedAverage(shape=(N + 1, N)),
            dx=AlignedAverage(shape=(N + 1, N)),
            s=AlignedAverage(shape=(N + 1, N)),
        )

    # averages

    fmt = "{:" + str(max(len(i) for i in args.files)) + "s}"
    pbar = tqdm.tqdm(args.files)
    pbar.set_description(fmt.format(""))

    for ifile, filepath in enumerate(pbar):

        pbar.set_description(fmt.format(filepath), refresh=True)

        with h5py.File(filepath, "r") as file:

            root = file["Dynamics"]
            system = QuasiStatic.System(file)
            system.restore_quasistatic_step(root, 0)

            # determine duration bin, ensure that only one measurement per bin is added
            # (take the one closest to the middle of the bin)

            nitem = root["inc"].size
            items_syncA = file["/Dynamics/sync-A"][...]
            A = file["/Dynamics/A"][...]
            t = file["/Dynamics/inc"][...].astype(np.int64)
            delta_t = t - np.min(t[A == N])
            t_ibin = np.digitize(delta_t, t_bin) - 1
            d = np.abs(delta_t - t_mid[t_ibin])

            for ibin in np.unique(t_ibin):
                idx = np.argwhere(t_ibin == ibin).ravel()
                if len(idx) <= 1:
                    continue
                jdx = idx[np.argmin(d[idx])]
                t_ibin[idx] = -1
                t_ibin[jdx] = ibin

            del d

            # add averages

            keep = t_ibin >= 0
            synct["delta_t"].add_point(dt * delta_t[keep], t_ibin[keep])
            syncA["delta_t"].add_point(dt * delta_t[items_syncA], A[items_syncA])

            for item in tqdm.tqdm(range(nitem)):

                if item not in items_syncA and t_ibin[item] < 0 and item > 0:
                    continue

                system.x = file[f"/Dynamics/x/{item:d}"][...]
                assert np.all(system.i > 5) and np.all(system.i < system.y.shape[1] - 5)

                if item == 0:
                    i_n = system.istart + system.i
                    x_n = np.copy(system.x)

                i = system.istart + system.i
                broken = i != i_n

                # synct / syncA

                for data, store, j in zip(
                    [synct, syncA],
                    [t_ibin[item] >= 0, item in items_syncA],
                    [t_ibin[item], A[item]],
                ):

                    if not store:
                        continue

                    data["f_potential"].add_point(np.mean(system.f_potential), j)
                    data["f_frame"].add_point(np.mean(system.f_frame), j)
                    data["f_neighbours"].add_point(np.mean(system.f_neighbours), j)
                    data["dx"].add_point(np.mean(system.x - x_n), j)
                    data["S"].add_point(np.mean(i - i_n), j)

                    if np.sum(broken) == 0:
                        continue

                    data["f_potential_moving"].add_point(np.mean(system.f_potential[broken]), j)
                    data["f_frame_moving"].add_point(np.mean(system.f_frame[broken]), j)
                    data["f_neighbours_moving"].add_point(np.mean(system.f_neighbours[broken]), j)
                    data["dx_moving"].add_point(np.mean((system.x - x_n)[broken]), j)

                # syncA["align_moving"]

                if item in items_syncA and np.sum(broken) > 0:

                    j = A[item]
                    roll = tools.center_avalanche(broken)

                    data = syncA["align"]
                    data["f_potential"].subsample(j, np.copy(system.f_potential), roll)
                    data["f_frame"].subsample(j, np.copy(system.f_frame), roll)
                    data["f_neighbours"].subsample(j, np.copy(system.f_neighbours), roll)
                    data["s"].subsample(j, i - i_n, roll)
                    data["dx"].subsample(j, system.x - x_n, roll)

                    data = syncA["align_moving"]
                    data["f_potential"].subsample(j, np.copy(system.f_potential), roll, broken)
                    data["f_frame"].subsample(j, np.copy(system.f_frame), roll, broken)
                    data["f_neighbours"].subsample(j, np.copy(system.f_neighbours), roll, broken)
                    data["s"].subsample(j, i - i_n, roll, broken)
                    data["dx"].subsample(j, system.x - x_n, roll, broken)

    with h5py.File(args.output, "w") as file:

        QuasiStatic.create_check_meta(file, f"/meta/{progname}", dev=args.develop)

        file["files"] = [os.path.relpath(i, os.path.dirname(args.output)) for i in args.files]

        for key in norm:
            file[f"/normalisation/{key}"] = norm[key]

        for title, data in zip(["sync-t", "sync-A"], [synct, syncA]):

            for key in data:
                if key in ["align", "align_moving"]:
                    continue
                file[f"/{title}/{key}/first"] = data[key].first
                file[f"/{title}/{key}/second"] = data[key].second
                file[f"/{title}/{key}/norm"] = data[key].norm

        for title in ["align", "align_moving"]:
            for key in syncA["align_moving"]:
                file[f"/sync-A/{title}/{key}/first"] = syncA[title][key].first
                file[f"/sync-A/{title}/{key}/second"] = syncA[title][key].second
                file[f"/sync-A/{title}/{key}/norm"] = syncA[title][key].norm
