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
from . import Dynamics
from . import storage
from . import tools
from ._version import version

entry_points = dict(
    cli_run="Relaxation_Run",
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
    parser.add_argument("--t-step", type=int, default=500, help="Save every t-step")
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
            assert f"/Trigger/branches/{args.branch:d}/x/{args.step - 1:d}" in src
        else:
            assert f"/QuasiStatic/x/{args.step - 1:d}" in src

    with h5py.File(args.output, "w") as file:

        with h5py.File(args.file) as src:
            GooseHDF5.copy(src, file, ["/param", "/meta", "/realisation"])

        meta = QuasiStatic.create_check_meta(file, f"/meta/{progname}", dev=args.develop)
        meta.attrs["file"] = os.path.basename(args.file)
        meta.attrs["step"] = args.step
        meta.attrs["t-step"] = args.t_step

        root = file.create_group("Relaxation")

        system, info = Dynamics.load_system(filepath=args.file, step=args.step, branch=args.branch, apply_trigger=True)

        if args.branch is not None:
            meta.attrs["branch"] = args.branch
            meta.attrs["p"] = p

        # rerun dynamics and store every other time

        pbar = tqdm.tqdm(total=info["duration"])
        pbar.set_description(args.output)

        systemspanning = False

        while not systemspanning:
            system.timeStepsUntilEvent()
            systemspanning = np.all(np.equal(system.i, info["i_n"]))

        ret = 1

        with GooseHDF5.ExtendableList(file, "v") as ret_v, GooseHDF5.ExtendableList(file, "f_frame") as ret_fext, GooseHDF5.ExtendableList(file, "f_potential") as ret_fpot:
            while ret != 0:
                ret_v.append(np.mean(system.v))
                ret_fext.append(np.mean(system.f_frame))
                ret_fpot.append(np.mean(system.f_potential))
                ret = system.minimise(max_iter=args.t_step, max_iter_is_error=False)
                assert ret >= 0

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
                norm = QuasiStatic.Normalisation(file).asdict()
                norm.pop("seed")
                N = norm["N"]
                dt = norm["dt"]

            else:

                assert t_step == file[f"/meta/{entry_points['cli_run']}"].attrs["t-step"]
                n = QuasiStatic.Normalisation(file).asdict()
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
            A=enstat.static(shape=n),
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
            system = QuasiStatic.allocate_system(file)

            if "fastload" in file:
                r = file["fastload"]
                system.chunk.restore(r["state"][...], r["value"][...], r["index"][...])

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

                if item == 0:
                    i_n = system.i
                    x_n = np.copy(system.x)

                i = system.i
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
                    data["A"].add_point(np.sum(broken), j)

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
