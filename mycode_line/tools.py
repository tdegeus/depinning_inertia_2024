from __future__ import annotations

import argparse
import os
import shutil
import sys

import click
import GooseHDF5 as g5
import h5py
import numpy as np
import yaml
from numpy.typing import ArrayLike


def h5py_read_unique(
    file: h5py.File,
    path: str,
    asstr: bool = False,
) -> np.ndarray:
    """
    Return original array stored by :py:func:`h5py_save_unique`.

    :param file: HDF5 archive.
    :param path: Group containing ``index`` and ``value``.
    :param asstr: Return as list of strings.
    :returns: Data.
    """

    index = file[g5.join(path, "index", root=True)][...]

    if asstr:
        value = file[g5.join(path, "value", root=True)].asstr()[...]
    else:
        value = file[g5.join(path, "value", root=True)][...]

    ret = value[index]

    if asstr:
        return ret.tolist()

    return ret


def h5py_save_unique(
    data: ArrayLike,
    file: h5py.File,
    path: str,
    asstr: bool = False,
    split: str = None,
):
    """
    Save a list of strings (or other data, but mostly relevant for strings)
    with many duplicates as two datasets:

    -   ``path/value``: list of unique strings.
    -   ``path/index``: per item which index from ``path/value`` to take.

    Use :py:func:`h5py_read_unique` to read data.

    :param data: Data to store.
    :param file: HDF5 archive.
    :param path: Group containing ``index`` and ``value``.
    :param asstr: Convert to list of strings before storing.
    :param split: Split every item for a list of strings before storing.
    """

    value, index = np.unique(data, return_inverse=True)

    if split is not None:
        value = list(map(lambda i: str(i).split(split), value))
    elif asstr:
        value = list(map(str, value))

    if isinstance(data, np.ndarray):
        index = index.reshape(data.shape)

    file[g5.join(path, "index", root=True)] = index
    file[g5.join(path, "value", root=True)] = value


def inboth(a: dict | list, b: dict | list, name_a: str = "a", name_b: str = "b"):
    """
    Check if a dictionary/list ``a`` has all fields as ``b`` and vice-versa.

    :param a: List or dictionary.
    :param b: List or dictionary.
    """

    for key in a:
        if key not in b:
            raise OSError(f"{key} not in {name_b}")

    for key in b:
        if key not in a:
            raise OSError(f"{key} not in {name_a}")


def check_docstring(string: str, variable: dict, key: str = ":return:"):
    """
    Make sure that all variables in a dictionary are documented in a docstring.
    The function assumes a docstring as follows::

        :param a: ...
        :param b: ...
        :return: ...::
            name: description

    Thereby the docstring is split:
    1.  At a parameter (e.g. `":return:"`)
    2.  At `.. code-block::` or `::`

    The indented code after is assumed to be formatted as YAML and is the code we search.
    """

    d = string.split(":return:")[1]

    if len(d.split(".. code-block::")) > 1:
        d = d.split(".. code-block::")[1].split("\n", 1)[1]
    elif len(d.split("::")) > 1:
        d = d.split("::")[1]

    d = d.split("\n")
    d = list(filter(None, d))
    d = list(filter(lambda name: name.strip(), d))
    indent = len(d[0]) - len(d[0].lstrip())
    d = list(filter(lambda name: len(name) - len(name.lstrip()) == indent, d))
    d = "\n".join(d)

    inboth(yaml.safe_load(d), variable, "docstring", "variable")


def _center_of_mass(x, L):
    """
    Compute the center of mass of a periodic system.
    Assume: all equal masses.

    :param x: List of coordinates.
    :param L: Length of the system.
    :return: Coordinate of the center of mass.
    """

    # todo: vectorise implementation
    # todo: implementation without allocation of coordinates

    if np.allclose(x, 0):
        return 0

    theta = 2.0 * np.pi * x / L
    xi = np.cos(theta)
    zeta = np.sin(theta)
    xi_bar = np.mean(xi)
    zeta_bar = np.mean(zeta)
    theta_bar = np.arctan2(-zeta_bar, -xi_bar) + np.pi
    return L * theta_bar / (2.0 * np.pi)


def _center_of_mass_per_row(arr):
    """
    Compute the center of mass per row.
    The function assumes that masses can be either 0 or 1:
    -   1: any positive value
    -   0: any zero or negative value

    :param: Input array [M, N].
    :return: x-position of the center of mass per row [M].
    """

    assert arr.ndim == 2
    m, n = arr.shape

    ret = np.empty(m)

    for i in range(m):
        ret[i] = _center_of_mass(np.argwhere(arr[i, :] > 0).ravel(), n)

    return ret


def indep_roll(arr, shifts, axis=1):
    """
    Apply an independent roll for each dimensions of a single axis.
    See: https://stackoverflow.com/a/56175538/2646505

    :param arr: Array of any shape.
    :param shifts: Shifting to use for each dimension. Shape: `(arr.shape[axis],)`.
    :param axis: Axis along which elements are shifted.
    :return: Rolled array.
    """
    arr = np.swapaxes(arr, axis, -1)
    all_idcs = np.ogrid[[slice(0, n) for n in arr.shape]]

    # Convert to a positive shift
    shifts[shifts < 0] += arr.shape[-1]
    all_idcs[-1] = all_idcs[-1] - shifts[:, np.newaxis]

    result = arr[tuple(all_idcs)]
    arr = np.swapaxes(result, -1, axis)
    return arr


def center_avalanche_per_row(arr):
    """
    Shift to center avalanche, per row. Example usage::

        R = center_avalanche_per_row(S)
        C = indep_roll(S, R, axis=1)

    Note that the input array is interpreted as follows:
    -   any positive value == 1
    -   any zero or negative value == 0

    :param arr: Per row: if the block yielded.
    :return: Shift per row.
    """

    assert arr.ndim == 2
    n = arr.shape[1]
    shift = np.floor(n / 2 - _center_of_mass_per_row(arr)).astype(int)
    return np.where(shift < 0, n + shift, shift)


def center_avalanche(arr):
    """
    Shift to center avalanche. Example usage::
        R = center_avalanche(S)
        C = np.roll(S, R)

    :param arr: If the block yielded (or the number of times it yielded).
    :return: Shift.
    """

    return center_avalanche_per_row(arr.reshape(1, -1))[0]


def fill_avalanche(broken):
    """
    Fill avalanche such that the largest spatial extension can be selected.

    :param broken: Per block if it is broken.
    :return: ``broken`` for filled avalanche.
    """

    assert broken.ndim == 1

    if np.sum(broken) <= 1:
        return broken

    N = broken.size
    broken = np.tile(broken, 3)
    ret = np.ones_like(broken)
    zero = np.zeros_like(broken)[0]

    x = np.argwhere(broken).ravel()
    du = np.diff(x)
    maxdx = np.max(du)
    j = np.argwhere(du == maxdx).ravel()

    x0 = x[j[0]]
    x1 = x[j[0] + 1]
    ret[x0:x1] = zero

    x0 = x[j[1]] + 1
    x1 = x[j[1] + 1]
    ret[x0:x1] = zero

    i = N
    j = 2 * N
    return ret[i:j]


def _parse(parser: argparse.ArgumentParser, cli_args: list[str]) -> argparse.ArgumentParser:
    if cli_args is None:
        return parser.parse_args(sys.argv[1:])

    return parser.parse_args([str(arg) for arg in cli_args])


def _check_overwrite_file(filepath: str, force: bool):
    if force or not os.path.isfile(filepath):
        return

    if not click.confirm(f'Overwrite "{filepath}"?'):
        raise OSError("Cancelled")


def _create_or_clear_directory(dirpath: str, force: bool):
    if os.path.isdir(dirpath):
        if not force:
            if not click.confirm(f'Clear "{dirpath}"?'):
                raise OSError("Cancelled")

        shutil.rmtree(dirpath)

    os.makedirs(dirpath)
