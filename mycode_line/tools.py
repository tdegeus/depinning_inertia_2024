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
