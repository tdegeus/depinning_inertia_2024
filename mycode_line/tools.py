from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

import click
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import GooseHDF5 as g5
import h5py
import numpy as np
import yaml
from numpy.typing import ArrayLike


def _parse(parser: argparse.ArgumentParser, cli_args: list[str]) -> argparse.ArgumentParser:

    if cli_args is None:
        return parser.parse_args(sys.argv[1:])

    return parser.parse_args([str(arg) for arg in cli_args])
