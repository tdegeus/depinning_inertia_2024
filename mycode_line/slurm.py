import argparse
import inspect
import os
import sys
import textwrap

import GooseSLURM
import numpy as np
import shelephant

from ._version import version

default_condabase = "code_line"

entry_points = dict(
    cli_serial_group="JobSerialGroup",
    cli_serial="JobSerial",
)

slurm_defaults = dict(
    account="pcsl",
)


def replace_entry_point(docstring):
    """
    Replace ":py:func:`...`" with the relevant entry_point name
    """
    for ep in entry_points:
        docstring = docstring.replace(rf":py:func:`{ep:s}`", entry_points[ep])
    return docstring


def snippet_initenv(cmd="source $HOME/myinit/compiler_conda.sh"):
    """
    Return code to initialise the environment.
    :param cmd: The command to run.
    :return: str
    """
    return f"# Initialise the environment\n{cmd}"


def snippet_export_omp_num_threads(ncores=1):
    """
    Return code to set OMP_NUM_THREADS
    :return: str
    """
    return f"# Set number of cores to use\nexport OMP_NUM_THREADS={ncores}"


def snippet_load_conda(condabase: str = default_condabase):
    """
    Return code to load the Conda environment.
    This function assumes that these BASH-functions are present:
    -   ``conda_activate_first_existing``
    -   ``get_simd_envname``
    Use snippet_initenv() to set them.

    :param condabase: Base name of the Conda environment, appended '_E5v4' and '_s6g1'".
    :return: str
    """

    ret = ["# Activate hardware optimised environment (or fallback environment)"]
    ret += [f'conda_activate_first_existing "{condabase}$(get_simd_envname)" "{condabase}"']
    ret += []

    return "\n".join(ret)


def snippet_flush(cmd):
    """
    Return code to run a command and flush the buffer of stdout.
    :param cmd: The command.
    :return: str
    """
    return "stdbuf -o0 -e0 " + cmd


def script_exec(cmd, initenv=True, omp_num_threads=True, conda=True, flush=True):
    """
    Return code to execute a command.
    Optionally a number of extra commands are run before the command itself, see options.
    Defaults of the underlying functions can be overwritten by passing a tuple or dictionary.
    The option can be skipped by specifying ``None`` or ``False``.

    For example::
        slurm.script_exec(cmd, conda=dict(condabase="my"))
        slurm.script_exec(cmd, conda=dict(condabase="my"))

    :param cmd: The command.
    :param initenv: Init the environment (see snippet_initenv()).
    :param omp_num_threads: Number of cores to use (see snippet_export_omp_num_threads()).
    :param conda: Load conda environment (see defaults of snippet_load_conda()).
    :param flush: Flush the buffer of stdout.
    :return: str
    """

    ret = []

    for opt, func in zip(
        [initenv, omp_num_threads, conda],
        [snippet_initenv, snippet_export_omp_num_threads, snippet_load_conda],
    ):
        if opt is True:
            ret += [func(), ""]
        elif opt is not None and opt is not False:
            if type(opt) == dict:
                ret += [func(**opt), ""]
            else:
                ret += [func(*opt), ""]

    if flush:
        ret += ["# --- Run ---", "", snippet_flush(cmd), ""]
    else:
        ret += ["# --- Run ---", "", cmd, ""]

    return "\n".join(ret)


def serial(
    command: str,
    basename: str,
    outdir: str = os.getcwd(),
    sbatch: dict = None,
    initenv=True,
    omp_num_threads=True,
    conda=True,
    flush=True,
):
    """
    Group a number of commands per job-script.

    :param commands: List of commands.
    :param basename: Base-name of the job-scripts (and their log-scripts),
    :param outdir: Directory where to write the job-scripts (nothing in changed for the commands).
    :param sbatch: Job options.
    :param initenv: Init the environment (see snippet_initenv()).
    :param omp_num_threads: Number of cores to use (see snippet_export_omp_num_threads()).
    :param conda: Load conda environment (see defaults of snippet_load_conda()).
    :param flush: Flush the buffer of stdout for each commands.
    """

    if sbatch is None:
        sbatch = {}

    assert "job-name" not in sbatch
    assert "out" not in sbatch
    sbatch.setdefault("nodes", 1)
    sbatch.setdefault("ntasks", 1)
    sbatch.setdefault("cpus-per-task", 1)
    sbatch.setdefault("time", "24h")
    sbatch.setdefault("mem", "2G")
    sbatch.setdefault("account", slurm_defaults["account"])
    sbatch.setdefault("partition", "serial")

    command = script_exec(
        command,
        initenv=initenv,
        omp_num_threads=omp_num_threads,
        conda=conda,
        flush=flush,
    )

    sbatch["job-name"] = basename
    sbatch["out"] = basename + "_%j.out"

    with open(os.path.join(outdir, basename + ".slurm"), "w") as file:
        file.write(GooseSLURM.scripts.plain(command=command, **sbatch))


def serial_group(
    commands: list[str],
    basename: str,
    group: int,
    outdir: str = os.getcwd(),
    sbatch: dict = None,
    initenv=True,
    omp_num_threads=True,
    conda=True,
    flush=True,
):
    """
    Group a number of commands per job-script.

    :param commands: List of commands.
    :param basename: Base-name of the job-scripts (and their log-scripts),
    :param group: Number of commands to group per job-script.
    :param outdir: Directory where to write the job-scripts (nothing in changed for the commands).
    :param sbatch: Job options.
    :param initenv: Init the environment (see snippet_initenv()).
    :param omp_num_threads: Number of cores to use (see snippet_export_omp_num_threads()).
    :param conda: Load conda environment (see defaults of snippet_load_conda()).
    :param flush: Flush the buffer of stdout for each commands.
    """

    if len(commands) == 0:
        return

    if sbatch is None:
        sbatch = {}

    assert "job-name" not in sbatch
    assert "out" not in sbatch
    sbatch.setdefault("nodes", 1)
    sbatch.setdefault("ntasks", 1)
    sbatch.setdefault("cpus-per-task", 1)
    sbatch.setdefault("time", "24h")
    sbatch.setdefault("mem", "2G")
    sbatch.setdefault("account", slurm_defaults["account"])
    sbatch.setdefault("partition", "serial")

    if flush:
        commands = [snippet_flush(cmd) for cmd in commands]

    chunks = int(np.ceil(len(commands) / float(group)))
    devided = np.array_split(commands, chunks)
    njob = len(devided)
    fmt = str(int(np.ceil(np.log10(njob + 1))))
    info = {}

    if type(conda) == dict:
        info["conda"] = conda["condabase"]
    elif conda:
        info["conda"] = default_condabase

    if basename.format(index="foo", conda="") == basename.format(index="", conda=""):
        basename = basename + "_{index:s}"

    for g, selection in enumerate(devided):

        command = script_exec(
            "\n".join(selection),
            initenv=initenv,
            omp_num_threads=omp_num_threads,
            conda=conda,
            flush=False,
        )

        jobname = basename.format(index=("{0:0" + fmt + "d}-of-{1:d}").format(g + 1, njob), **info)
        sbatch["job-name"] = jobname
        sbatch["out"] = jobname + "_%j.out"

        with open(os.path.join(outdir, jobname + ".slurm"), "w") as file:
            file.write(GooseSLURM.scripts.plain(command=command, **sbatch))


def cli_serial_group(cli_args=None):
    """
    Job-script to run commands.
    Note that ``--basename`` can be a format. For example::

        # default: postfix by "_{index:s}", see below
        "myname"

        # index is counter (e.g. "1-of-4"), or
        # (an approximated) basename of the file run if ``--group=1``
        "myname_{index:s}"

        # conda == condabase
        "myname_{index:s}_{conda:s}"
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    docstring = textwrap.dedent(inspect.getdoc(globals()[funcname]))

    if cli_args is None:
        cli_args = sys.argv[1:]
    else:
        cli_args = [str(arg) for arg in cli_args]

    class MyFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter, description=replace_entry_point(docstring)
    )

    account = slurm_defaults["account"]
    parser.add_argument("--conda", type=str, default=default_condabase, help="Env-basename")
    parser.add_argument("-a", "--account", type=str, default=account, help="Account")
    parser.add_argument("-b", "--basename", type=str, help="Basename for scripts. Default: command")
    parser.add_argument("-c", "--command", type=str, help="Command to use")
    parser.add_argument("-k", "--yaml-key", type=str, default="new", help="Key to read from file")
    parser.add_argument("-n", "--group", type=int, default=1, help="#commands to group")
    parser.add_argument("-o", "--outdir", type=str, default=".", help="Output dir")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-w", "--time", type=str, default="24h", help="Walltime")
    parser.add_argument("-y", "--yaml", type=str, help="Input files from YAML file")
    parser.add_argument("files", nargs="*", type=str, help="Files")

    args = parser.parse_args(cli_args)

    if args.yaml is not None:
        args.files += shelephant.yaml.read(args.yaml)[args.yaml_key]

    assert np.all([os.path.isfile(file) for file in args.files])

    files = [os.path.relpath(file, args.outdir) for file in args.files]
    commands = [f"{args.command} {file}" for file in files]
    serial_group(
        commands,
        basename=args.basename if args.basename else args.command,
        group=args.group,
        outdir=args.outdir,
        conda=dict(condabase=args.conda),
        sbatch={"time": args.time, "account": args.account},
    )


def cli_serial(cli_args=None):
    """
    Job-script to run a command.
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    docstring = textwrap.dedent(inspect.getdoc(globals()[funcname]))

    if cli_args is None:
        cli_args = sys.argv[1:]
    else:
        cli_args = [str(arg) for arg in cli_args]

    class MyFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter, description=replace_entry_point(docstring)
    )

    account = slurm_defaults["account"]
    parser.add_argument("--conda", type=str, default=default_condabase, help="Env-basename")
    parser.add_argument("-a", "--account", type=str, default=account, help="Account")
    parser.add_argument("-n", "--name", type=str, help="Job name (default: from command)")
    parser.add_argument("-o", "--outdir", type=str, default=".", help="Output dir")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-w", "--time", type=str, default="24h", help="Walltime")
    parser.add_argument("command", type=str, help="The command")

    args = parser.parse_args(cli_args)

    basename = args.name

    if not basename:
        basename = args.command.split(" ")[0]

    serial(
        args.command,
        basename=basename,
        outdir=args.outdir,
        conda=dict(condabase=args.conda),
        sbatch={"time": args.time},
    )
