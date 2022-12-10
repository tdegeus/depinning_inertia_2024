import argparse
import inspect
import os
import pathlib
import textwrap

import GooseSLURM
import numpy as np
import shelephant

from . import tools
from ._version import version

default_condabase = "code_line"

entry_points = dict(
    cli_serial="JobSerial",
    cli_from_yaml="JobFromYAML",
    cli_to_text="JobToText",
)


def replace_ep(docstring):
    """
    Replace ":py:func:`...`" with the relevant entry_point name
    """
    for ep in entry_points:
        docstring = docstring.replace(rf":py:func:`{ep:s}`", entry_points[ep])
    return docstring


def flush_command(cmd):
    """
    Return code to run a command and flush the buffer of stdout.
    :param cmd: The command.
    :return: str
    """
    return "stdbuf -o0 -e0 " + cmd


def script_exec(commands: list[str] | str, condabase: str = None, append_simd: bool = False):
    """
    Return code to execute a command.

    :param commands: The command(s).
    :param condabase: The (base)name of the Conda environment.
    :param append_simd: Append the SIMD extension to the Conda environment name.
    :param flush: Flush the buffer of stdout.
    :return: str
    """

    if isinstance(commands, str):
        commands = [commands]
    else:
        assert isinstance(commands, list)

    ret = []

    ret += ["# Initialise the environment"]
    ret += ["source $HOME/myinit/compiler_conda.sh"]
    ret += [""]

    ret += ["# Set number of cores to use"]
    ret += ["export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}"]
    ret += [""]

    if condabase is not None:
        activate = "conda_activate_first_existing"
        ret += ["# Activate environment. To overwrite the default environment at submit-time:"]
        ret += ['# export conda_basename="myenv"; sbatch ...']
        ret += ['if [[ -z "${conda_basename}" ]]; then']
        ret += [f'    conda_basename="{condabase}"']
        ret += ["fi"]
        if append_simd:
            ret += [activate + ' "${conda_basename}$(get_simd_envname)" "${conda_basename}"']
        else:
            ret += [activate + ' "${conda_basename}"']
        ret += [""]

    ret += ["# Run"]
    ret += commands
    ret += [""]

    return "\n".join(ret)


def serial(
    command: str,
    name: str,
    outdir: str = os.getcwd(),
    sbatch: dict = None,
    condabase: str = None,
    append_simd: bool = False,
    flush: bool = True,
):
    """
    Create job script to run a command.

    :param command: Command.
    :param name: Basename of the filenames of the job-script (and log-scripts), and the job-name.
    :param outdir: Directory where to write the job-scripts (nothing in changed for the commands).
    :param sbatch: Job options.
    :param condabase: The (base)name of the Conda environment.
    :param append_simd: Append the SIMD extension to the Conda environment name.
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
    sbatch.setdefault("account", "pcsl")
    sbatch.setdefault("partition", "serial")

    if flush:
        command = flush_command(command)

    command = script_exec(command, condabase, append_simd)

    sbatch["job-name"] = name
    sbatch["out"] = name + "_%j.out"

    with open(os.path.join(outdir, name + ".slurm"), "w") as file:
        file.write(GooseSLURM.scripts.plain(command=command, **sbatch))


def serial_group(
    commands: list[str],
    name: str,
    group: int,
    outdir: str = os.getcwd(),
    sbatch: dict = None,
    condabase: str = None,
    append_simd: bool = False,
    flush: bool = True,
):
    """
    Group a number of commands per job-script.
    Note that the ``name`` is the basename of all jobs.

    :param commands: List of commands.
    :param name: Basename of the filenames of the job-script (and log-scripts), and the job-name.
    :param group: Number of commands to group per job-script.
    :param outdir: Directory where to write the job-scripts (nothing in changed for the commands).
    :param sbatch: Job options.
    :param condabase: The (base)name of the Conda environment.
    :param append_simd: Append the SIMD extension to the Conda environment name.
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
    sbatch.setdefault("account", "pcsl")
    sbatch.setdefault("partition", "serial")

    if flush:
        commands = [flush_command(cmd) for cmd in commands]

    chunks = int(np.ceil(len(commands) / float(group)))
    devided = np.array_split(commands, chunks)
    njob = len(devided)
    fmt = str(int(np.ceil(np.log10(njob + 1))))

    if name.format(index="foo") == name.format(index=""):
        name = name + "_{index:s}"

    for g, selection in enumerate(devided):

        command = script_exec(list(selection), condabase, append_simd)
        jobname = name.format(index=("{0:0" + fmt + "d}-of-{1:d}").format(g + 1, njob))
        sbatch["job-name"] = jobname
        sbatch["out"] = jobname + "_%j.out"

        with open(os.path.join(outdir, jobname + ".slurm"), "w") as file:
            file.write(GooseSLURM.scripts.plain(command=command, **sbatch))


def cli_serial(cli_args=None):
    """
    Job-script to run a command.
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

    a = "pcsl"
    c = default_condabase

    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-n", "--name", type=str, default="job", help="Basename for all scripts.")
    parser.add_argument("-o", "--outdir", type=str, default=".", help="Output dir")

    parser.add_argument("--conda", type=str, default=c, help="(Base)name of the conda environment")
    parser.add_argument(
        "--append-simd", action="store_true", help="Append SIMD extension to conda environment name"
    )

    parser.add_argument("-a", "--account", type=str, default=a, help="Account (sbatch)")
    parser.add_argument("-w", "--time", type=str, default="24h", help="Walltime (sbatch)")
    parser.add_argument("-m", "--mem", type=str, default="2G", help="Memory (sbatch)")
    parser.add_argument("-p", "--partition", type=str, default="serial", help="Partition (sbatch)")
    parser.add_argument("command", type=str, help="The command")

    args = tools._parse(parser, cli_args)

    serial(
        args.command,
        name=args.name,
        outdir=args.outdir,
        condabase=args.conda,
        append_simd=args.append_simd,
        sbatch={
            "time": args.time,
            "mem": args.mem,
            "account": args.account,
            "partition": args.partition,
        },
    )


def cli_from_yaml(cli_args=None):
    """
    Create job-scripts from commands stored in a YAML file.
    Note that the job-scripts are written to the same directory as the YAML file.
    Note that the ``--name`` is the basename of all jobs.
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

    a = "pcsl"
    c = default_condabase

    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-n", "--name", type=str, default="job", help="Basename for all scripts.")

    parser.add_argument("--group", type=int, default=1, help="#commands to group in one script")

    parser.add_argument("--conda", type=str, default=c, help="(Base)name of the conda environment")
    parser.add_argument(
        "--append-simd", action="store_true", help="Append SIMD extension to conda environment name"
    )

    parser.add_argument("-a", "--account", type=str, default=a, help="Account (sbatch)")
    parser.add_argument("-w", "--time", type=str, default="24h", help="Walltime (sbatch)")
    parser.add_argument("-m", "--mem", type=str, default="2G", help="Memory (sbatch)")
    parser.add_argument("-p", "--partition", type=str, default="serial", help="Partition (sbatch)")

    parser.add_argument("-k", "--key", type=str, help="Key to read from the YAML file")
    parser.add_argument("yaml", nargs="*", type=str, help="The YAML file")

    args = tools._parse(parser, cli_args)

    for filepath in args.yaml:

        commands = shelephant.yaml.read(filepath)

        if args.key is not None:
            commands = commands[args.key]

        assert isinstance(commands, list)

        serial_group(
            commands,
            name=args.name,
            group=args.group,
            outdir=pathlib.Path(filepath).parent,
            condabase=args.conda,
            append_simd=args.append_simd,
            sbatch={
                "time": args.time,
                "mem": args.mem,
                "account": args.account,
                "partition": args.partition,
            },
        )


def cli_to_text(cli_args=None):
    """
    Convert to plain text, to run e.g. as::

        parallel --max-procs=1 :::: mytext.txt
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

    parser.add_argument("--develop", action="store_true", help="Run all commands in develop mode")

    parser.add_argument("-k", "--key", type=str, help="Key to read from the YAML file")
    parser.add_argument("yaml", nargs="*", type=str, help="The YAML file")

    args = tools._parse(parser, cli_args)

    for filepath in args.yaml:

        commands = shelephant.yaml.read(filepath)

        if args.key is not None:
            commands = commands[args.key]

        assert isinstance(commands, list)

        print(filepath.replace(".yaml", "") + ".txt")

        if args.develop:
            commands = [c + " --develop" for c in commands]

        with open(filepath.replace(".yaml", "") + ".txt", "w") as fh:
            fh.write("\n".join(commands))
