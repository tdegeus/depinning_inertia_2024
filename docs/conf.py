import os
import pathlib
import re
import sys
import tomllib

sys.path.insert(0, os.path.abspath(".."))

project = "depinning_inertia_2024"
copyright = "2024, Tom de Geus"
author = "Tom de Geus"
html_theme = "furo"
autodoc_type_aliases = {"Iterable": "Iterable", "ArrayLike": "ArrayLike"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinxarg.ext",
]
templates_path = ["_templates"]

# autogenerate modules.rst

modules = list((pathlib.Path(__file__).parent / ".." / project).glob("*.py"))
modules = [m.stem for m in modules]
for name in ["__init__", "_version"]:
    if name in modules:
        modules.remove(name)
modules = sorted(modules)

header = "Python module"
ret = [
    "\n".join(
        [
            "*" * len(header),
            header,
            "*" * len(header),
        ]
    )
]

ret += [
    "",
    f".. currentmodule:: {project}",
    "",
    ".. autosummary::",
    "    :toctree: generated",
    "",
]

for module in modules:
    ret.append(f"    {module}")

(pathlib.Path(__file__).parent / "module.rst").write_text("\n".join(ret) + "\n")

# autogenerate cli.rst

data = tomllib.loads((pathlib.Path(__file__).parent / ".." / "pyproject.toml").read_text())
scripts = data["project"]["scripts"]

header = "Command-line tools"
ret = [
    "\n".join(
        [
            "*" * len(header),
            header,
            "*" * len(header),
        ]
    )
]

for name, funcname in scripts.items():
    modname, funcname = funcname.split(":")
    libname, modname = modname.split(".")
    funcname = re.split(r"(\_)(.*)(\_cli)", funcname)[2]
    parser = f"_{funcname}_parser"
    progname = f"{modname}_{funcname}"

    ret.append(
        "\n".join(
            [
                f".. _{modname}_{funcname}:",
                "",
                progname,
                "-" * len(progname),
                "",
                ".. argparse::",
                f"    :module: {libname}.{modname}",
                f"    :func: {parser}",
                f"    :prog: {progname}",
            ]
        )
    )

(pathlib.Path(__file__).parent / "cli.rst").write_text("\n\n".join(ret) + "\n")
