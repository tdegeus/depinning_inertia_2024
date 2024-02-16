import os
import sys

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
