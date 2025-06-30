# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "alps2qutipplus"
copyright = "2025, QILPCM-IFLP-CONICET"
author = "QILPCM-IFLP-CONICET"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "alabaster"
html_static_path = ["_static"]

# -- Autodoc settings --------------------------------------------------------

autoclass_content = "both"
autodoc_member_order = "bysource"

# -- Napoleon settings (for Google/NumPy docstrings) ------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Todo extension settings -------------------------------------------------

todo_include_todos = True

autosummary_generate = True
autodoc_member_order = "bysource"
add_module_names = False
