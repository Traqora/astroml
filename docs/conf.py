"""Sphinx configuration for AstroML documentation."""

import os
import sys

# Make the package importable without installing it
sys.path.insert(0, os.path.abspath(".."))

# -- Project info -------------------------------------------------------------
project = "AstroML"
author = "AstroML Contributors"
copyright = "2026, AstroML Contributors"
release = "1.0.0"

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # Pull docstrings from source
    "sphinx.ext.napoleon",  # Google / NumPy style docstrings
    "sphinx.ext.viewcode",  # Add [source] links
    "sphinx.ext.autosummary",  # Auto-generate summary tables
    "sphinx.ext.intersphinx",  # Cross-link to Python / NumPy / PyTorch docs
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output --------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
}
