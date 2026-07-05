"""
Sphinx configuration for the Cursus documentation.

Design (see slipbox/2_project_planning/2026-07-05_comprehensive_code_documentation_publishing_plan.md):
- Recursive ``autosummary`` over the single ``cursus`` root -> zero hand-maintained API .rst.
- ``sphinx-click`` renders the whole CLI from the live ``cursus.cli:cli`` group.
- ``_ext/gen_reference.py`` (wired to ``builder-inited``) emits the MCP-tool, step-interface,
  and pipeline-catalog reference pages from their live self-describing sources, so those pages
  never drift from the code.
- Google-style docstrings via napoleon; heavy runtime deps mocked so a docs-only environment
  (e.g. Read the Docs) can import ``cursus`` without the full ML stack.
"""

import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath("_ext"))

# -- Version (single source of truth: the installed package) -----------------
try:
    import cursus

    version = cursus.__version__
    release = version
except Exception:  # pragma: no cover - docs must still build if import degrades
    version = "unknown"
    release = "unknown"

# -- Project information -----------------------------------------------------
project = "Cursus"
copyright = "2026, Tianpei Xie"
author = "Tianpei Xie"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",       # pull docstrings
    "sphinx.ext.autosummary",   # recursive API tree
    "sphinx.ext.napoleon",      # Google/NumPy docstrings
    "sphinx.ext.viewcode",      # [source] links
    "sphinx.ext.intersphinx",   # cross-project links
    "sphinx.ext.doctest",       # doctest blocks (opt-in per page)
    "sphinx.ext.todo",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "sphinx_click",             # CLI reference from the live Click app
    "myst_parser",              # Markdown (MyST) authoring
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
language = "en"

# -- HTML output (Furo) ------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = f"Cursus {version}"
html_theme_options = {
    "source_repository": "https://github.com/TianpeiLuke/cursus/",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
}

# -- autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
# Do NOT re-document the many symbols re-exported via package __init__ files
# (they are documented once on their defining module).
autosummary_imported_members = False
add_module_names = False

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Heavy runtime deps mocked so ``import cursus`` succeeds in a docs-only env.
# When an upstream sync adds a new heavy dependency, add it here (release checklist).
autodoc_mock_imports = [
    "boto3",
    "botocore",
    "sagemaker",
    "torch",
    "pytorch_lightning",
    "lightning",
    "transformers",
    "xgboost",
    "lightgbm",
    "sklearn",
    "scipy",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "plotly",
    "networkx",
    "pyarrow",
    "gensim",
    "spacy",
    "nltk",
    "bs4",
    "tqdm",
]

# -- napoleon ----------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

# -- MyST --------------------------------------------------------------------
myst_enable_extensions = [
    "deflist",
    "tasklist",
    "colon_fence",
    "linkify",
    "substitution",
]
myst_heading_anchors = 3

# -- misc --------------------------------------------------------------------
todo_include_todos = True
rst_prolog = """
.. |project| replace:: Cursus
"""


def setup(app):
    """Wire custom CSS and the build-time reference generator."""
    app.add_css_file("custom.css")
    try:
        import gen_reference

        gen_reference.register(app)
    except Exception as exc:  # pragma: no cover - never fail the build on the generator
        import logging

        logging.getLogger("sphinx.cursus").warning(
            "gen_reference not wired (generated reference pages skipped): %s", exc
        )
