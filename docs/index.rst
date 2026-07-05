Cursus
======

.. image:: https://img.shields.io/pypi/v/cursus.svg
   :target: https://pypi.org/project/cursus/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/cursus.svg
   :target: https://pypi.org/project/cursus/
   :alt: Python versions

.. image:: https://img.shields.io/github/license/TianpeiLuke/cursus.svg
   :target: https://github.com/TianpeiLuke/cursus/blob/main/LICENSE
   :alt: License

**Cursus** turns a pipeline *graph* plus a JSON *configuration* into a complete,
production-ready **Amazon SageMaker pipeline** — resolving inter-step dependencies,
wiring inputs and outputs, and generating the SageMaker step objects automatically.
You describe *what* the pipeline is; Cursus figures out *how* to build it.

- **Graph-to-pipeline automation** — a DAG of step names compiles to a full SageMaker pipeline.
- **Intelligent dependency resolution** — step connections and data flow are inferred, not hand-wired.
- **Declarative, data-driven steps** — every step is one ``<step>.step.yaml`` interface; builders are synthesized at runtime.
- **A pre-built pipeline catalog** — 44 validated DAGs across 8 frameworks.
- **Agent-ready** — a framework-neutral 70-tool MCP surface mirrors the CLI/API for LLM agents.

.. code-block:: bash

   pip install cursus
   cursus compile -d my_dag.json -c my_config.json -o pipeline.json

New here? Start with :doc:`getting_started/index`.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/index
   getting_started/installation
   getting_started/quickstart
   getting_started/core_concepts

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Concepts & Guides

   concepts/index
   guides/index
   migration/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   cli
   reference/index

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
