Pipeline Catalog Module
=======================

.. currentmodule:: cursus.pipeline_catalog

The Pipeline Catalog is a Zettelkasten-inspired pipeline organization system that implements flat structure and connection-based discovery principles. It provides atomic, independent pipeline implementations with enhanced metadata integration and sophisticated discovery mechanisms.

Overview
--------

The Pipeline Catalog represents a practical application of Zettelkasten knowledge management principles to software organization. By implementing atomicity, connectivity, and emergent organization, it creates a discoverable, maintainable, and scalable foundation for pipeline management that grows naturally with the system's evolution.

The catalog follows a three-tier structure based on Zettelkasten knowledge management principles:

1. **Atomic Pipeline Organization** - Flat structure with semantic naming and independence
2. **Connection-Based Discovery** - Registry-based relationships and bidirectional linking
3. **Enhanced Metadata Integration** - Type-safe metadata with MODS compatibility

Architecture
------------

Atomic Pipeline Organization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Flat Structure**: All pipelines stored in simple directories without deep hierarchies
- **Semantic Naming**: Self-documenting filenames following ``{framework}_{purpose}_{complexity}`` pattern
- **Independence**: Each pipeline is fully self-contained and can run standalone
- **Single Responsibility**: Each pipeline focuses on one coherent workflow concept

Connection-Based Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Connection Registry**: JSON-based registry mapping relationships between pipelines
- **Multiple Relationship Types**: alternatives, related, used_in connections
- **Bidirectional Linking**: Discovery from multiple entry points
- **Curated Connections**: Human-authored relationships capture semantic meaning

Enhanced Metadata Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **DAGMetadata Integration**: Type-safe metadata through existing DAGMetadata system
- **MODS Compatibility**: Enhanced metadata for MODS-compatible pipelines
- **Tag-Based Classification**: Multi-dimensional tagging for flexible organization
- **Registry Synchronization**: Automatic sync between pipeline metadata and registry

Key Components
--------------

Core Modules
~~~~~~~~~~~~

- **Main Entry Point**: Unified API with ``discover_pipelines`` and ``load_pipeline`` functions
- **Utils Module**: ``PipelineCatalogManager`` for advanced catalog operations
- **Catalog Index**: ``catalog_index.json`` connection registry and metadata store

Pipeline Collections
~~~~~~~~~~~~~~~~~~~~

- **Standard Pipelines**: ``pipelines/`` directory with atomic pipeline implementations
- **MODS Pipelines**: ``mods_pipelines/`` directory with MODS-compatible atomic pipelines
- **Shared DAGs**: ``shared_dags/`` directory with reusable DAG components and metadata utilities

Utility Modules
~~~~~~~~~~~~~~~

- **Catalog Registry**: Registry management and synchronization
- **Connection Traverser**: Connection navigation and path finding
- **Tag Discovery**: Tag-based search and filtering
- **Recommendation Engine**: Pipeline recommendations based on use cases
- **Registry Validator**: Registry validation and integrity checking

Quick Start
-----------

Basic Discovery
~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.pipeline_catalog import discover_pipelines, load_pipeline

   # Discover all available pipelines
   all_pipelines = discover_pipelines()
   print(f"Found {len(all_pipelines)} pipelines")

   # Load a specific pipeline
   pipeline_module = load_pipeline("xgb_training_simple")

Advanced Discovery with Manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.pipeline_catalog.utils import PipelineCatalogManager

   manager = PipelineCatalogManager()

   # Get recommendations for a use case
   recommendations = manager.get_recommendations("tabular_classification")

   # Find alternative approaches
   connections = manager.get_pipeline_connections("xgb_training_simple")
   alternatives = connections.get("alternatives", [])

   # Navigate connection paths
   path = manager.find_path("data_preprocessing", "model_evaluation")

Registry Operations
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Validate registry integrity
   validation_report = manager.validate_registry()

   # Get catalog statistics
   stats = manager.get_registry_stats()

   # Sync pipeline metadata
   from cursus.pipeline_catalog.shared_dags import EnhancedDAGMetadata
   success = manager.sync_pipeline(metadata, "my_pipeline.py")

Key Features
------------

Zettelkasten Principles Applied
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Atomicity**
   Each pipeline represents one atomic workflow concept with clear boundaries and interfaces.

**Connectivity**
   Explicit connections replace hierarchical positioning, enabling discovery through relationship traversal.

**Anti-Categories**
   Flat structure eliminates rigid framework hierarchies, using tag-based classification instead.

**Manual Linking**
   Curated connections between related pipelines provide structured navigation paths.

**Dual-Form Structure**
   Separation between organizational metadata (outer form) and pipeline implementation (inner form).

Discovery Mechanisms
~~~~~~~~~~~~~~~~~~~~

- **Framework-based**: Find pipelines by ML framework (XGBoost, PyTorch, etc.)
- **Task-based**: Discover by purpose (training, evaluation, preprocessing, etc.)
- **Complexity-based**: Filter by sophistication level (simple, standard, comprehensive)
- **Tag-based**: Multi-dimensional search using framework, task, domain, and pattern tags
- **Connection-based**: Navigate through alternative, related, and composition relationships
- **Use-case driven**: Get recommendations based on specific ML use cases

MODS Integration
~~~~~~~~~~~~~~~~

- **Enhanced Metadata**: MODS-compatible pipelines include additional operational metadata
- **Template Decoration**: Automatic MODS template decoration for enhanced tracking
- **Registry Integration**: MODS metadata automatically synced to connection registry
- **Operational Capabilities**: Support for MODS dashboard integration and governance features

Core Functions
--------------

discover_pipelines
~~~~~~~~~~~~~~~~~~

.. function:: discover_pipelines()

   Discover all available pipeline modules in the catalog.

   :returns: List of pipeline module names
   :rtype: List[str]

   .. code-block:: python

      from cursus.pipeline_catalog import discover_pipelines

      # Discover all available pipelines
      all_pipelines = discover_pipelines()
      print(f"Found {len(all_pipelines)} pipelines")

load_pipeline
~~~~~~~~~~~~~

.. function:: load_pipeline(pipeline_id)

   Load a specific pipeline by its identifier.

   :param pipeline_id: Unique identifier for the pipeline
   :type pipeline_id: str
   :returns: Pipeline function or class
   :rtype: Callable
   :raises: PipelineNotFoundError if pipeline doesn't exist

   .. code-block:: python

      # Load specific pipeline
      pipeline_func = load_pipeline("xgb_training_simple")
      
      # Execute pipeline
      result = pipeline_func(config=my_config)

PipelineCatalogManager
~~~~~~~~~~~~~~~~~~~~~~

.. class:: PipelineCatalogManager(catalog_path=None, auto_sync=True)

   Advanced catalog management with connection traversal, recommendations, and registry operations.

   :param catalog_path: Path to catalog directory
   :type catalog_path: Optional[Union[str, Path]]
   :param auto_sync: Whether to automatically sync metadata changes
   :type auto_sync: bool

   .. code-block:: python

      from cursus.pipeline_catalog.utils import PipelineCatalogManager

      # Initialize manager
      manager = PipelineCatalogManager()

      # Advanced operations
      recommendations = manager.get_recommendations("tabular_classification")
      connections = manager.get_pipeline_connections("my_pipeline")

   .. method:: get_recommendations(use_case, max_results=10)

      Get pipeline recommendations for a specific use case.

      :param use_case: Use case description or identifier
      :type use_case: str
      :param max_results: Maximum number of recommendations
      :type max_results: int
      :returns: List of recommended pipeline identifiers with scores
      :rtype: List[Dict[str, Any]]

      .. code-block:: python

         # Get recommendations for tabular classification
         recommendations = manager.get_recommendations("tabular_classification")
         
         for rec in recommendations:
             print(f"Pipeline: {rec['pipeline_id']}, Score: {rec['score']}")

   .. method:: get_pipeline_connections(pipeline_id)

      Get all connections for a specific pipeline.

      :param pipeline_id: Pipeline identifier
      :type pipeline_id: str
      :returns: Dictionary of connection types and related pipelines
      :rtype: Dict[str, List[str]]

      .. code-block:: python

         connections = manager.get_pipeline_connections("xgb_training_simple")
         
         print(f"Alternatives: {connections.get('alternatives', [])}")
         print(f"Related: {connections.get('related', [])}")
         print(f"Used in: {connections.get('used_in', [])}")

   .. method:: find_path(source_pipeline, target_pipeline, max_depth=5)

      Find connection path between two pipelines.

      :param source_pipeline: Source pipeline identifier
      :type source_pipeline: str
      :param target_pipeline: Target pipeline identifier
      :type target_pipeline: str
      :param max_depth: Maximum search depth
      :type max_depth: int
      :returns: List of pipeline identifiers forming the path
      :rtype: Optional[List[str]]

      .. code-block:: python

         # Find path from preprocessing to evaluation
         path = manager.find_path("data_preprocessing", "model_evaluation")
         
         if path:
             print(f"Connection path: {' -> '.join(path)}")

   .. method:: validate_registry()

      Validate registry integrity and consistency.

      :returns: Validation report with issues and recommendations
      :rtype: Dict[str, Any]

      .. code-block:: python

         report = manager.validate_registry()
         
         if report['is_valid']:
             print("Registry is valid")
         else:
             print(f"Issues found: {report['issues']}")

   .. method:: get_registry_stats()

      Get comprehensive statistics about the catalog registry.

      :returns: Statistics including pipeline counts, connection types, and coverage metrics
      :rtype: Dict[str, Any]

      .. code-block:: python

         stats = manager.get_registry_stats()
         
         print(f"Total pipelines: {stats['total_pipelines']}")
         print(f"Connection types: {stats['connection_types']}")
         print(f"Framework coverage: {stats['framework_coverage']}")

   .. method:: sync_pipeline(metadata, pipeline_path)

      Synchronize pipeline metadata with the registry.

      :param metadata: Pipeline metadata object
      :type metadata: EnhancedDAGMetadata
      :param pipeline_path: Path to pipeline file
      :type pipeline_path: str
      :returns: True if sync successful, False otherwise
      :rtype: bool

      .. code-block:: python

         from cursus.pipeline_catalog.shared_dags import EnhancedDAGMetadata
         
         metadata = EnhancedDAGMetadata(
             pipeline_id="my_custom_pipeline",
             framework="pytorch",
             task="training"
         )
         
         success = manager.sync_pipeline(metadata, "my_pipeline.py")

Usage Examples
--------------

Framework-Based Discovery with Manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.pipeline_catalog import discover_pipelines, load_pipeline
   from cursus.pipeline_catalog.utils import PipelineCatalogManager

   # Discover all available pipelines
   all_pipelines = discover_pipelines()
   print(f"Found {len(all_pipelines)} pipelines")

   # Use manager for framework-based discovery
   manager = PipelineCatalogManager()
   xgb_pipelines = manager.discover_pipelines(framework="xgboost")
   print(f"Found {len(xgb_pipelines)} XGBoost pipelines")

   # Load and execute a training pipeline
   training_module = load_pipeline("xgb_training_comprehensive")

Task-Based Discovery with Manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.pipeline_catalog.utils import PipelineCatalogManager

   manager = PipelineCatalogManager()

   # Find pipelines by complexity
   simple_pipelines = manager.discover_pipelines(complexity="simple")

   # Find pipelines by tags
   tabular_pipelines = manager.discover_pipelines(tags=["tabular"])

   # Find pipelines by use case
   classification_pipelines = manager.discover_pipelines(use_case="classification")

Connection-Based Navigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.pipeline_catalog.utils import PipelineCatalogManager

   manager = PipelineCatalogManager()

   # Start with a training pipeline
   training_pipeline = "xgb_training_simple"

   # Find related evaluation pipelines
   connections = manager.get_pipeline_connections(training_pipeline)
   eval_alternatives = connections.get("related", [])

   # Navigate to evaluation
   for eval_pipeline in eval_alternatives:
       if "evaluation" in eval_pipeline:
           eval_func = load_pipeline(eval_pipeline)
           break

Use-Case Driven Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get recommendations for specific use cases
   tabular_recs = manager.get_recommendations("tabular_classification")
   nlp_recs = manager.get_recommendations("text_classification")
   cv_recs = manager.get_recommendations("image_classification")

   # Select best recommendation
   best_pipeline = tabular_recs[0]["pipeline_id"]
   pipeline_func = load_pipeline(best_pipeline)

Registry Management
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Validate entire registry
   validation_report = manager.validate_registry()
   
   if not validation_report["is_valid"]:
       print("Registry issues found:")
       for issue in validation_report["issues"]:
           print(f"  - {issue}")
   
   # Get comprehensive statistics
   stats = manager.get_registry_stats()
   print(f"Registry contains {stats['total_pipelines']} pipelines")
   print(f"Frameworks covered: {', '.join(stats['frameworks'])}")

Benefits
--------

Reduced Complexity
~~~~~~~~~~~~~~~~~~

- **60% reduction** in navigation complexity (from 5-level to 2-level depth)
- Simplified mental model for users
- Easier maintenance and updates

Enhanced Discoverability
~~~~~~~~~~~~~~~~~~~~~~~~

- Multiple access paths through different criteria
- Connection-based navigation for exploring relationships
- Tag-based filtering for precise searches
- Use-case-driven recommendations

Improved Maintainability
~~~~~~~~~~~~~~~~~~~~~~~~

- Atomic organization with clear boundaries
- Independent versioning and updates
- Explicit relationship documentation
- Automated validation capabilities

Scalable Growth
~~~~~~~~~~~~~~~

- Organic expansion without structural changes
- Framework-agnostic foundation
- Extensible metadata schema
- Tool-friendly structure

Performance Characteristics
---------------------------

- **Sub-second discovery** for 100+ pipelines
- **Efficient connection traversal** through optimized graph algorithms
- **Minimal memory footprint** with lazy loading
- **Cached operations** for repeated queries

Integration Points
------------------

MODS Ecosystem Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

- MODS template decoration for enhanced tracking
- Operational capabilities and dashboard integration
- Compliance and governance features
- GitFarm integration for version control

DAG Compiler Integration
~~~~~~~~~~~~~~~~~~~~~~~~

- Enhanced DAGMetadata integration
- Type-safe metadata validation
- Automatic registry synchronization
- Template lifecycle management

Validation Framework Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Pipeline validation through registry
- Connection consistency checking
- Metadata completeness verification
- Automated quality assurance

API Reference
-------------

.. automodule:: cursus.pipeline_catalog
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::
   :maxdepth: 2

   generated/cursus.pipeline_catalog.utils
   generated/cursus.pipeline_catalog.shared_dags
   generated/cursus.pipeline_catalog.pipelines
   generated/cursus.pipeline_catalog.mods_pipelines
