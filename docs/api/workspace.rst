Workspace Module
================

.. currentmodule:: cursus.workspace

The Workspace module provides comprehensive workspace management system with isolated development environments, cross-workspace collaboration, and unified workspace operations for the Cursus pipeline system.

Overview
--------

The Workspace Management module provides a complete solution for managing isolated developer workspaces, enabling collaborative pipeline development while maintaining strict isolation boundaries. The system supports workspace lifecycle management, template-based workspace creation, cross-workspace dependency resolution, and comprehensive validation and monitoring capabilities.

The module implements a hierarchical architecture with a high-level unified API, specialized core managers for different functional areas, and comprehensive validation systems. It supports multiple workspace types including developer workspaces, shared workspaces, and test environments, with full integration into the Cursus pipeline ecosystem.

Quick Start
-----------

Basic Workspace Setup
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.workspace import WorkspaceAPI

   # Initialize API
   api = WorkspaceAPI()

   # Set up developer workspace
   result = api.setup_developer_workspace("alice", template="basic")
   if result.success:
       print(f"Workspace created at: {result.workspace_path}")
       
       # Validate the new workspace
       report = api.validate_workspace(result.workspace_path)
       print(f"Validation status: {report.status}")

ML Pipeline Workspace
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create ML pipeline workspace
   result = api.setup_developer_workspace(
       "data_scientist_1", 
       template="ml_pipeline",
       config_overrides={
           "enable_gpu": True,
           "data_path": "/shared/datasets"
       }
   )

   # List all workspaces
   workspaces = api.list_workspaces()
   ml_workspaces = [w for w in workspaces if "ml" in w.path.name]
   print(f"Found {len(ml_workspaces)} ML workspaces")

Core API Classes
----------------

WorkspaceAPI
~~~~~~~~~~~~

.. class:: WorkspaceAPI(base_path=None)

   Unified high-level API for workspace-aware system operations, providing a simplified interface to the workspace system and abstracting the complexity of underlying managers.

   :param base_path: Base path for workspace operations (defaults to "development")
   :type base_path: Optional[Union[str, Path]]

   .. code-block:: python

      from cursus.workspace import WorkspaceAPI

      # Initialize the unified API
      api = WorkspaceAPI()

      # Developer operations
      result = api.setup_developer_workspace("developer_1", "ml_template")
      pipeline = api.build_cross_workspace_pipeline(pipeline_spec)
      report = api.validate_workspace_components("developer_1")

   .. method:: setup_developer_workspace(developer_id, template=None, config_overrides=None)

      Set up a new developer workspace with optional template and configuration overrides.

      :param developer_id: Unique identifier for the developer
      :type developer_id: str
      :param template: Optional template to use for workspace setup
      :type template: Optional[str]
      :param config_overrides: Optional configuration overrides
      :type config_overrides: Optional[Dict[str, Any]]
      :returns: Setup result with workspace details and any warnings
      :rtype: WorkspaceSetupResult

      .. code-block:: python

         # Create basic developer workspace
         result = api.setup_developer_workspace("alice")

         # Create ML workspace with template
         result = api.setup_developer_workspace(
             "bob", 
             template="ml_pipeline",
             config_overrides={"enable_gpu": True}
         )

   .. method:: validate_workspace(workspace_path)

      Validate a workspace for compliance and isolation violations.

      :param workspace_path: Path to the workspace to validate
      :type workspace_path: Union[str, Path]
      :returns: Validation results with status, issues, and recommendations
      :rtype: ValidationReport

      .. code-block:: python

         # Validate workspace
         report = api.validate_workspace("/path/to/workspace")
         if report.status == WorkspaceStatus.ERROR:
             print(f"Validation errors: {report.issues}")

   .. method:: list_workspaces()

      List all available workspaces with their current status and information.

      :returns: List of workspace information objects
      :rtype: List[WorkspaceInfo]

      .. code-block:: python

         workspaces = api.list_workspaces()
         for workspace in workspaces:
             print(f"{workspace.developer_id}: {workspace.status}")

   .. method:: promote_workspace_artifacts(workspace_path, target_environment="staging")

      Promote artifacts from a workspace to target environment.

      :param workspace_path: Path to the source workspace
      :type workspace_path: Union[str, Path]
      :param target_environment: Target environment (staging, production, etc.)
      :type target_environment: str
      :returns: Promotion results with promoted artifacts list
      :rtype: PromotionResult

      .. code-block:: python

         # Promote to staging
         result = api.promote_workspace_artifacts("/path/to/workspace", "staging")
         print(f"Promoted {len(result.artifacts_promoted)} artifacts")

   .. method:: get_system_health()

      Get overall system health report including all workspace validation results.

      :returns: System-wide health information with workspace reports
      :rtype: HealthReport

      .. code-block:: python

         health = api.get_system_health()
         print(f"Overall status: {health.overall_status}")
         for report in health.workspace_reports:
             print(f"Workspace {report.workspace_path}: {report.status}")

   .. method:: cleanup_workspaces(inactive_days=30, dry_run=True)

      Clean up inactive workspaces based on inactivity threshold.

      :param inactive_days: Number of days of inactivity before cleanup
      :type inactive_days: int
      :param dry_run: If True, only report what would be cleaned
      :type dry_run: bool
      :returns: Cleanup results with cleaned workspaces and errors
      :rtype: CleanupReport

      .. code-block:: python

         # Dry run cleanup
         report = api.cleanup_workspaces(inactive_days=60, dry_run=True)
         print(f"Would clean {len(report.cleaned_workspaces)} workspaces")

         # Actual cleanup
         report = api.cleanup_workspaces(inactive_days=60, dry_run=False)

WorkspaceSetupResult
~~~~~~~~~~~~~~~~~~~~

.. class:: WorkspaceSetupResult(success, workspace_path, developer_id, message, warnings=[])

   Result of workspace setup operation with success status and details.

   :param success: Whether setup was successful
   :type success: bool
   :param workspace_path: Path to the created workspace
   :type workspace_path: Path
   :param developer_id: Unique identifier for the developer
   :type developer_id: str
   :param message: Setup result message
   :type message: str
   :param warnings: List of warnings encountered during setup
   :type warnings: List[str]

   .. code-block:: python

      result = api.setup_developer_workspace("developer_1")
      if result.success:
          print(f"Workspace created at: {result.workspace_path}")
          if result.warnings:
              print(f"Warnings: {result.warnings}")

ValidationReport
~~~~~~~~~~~~~~~~

.. class:: ValidationReport(workspace_path, status, issues=[], recommendations=[], isolation_violations=[])

   Workspace validation report with status, issues, and recommendations.

   :param workspace_path: Path to the validated workspace
   :type workspace_path: Path
   :param status: Validation status (HEALTHY, WARNING, ERROR, UNKNOWN)
   :type status: WorkspaceStatus
   :param issues: List of validation issues found
   :type issues: List[str]
   :param recommendations: List of recommendations for fixing issues
   :type recommendations: List[str]
   :param isolation_violations: Detailed isolation violation information
   :type isolation_violations: List[Dict[str, Any]]

   .. code-block:: python

      report = api.validate_workspace("workspace_path")
      if report.status == WorkspaceStatus.WARNING:
          for issue in report.issues:
              print(f"Issue: {issue}")
          for rec in report.recommendations:
              print(f"Recommendation: {rec}")

Template System
---------------

WorkspaceTemplate
~~~~~~~~~~~~~~~~~

.. class:: WorkspaceTemplate(name, type, description="", directories=[], files={}, dependencies=[], config_overrides={}, created_at=None, version="1.0.0")

   Workspace template configuration defining structure, files, and dependencies.

   :param name: Template name
   :type name: str
   :param type: Type of template (BASIC, ML_PIPELINE, DATA_PROCESSING, CUSTOM)
   :type type: TemplateType
   :param description: Template description
   :type description: str
   :param directories: Directories to create
   :type directories: List[str]
   :param files: Files to create with content
   :type files: Dict[str, str]
   :param dependencies: Required dependencies
   :type dependencies: List[str]
   :param config_overrides: Default configuration
   :type config_overrides: Dict[str, Any]
   :param created_at: Template creation timestamp
   :type created_at: Optional[str]
   :param version: Template version
   :type version: str

   .. code-block:: python

      from cursus.workspace.templates import WorkspaceTemplate, TemplateType

      # Create custom template
      template = WorkspaceTemplate(
          name="custom_ml",
          type=TemplateType.ML_PIPELINE,
          description="Custom ML pipeline template",
          directories=["data", "models", "notebooks"],
          files={"README.md": "# Custom ML Workspace"},
          dependencies=["pandas", "scikit-learn"]
      )

TemplateManager
~~~~~~~~~~~~~~~

.. class:: TemplateManager(templates_dir=None)

   Manages workspace templates including built-in and custom templates.

   :param templates_dir: Directory containing template definitions
   :type templates_dir: Optional[Path]

   .. code-block:: python

      from cursus.workspace.templates import TemplateManager

      # Initialize template manager
      manager = TemplateManager()

      # List available templates
      templates = manager.list_templates()
      for template in templates:
          print(f"{template.name}: {template.description}")

   .. method:: get_template(name)

      Get a template by name from the template directory.

      :param name: Template name
      :type name: str
      :returns: Template if found, None otherwise
      :rtype: Optional[WorkspaceTemplate]

      .. code-block:: python

         template = manager.get_template("ml_pipeline")
         if template:
             print(f"Template: {template.description}")

   .. method:: list_templates()

      List all available templates in the template directory.

      :returns: List of available templates
      :rtype: List[WorkspaceTemplate]

      .. code-block:: python

         templates = manager.list_templates()
         print(f"Found {len(templates)} templates")

   .. method:: create_template(template)

      Create a new template and save it to the template directory.

      :param template: Template to create
      :type template: WorkspaceTemplate
      :returns: True if successful, False otherwise
      :rtype: bool

      .. code-block:: python

         success = manager.create_template(custom_template)
         if success:
             print("Template created successfully")

   .. method:: apply_template(template_name, workspace_path)

      Apply a template to a workspace directory.

      :param template_name: Name of template to apply
      :type template_name: str
      :param workspace_path: Path to workspace directory
      :type workspace_path: Path
      :returns: True if successful, False otherwise
      :rtype: bool

      .. code-block:: python

         success = manager.apply_template("ml_pipeline", Path("workspace"))
         if success:
             print("Template applied successfully")

Core Management
---------------

WorkspaceManager
~~~~~~~~~~~~~~~~

.. class:: WorkspaceManager(workspace_root=None, config_file=None, auto_discover=True)

   Centralized workspace management with functional separation through specialized managers.

   :param workspace_root: Root directory for workspaces
   :type workspace_root: Optional[Union[str, Path]]
   :param config_file: Path to workspace configuration file
   :type config_file: Optional[Union[str, Path]]
   :param auto_discover: Whether to automatically discover workspaces
   :type auto_discover: bool

   .. code-block:: python

      from cursus.workspace.core.manager import WorkspaceManager

      # Initialize workspace manager
      manager = WorkspaceManager("/path/to/workspaces")

      # Create workspace
      context = manager.create_workspace("developer_1", template="basic")

   .. method:: create_workspace(developer_id, workspace_type="developer", template=None, **kwargs)

      Create a new workspace with specified type and optional template.

      :param developer_id: Developer identifier for the workspace
      :type developer_id: str
      :param workspace_type: Type of workspace ("developer", "shared", "test")
      :type workspace_type: str
      :param template: Optional template to use for workspace creation
      :type template: str
      :param kwargs: Additional arguments passed to lifecycle manager
      :returns: Context for the created workspace
      :rtype: WorkspaceContext

      .. code-block:: python

         # Create developer workspace
         context = manager.create_workspace("alice", "developer", "ml_pipeline")
         print(f"Created workspace: {context.workspace_id}")

   .. method:: discover_components(workspace_ids=None, developer_id=None)

      Discover components across workspaces with optional filtering.

      :param workspace_ids: Optional list of workspace IDs to search
      :type workspace_ids: Optional[List[str]]
      :param developer_id: Optional specific developer ID to search
      :type developer_id: Optional[str]
      :returns: Dictionary containing discovered components
      :rtype: Dict[str, Any]

      .. code-block:: python

         # Discover all components
         components = manager.discover_components()

         # Discover for specific developer
         components = manager.discover_components(developer_id="alice")

   .. method:: validate_workspace_structure(workspace_root=None, strict=False)

      Validate workspace structure for compliance and best practices.

      :param workspace_root: Root directory to validate
      :type workspace_root: Optional[Union[str, Path]]
      :param strict: Whether to apply strict validation rules
      :type strict: bool
      :returns: Tuple of (is_valid, list_of_issues)
      :rtype: Tuple[bool, List[str]]

      .. code-block:: python

         is_valid, issues = manager.validate_workspace_structure(strict=True)
         if not is_valid:
             for issue in issues:
                 print(f"Issue: {issue}")

Usage Examples
--------------

Template Management
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.workspace.templates import TemplateManager, WorkspaceTemplate, TemplateType

   # Initialize template manager
   manager = TemplateManager()

   # Create custom template
   custom_template = WorkspaceTemplate(
       name="data_science",
       type=TemplateType.ML_PIPELINE,
       description="Data science workspace with Jupyter and visualization tools",
       directories=["data", "notebooks", "models", "reports"],
       files={
           "README.md": "# Data Science Workspace",
           "requirements.txt": "pandas\nnumpy\njupyter\nmatplotlib\nseaborn"
       },
       dependencies=["pandas", "numpy", "jupyter", "matplotlib", "seaborn"]
   )

   # Save template
   manager.create_template(custom_template)

   # Apply to workspace
   manager.apply_template("data_science", Path("workspace"))

Cross-Workspace Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Discover components across workspaces
   components = api.workspace_manager.discover_components()
   print(f"Found components: {list(components.keys())}")

   # Resolve cross-workspace dependencies
   pipeline_def = {
       "steps": [
           {"name": "data_prep", "workspace": "alice"},
           {"name": "training", "workspace": "bob", "depends_on": ["data_prep"]}
       ]
   }

   resolved = api.workspace_manager.resolve_cross_workspace_dependencies(pipeline_def)
   print(f"Resolved dependencies: {resolved}")

System Health Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get system health
   health = api.get_system_health()
   print(f"Overall status: {health.overall_status}")

   # Check individual workspace health
   for report in health.workspace_reports:
       if report.status != WorkspaceStatus.HEALTHY:
           print(f"Workspace {report.workspace_path} has issues:")
           for issue in report.issues:
               print(f"  - {issue}")
           for rec in report.recommendations:
               print(f"  Recommendation: {rec}")

Workspace Cleanup
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Dry run cleanup to see what would be cleaned
   cleanup_report = api.cleanup_workspaces(inactive_days=30, dry_run=True)
   print(f"Would clean {len(cleanup_report.cleaned_workspaces)} workspaces")

   # Actual cleanup
   if len(cleanup_report.cleaned_workspaces) > 0:
       cleanup_report = api.cleanup_workspaces(inactive_days=30, dry_run=False)
       print(f"Cleaned {len(cleanup_report.cleaned_workspaces)} workspaces")

Integration Points
------------------

Pipeline System Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The workspace system integrates with the Cursus pipeline system for component discovery, dependency resolution, and pipeline execution across workspace boundaries.

Validation Framework Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive integration with the validation framework for workspace structure validation, isolation checking, and compliance monitoring.

CLI Integration
~~~~~~~~~~~~~~~

Full integration with the Cursus CLI for workspace management commands, validation operations, and administrative tasks.

Utility Functions
-----------------

.. function:: get_default_config()

   Get default configuration for workspace API with standard settings.

   :returns: Default configuration dictionary
   :rtype: Dict[str, Any]

   .. code-block:: python

      from cursus.workspace import get_default_config

      config = get_default_config()
      print(f"Default workspace root: {config['workspace_root']}")

API Reference
-------------

.. automodule:: cursus.workspace
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::
   :maxdepth: 2

   generated/cursus.workspace.api
   generated/cursus.workspace.templates
   generated/cursus.workspace.core
   generated/cursus.workspace.validation
   generated/cursus.workspace.quality
