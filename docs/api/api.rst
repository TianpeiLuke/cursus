API Module
==========

.. currentmodule:: cursus.api

The API module provides the core directed acyclic graph structures for pipeline construction and dependency management. It enables automatic dependency resolution, proper step ordering, and collaborative pipeline development through a hierarchical approach with multiple DAG types.

Overview
--------

The Pipeline DAG module provides comprehensive graph structure representations for pipeline construction, enabling automatic dependency resolution, proper step ordering, and collaborative pipeline development. The module implements a hierarchical approach with three main DAG types:

- **Base DAG** for core graph operations
- **Enhanced DAG** for intelligent dependency resolution  
- **Workspace-aware DAG** for multi-developer collaboration

The module supports various edge types with confidence scoring, port-based dependency management, and cross-workspace dependency validation. It integrates seamlessly with the pipeline builder system to provide robust foundation for complex pipeline construction workflows.

Quick Start
-----------

Basic Pipeline DAG
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.api import PipelineDAG

   # Create pipeline DAG with dependencies
   dag = PipelineDAG()
   dag.add_node("data_loading")
   dag.add_node("preprocessing") 
   dag.add_edge("data_loading", "preprocessing")
   build_order = dag.topological_sort()

Enhanced DAG with Automatic Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.api.dag import EnhancedPipelineDAG

   # Create enhanced DAG with automatic resolution
   dag = EnhancedPipelineDAG()
   dag.add_step_with_spec("training", training_spec)
   dag.resolve_dependencies()

Workspace-Aware Collaboration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.api.dag import WorkspaceAwareDAG

   # Create workspace-aware DAG for collaboration
   dag = WorkspaceAwareDAG("team_alpha")
   dag.add_cross_workspace_dependency("shared_data", "team_beta")

Core Classes
------------

PipelineDAG
~~~~~~~~~~~

.. class:: PipelineDAG()

   Lightweight, efficient directed acyclic graph for pipeline construction with fast node and edge operations, topological sorting, and cycle detection.

   .. code-block:: python

      # Create pipeline DAG with dependencies
      dag = PipelineDAG()
      dag.add_node("data_loading")
      dag.add_node("preprocessing") 
      dag.add_edge("data_loading", "preprocessing")
      build_order = dag.topological_sort()

   .. method:: add_node(node_id)

      Add a node to the DAG.

      :param node_id: Unique identifier for the node
      :type node_id: str

      .. code-block:: python

         dag.add_node("training_step")

   .. method:: add_edge(from_node, to_node)

      Add a directed edge between two nodes.

      :param from_node: Source node identifier
      :type from_node: str
      :param to_node: Target node identifier
      :type to_node: str

      .. code-block:: python

         dag.add_edge("preprocessing", "training")

   .. method:: topological_sort()

      Perform topological sort using Kahn's algorithm.

      :returns: Topologically sorted list of node identifiers
      :rtype: List[str]

      .. code-block:: python

         execution_order = dag.topological_sort()

EdgeType
~~~~~~~~

.. class:: EdgeType(value)

   Enumeration of edge types with associated confidence scores for dependency resolution.

   :param value: Edge type identifier
   :type value: str

   .. code-block:: python

      # Access edge types with confidence scores
      edge_type = EdgeType.DEPENDENCY
      confidence = edge_type.confidence_score

DependencyEdge
~~~~~~~~~~~~~~

.. class:: DependencyEdge(from_node, to_node, edge_type=EdgeType.DEPENDENCY, confidence=None)

   Base class for typed pipeline dependencies with confidence scoring and validation.

   :param from_node: Source node identifier
   :type from_node: str
   :param to_node: Target node identifier
   :type to_node: str
   :param edge_type: Type of dependency edge
   :type edge_type: EdgeType
   :param confidence: Confidence score for auto-resolved edges
   :type confidence: Optional[float]

   .. code-block:: python

      # Create dependency edge with confidence
      edge = DependencyEdge("step1", "step2", EdgeType.DEPENDENCY, confidence=0.9)

ConditionalEdge
~~~~~~~~~~~~~~~

.. class:: ConditionalEdge(from_node, to_node, condition, confidence=None)

   Conditional dependency that activates based on boolean evaluation.

   :param from_node: Source node identifier
   :type from_node: str
   :param to_node: Target node identifier
   :type to_node: str
   :param condition: Boolean condition expression
   :type condition: str
   :param confidence: Confidence score for the condition
   :type confidence: Optional[float]

   .. code-block:: python

      # Create conditional dependency
      edge = ConditionalEdge("validation", "deployment", "accuracy > 0.95")

ParallelEdge
~~~~~~~~~~~~

.. class:: ParallelEdge(from_node, to_node, parallel_group, confidence=None)

   Parallel execution dependency for concurrent step execution.

   :param from_node: Source node identifier
   :type from_node: str
   :param to_node: Target node identifier
   :type to_node: str
   :param parallel_group: Parallel execution group identifier
   :type parallel_group: str
   :param confidence: Confidence score for parallel execution
   :type confidence: Optional[float]

   .. code-block:: python

      # Create parallel execution edge
      edge = ParallelEdge("feature_eng", "model_train", "training_group")

EdgeCollection
~~~~~~~~~~~~~~

.. class:: EdgeCollection()

   Collection management for pipeline edges with statistics and validation.

   .. code-block:: python

      # Manage edge collections
      collection = EdgeCollection()
      collection.add_edge(dependency_edge)
      stats = collection.get_statistics()

   .. method:: add_edge(edge)

      Add an edge to the collection.

      :param edge: Edge to add to the collection
      :type edge: DependencyEdge

      .. code-block:: python

         collection.add_edge(DependencyEdge("step1", "step2"))

   .. method:: get_statistics()

      Get statistical summary of edges in the collection.

      :returns: Statistics including edge counts and confidence scores
      :rtype: Dict[str, Any]

      .. code-block:: python

         stats = collection.get_statistics()

Enhanced DAG Classes
--------------------

EnhancedPipelineDAG
~~~~~~~~~~~~~~~~~~~

.. class:: EnhancedPipelineDAG()

   Enhanced DAG with port-based dependency management and intelligent resolution capabilities.

   .. code-block:: python

      # Create enhanced DAG with automatic resolution
      dag = EnhancedPipelineDAG()
      dag.add_step_with_spec("training", training_spec)
      dag.resolve_dependencies()

   .. method:: add_step_with_spec(step_name, step_spec)

      Add a step with its specification for intelligent dependency resolution.

      :param step_name: Name of the pipeline step
      :type step_name: str
      :param step_spec: Step specification with input/output ports
      :type step_spec: StepSpecification

      .. code-block:: python

         dag.add_step_with_spec("preprocessing", preprocessing_spec)

   .. method:: resolve_dependencies()

      Automatically resolve dependencies based on step specifications and port compatibility.

      :returns: List of resolved dependency edges with confidence scores
      :rtype: List[DependencyEdge]

      .. code-block:: python

         resolved_edges = dag.resolve_dependencies()

PipelineDAGResolver
~~~~~~~~~~~~~~~~~~~

.. class:: PipelineDAGResolver(dag)

   DAG resolution system with execution planning and configuration management.

   :param dag: Pipeline DAG to resolve
   :type dag: PipelineDAG

   .. code-block:: python

      # Create resolver with execution planning
      resolver = PipelineDAGResolver(pipeline_dag)
      execution_plan = resolver.create_execution_plan()

   .. method:: create_execution_plan(config=None)

      Create execution plan with configuration resolution and dependency validation.

      :param config: Pipeline configuration parameters
      :type config: Optional[Dict[str, Any]]
      :returns: Execution plan with resolved configurations
      :rtype: PipelineExecutionPlan

      .. code-block:: python

         plan = resolver.create_execution_plan({"batch_size": 32})

PipelineExecutionPlan
~~~~~~~~~~~~~~~~~~~~~

.. class:: PipelineExecutionPlan(execution_order, configurations)

   Execution plan with step ordering and resolved configurations.

   :param execution_order: Topologically sorted execution order
   :type execution_order: List[str]
   :param configurations: Resolved step configurations
   :type configurations: Dict[str, Any]

   .. code-block:: python

      # Access execution plan details
      plan = PipelineExecutionPlan(execution_order, step_configs)
      next_step = plan.get_next_step()

   .. method:: get_next_step()

      Get the next step in the execution sequence.

      :returns: Next step identifier or None if complete
      :rtype: Optional[str]

      .. code-block:: python

         next_step = plan.get_next_step()

Workspace Collaboration
-----------------------

WorkspaceAwareDAG
~~~~~~~~~~~~~~~~~

.. class:: WorkspaceAwareDAG(workspace_id)

   Multi-workspace DAG supporting cross-workspace dependencies and collaborative pipeline development.

   :param workspace_id: Unique workspace identifier
   :type workspace_id: str

   .. code-block:: python

      # Create workspace-aware DAG for collaboration
      dag = WorkspaceAwareDAG("team_alpha")
      dag.add_cross_workspace_dependency("shared_data", "team_beta")

   .. method:: add_cross_workspace_dependency(local_step, remote_workspace, remote_step=None)

      Add dependency on step from another workspace.

      :param local_step: Local step that depends on remote step
      :type local_step: str
      :param remote_workspace: Remote workspace identifier
      :type remote_workspace: str
      :param remote_step: Remote step identifier (defaults to local_step)
      :type remote_step: Optional[str]

      .. code-block:: python

         dag.add_cross_workspace_dependency("training", "data_team", "processed_data")

   .. method:: validate_cross_workspace_dependencies()

      Validate all cross-workspace dependencies for consistency and availability.

      :returns: Validation results for each cross-workspace dependency
      :rtype: Dict[str, bool]

      .. code-block:: python

         validation_results = dag.validate_cross_workspace_dependencies()

Utility Functions
-----------------

.. function:: create_execution_plan(dag, config=None)

   Create execution plan from DAG structure with configuration resolution.

   :param dag: Pipeline DAG to create execution plan from
   :type dag: PipelineDAG
   :param config: Pipeline configuration parameters
   :type config: Optional[Dict[str, Any]]
   :returns: Execution plan with resolved configurations
   :rtype: PipelineExecutionPlan

   .. code-block:: python

      plan = create_execution_plan(pipeline_dag, {"model_type": "xgboost"})

.. function:: validate_cross_workspace_dependencies(workspace_dag)

   Validate dependencies across workspaces for consistency and availability.

   :param workspace_dag: Workspace-aware DAG to validate
   :type workspace_dag: WorkspaceAwareDAG
   :returns: Validation results for each dependency
   :rtype: Dict[str, bool]

   .. code-block:: python

      results = validate_cross_workspace_dependencies(workspace_dag)

Usage Examples
--------------

Complex Pipeline Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.api import PipelineDAG
   from cursus.api.dag import DependencyEdge, EdgeType

   # Create complex pipeline with multiple dependencies
   dag = PipelineDAG()
   
   # Add all pipeline steps
   steps = ["data_ingestion", "data_validation", "feature_engineering", 
            "model_training", "model_evaluation", "model_deployment"]
   
   for step in steps:
       dag.add_node(step)
   
   # Add dependencies with different edge types
   dag.add_edge("data_ingestion", "data_validation")
   dag.add_edge("data_validation", "feature_engineering")
   dag.add_edge("feature_engineering", "model_training")
   dag.add_edge("model_training", "model_evaluation")
   dag.add_edge("model_evaluation", "model_deployment")
   
   # Get execution order
   execution_order = dag.topological_sort()
   print(f"Pipeline execution order: {execution_order}")

Intelligent Dependency Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.api.dag import EnhancedPipelineDAG
   from cursus.steps import StepSpecification

   # Create enhanced DAG with automatic dependency resolution
   dag = EnhancedPipelineDAG()
   
   # Define step specifications with input/output ports
   data_spec = StepSpecification(
       name="data_loading",
       outputs=["raw_data"]
   )
   
   preprocess_spec = StepSpecification(
       name="preprocessing",
       inputs=["raw_data"],
       outputs=["processed_data"]
   )
   
   training_spec = StepSpecification(
       name="training",
       inputs=["processed_data"],
       outputs=["trained_model"]
   )
   
   # Add steps with specifications
   dag.add_step_with_spec("data_loading", data_spec)
   dag.add_step_with_spec("preprocessing", preprocess_spec)
   dag.add_step_with_spec("training", training_spec)
   
   # Automatically resolve dependencies
   resolved_edges = dag.resolve_dependencies()
   print(f"Auto-resolved {len(resolved_edges)} dependencies")

Cross-Workspace Collaboration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.api.dag import WorkspaceAwareDAG

   # Create workspace-aware DAGs for different teams
   data_team_dag = WorkspaceAwareDAG("data_team")
   ml_team_dag = WorkspaceAwareDAG("ml_team")
   
   # Data team pipeline
   data_team_dag.add_node("data_collection")
   data_team_dag.add_node("data_cleaning")
   data_team_dag.add_edge("data_collection", "data_cleaning")
   
   # ML team depends on data team's output
   ml_team_dag.add_node("feature_engineering")
   ml_team_dag.add_node("model_training")
   ml_team_dag.add_cross_workspace_dependency(
       "feature_engineering", "data_team", "data_cleaning"
   )
   ml_team_dag.add_edge("feature_engineering", "model_training")
   
   # Validate cross-workspace dependencies
   validation_results = ml_team_dag.validate_cross_workspace_dependencies()
   print(f"Cross-workspace validation: {validation_results}")

Execution Planning
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.api.dag import PipelineDAGResolver, create_execution_plan

   # Create execution plan with configuration
   config = {
       "batch_size": 1000,
       "model_type": "xgboost",
       "validation_split": 0.2
   }
   
   # Method 1: Using resolver
   resolver = PipelineDAGResolver(pipeline_dag)
   execution_plan = resolver.create_execution_plan(config)
   
   # Method 2: Using utility function
   execution_plan = create_execution_plan(pipeline_dag, config)
   
   # Execute pipeline step by step
   while True:
       next_step = execution_plan.get_next_step()
       if next_step is None:
           break
       
       print(f"Executing step: {next_step}")
       # Execute the step...

API Reference
-------------

.. automodule:: cursus.api
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::
   :maxdepth: 2

   generated/cursus.api.dag
