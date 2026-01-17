"""
Pipeline assembler that builds pipelines from a DAG structure and step builders.

This assembler leverages the specification-based dependency resolution system
to intelligently connect steps and build complete SageMaker pipelines.
"""

from typing import Dict, List, Any, Optional, Type, Set, Tuple, DefaultDict
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import Step
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession
from pathlib import Path
import logging
import time
import traceback
from collections import defaultdict

from ..base import BasePipelineConfig, StepBuilderBase
from ..deps.registry_manager import RegistryManager
from ..deps.dependency_resolver import (
    UnifiedDependencyResolver,
    create_dependency_resolver,
)
from ..deps.factory import create_pipeline_components
from ..deps.property_reference import PropertyReference
from ..base import OutputSpec
from ...registry.step_names import CONFIG_STEP_REGISTRY
from ...step_catalog import StepCatalog

from ...api.dag.base_dag import PipelineDAG


logger = logging.getLogger(__name__)


def safe_value_for_logging(value: Any) -> str:
    """
    Safely format a value for logging, handling Pipeline variables appropriately.

    Args:
        value: Any value that might be a Pipeline variable

    Returns:
        A string representation safe for logging
    """
    # Check if it's a Pipeline variable or has the expr attribute
    if hasattr(value, "expr"):
        return f"[Pipeline Variable: {value.__class__.__name__}]"

    # Handle collections containing Pipeline variables
    if isinstance(value, dict):
        return "{...}"  # Avoid iterating through dict values which might contain Pipeline variables
    if isinstance(value, (list, tuple, set)):
        return f"[{type(value).__name__} with {len(value)} items]"

    # For simple values, return the string representation
    try:
        return str(value)
    except Exception:
        return f"[Object of type: {type(value).__name__}]"


class PipelineAssembler:
    """
    Assembles pipeline steps using a DAG and step builders with specification-based dependency resolution.

    This class implements a component-based approach to building SageMaker Pipelines,
    leveraging the specification-based dependency resolution system to simplify
    the code and improve maintainability.

    The assembler follows these steps to build a pipeline:
    1. Initialize step builders for all steps in the DAG
    2. Determine the build order using topological sort
    3. Propagate messages between steps using the dependency resolver
    4. Instantiate steps in topological order, delegating input/output handling to builders
    5. Create the pipeline with the instantiated steps

    This approach allows for a flexible and modular pipeline definition, where
    each step is responsible for its own configuration and input/output handling.
    """

    # Note: cradle_loading_requests removed as part of Phase 2 cleanup
    # Cradle data loading requests are now handled by the standalone execution document generator
    # (CradleDataLoadingHelper in cursus.mods.exe_doc.cradle_helper)

    def __init__(
        self,
        dag: PipelineDAG,
        config_map: Dict[str, BasePipelineConfig],
        step_catalog: Optional[StepCatalog] = None,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        pipeline_parameters: Optional[List[ParameterString]] = None,
        registry_manager: Optional[RegistryManager] = None,
        dependency_resolver: Optional[UnifiedDependencyResolver] = None,
    ):
        """
        Initialize the pipeline assembler.

        Args:
            dag: PipelineDAG instance defining the pipeline structure
            config_map: Mapping from step name to config instance
            step_catalog: StepCatalog for config-to-builder resolution
            sagemaker_session: SageMaker session to use for creating the pipeline
            role: IAM role to use for the pipeline
            pipeline_parameters: List of pipeline parameters
            registry_manager: Optional registry manager for dependency injection
            dependency_resolver: Optional dependency resolver for dependency injection
        """
        self.dag = dag
        self.config_map = config_map
        self.step_catalog = step_catalog or StepCatalog()
        self.sagemaker_session = sagemaker_session
        self.role = role
        self.pipeline_parameters = pipeline_parameters or []

        # Store or create dependency components
        context_name = None
        for cfg in config_map.values():
            if hasattr(cfg, "pipeline_name"):
                context_name = cfg.pipeline_name
                break

        # Use provided components or create new ones
        self._registry_manager = registry_manager or RegistryManager()
        registry = self._registry_manager.get_registry(context_name or "default")
        self._dependency_resolver = dependency_resolver or create_dependency_resolver(
            registry
        )

        self.step_instances: Dict[str, Step] = {}
        self.step_builders: Dict[str, StepBuilderBase] = {}

        # Store connections between steps
        self.step_messages: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)

        # Validate inputs
        # Check that all nodes in the DAG have a corresponding config
        missing_configs = [
            node for node in self.dag.nodes if node not in self.config_map
        ]
        if missing_configs:
            raise ValueError(f"Missing configs for nodes: {missing_configs}")

        # Check that all configs have a corresponding step builder using StepCatalog
        for step_name, config in self.config_map.items():
            # Use StepCatalog to validate builder availability
            builder_class = self.step_catalog.get_builder_for_config(config, step_name)
            if not builder_class:
                config_class_name = type(config).__name__
                raise ValueError(
                    f"No step builder found for config: {config_class_name}"
                )

        # Check that all edges in the DAG connect nodes that exist in the DAG
        for src, dst in self.dag.edges:
            if src not in self.dag.nodes:
                raise ValueError(f"Edge source node not in DAG: {src}")
            if dst not in self.dag.nodes:
                raise ValueError(f"Edge destination node not in DAG: {dst}")

        logger.info("Input validation successful")

        # Initialize step builders
        self._initialize_step_builders()

    def _initialize_step_builders(self) -> None:
        """
        Initialize step builders for all steps in the DAG.

        This method creates a step builder instance for each step in the DAG,
        using the corresponding config from config_map and StepCatalog for
        direct config-to-builder resolution.
        """
        logger.info("Initializing step builders")
        start_time = time.time()

        for step_name in self.dag.nodes:
            try:
                config = self.config_map[step_name]

                # Use StepCatalog for direct config-to-builder resolution
                builder_cls = self.step_catalog.get_builder_for_config(
                    config, step_name
                )
                if not builder_cls:
                    config_class_name = type(config).__name__
                    raise ValueError(
                        f"No step builder found for config: {config_class_name}"
                    )

                # Initialize the builder with dependency components
                builder = builder_cls(
                    config=config,
                    sagemaker_session=self.sagemaker_session,
                    role=self.role,
                    registry_manager=self._registry_manager,  # Pass component
                    dependency_resolver=self._dependency_resolver,  # Pass component
                )

                # Pass execution prefix to the builder using the public method
                # Find PIPELINE_EXECUTION_TEMP_DIR in pipeline_parameters and pass it to the builder
                execution_prefix = None
                for param in self.pipeline_parameters:
                    if hasattr(param, "name") and param.name == "EXECUTION_S3_PREFIX":
                        execution_prefix = param
                        break

                if execution_prefix:
                    builder.set_execution_prefix(execution_prefix)
                    logger.info(f"Set execution prefix for {step_name}")
                # If no PIPELINE_EXECUTION_TEMP_DIR found, builder will fall back to config.pipeline_s3_loc

                self.step_builders[step_name] = builder
                logger.info(
                    f"Initialized builder for step {step_name} using StepCatalog"
                )
            except Exception as e:
                logger.error(f"Error initializing builder for step {step_name}: {e}")
                raise ValueError(
                    f"Failed to initialize step builder for {step_name}: {e}"
                ) from e

        elapsed_time = time.time() - start_time
        logger.info(
            f"Initialized {len(self.step_builders)} step builders in {elapsed_time:.2f} seconds"
        )

    def _propagate_messages(self) -> None:
        """
        Initialize step connections using the dependency resolver.

        This method analyzes the DAG structure and uses the dependency resolver
        to intelligently match inputs to outputs based on specifications.
        """
        logger.info("Initializing step connections using specifications")

        # Get dependency resolver
        resolver = self._get_dependency_resolver()

        # Process each edge in the DAG
        for src_step, dst_step in self.dag.edges:
            # Skip if builders don't exist
            if src_step not in self.step_builders or dst_step not in self.step_builders:
                continue

            # Get specs
            src_builder = self.step_builders[src_step]
            dst_builder = self.step_builders[dst_step]

            # Skip if no specifications
            if (
                not hasattr(src_builder, "spec")
                or not src_builder.spec
                or not hasattr(dst_builder, "spec")
                or not dst_builder.spec
            ):
                continue

            # Let resolver match outputs to inputs
            for dep_name, dep_spec in dst_builder.spec.dependencies.items():
                matches = []

                # Check if source step can provide this dependency
                for out_name, out_spec in src_builder.spec.outputs.items():
                    compatibility = resolver._calculate_compatibility(
                        dep_spec, out_spec, src_builder.spec
                    )
                    if compatibility > 0.5:  # Same threshold as resolver
                        matches.append((out_name, out_spec, compatibility))

                # Use best match if found
                if matches:
                    # Sort by compatibility score
                    matches.sort(key=lambda x: x[2], reverse=True)
                    best_match = matches[0]

                    # Check if there's already a better match
                    existing_match = self.step_messages.get(dst_step, {}).get(dep_name)
                    should_update = True

                    if existing_match:
                        existing_score = existing_match.get("compatibility", 0)
                        if existing_score >= best_match[2]:
                            should_update = False
                            logger.debug(
                                f"Skipping lower-scoring match for {dst_step}.{dep_name}: {src_step}.{best_match[0]} (score: {best_match[2]:.2f} < existing: {existing_score:.2f})"
                            )

                    if should_update:
                        # Store in step_messages
                        self.step_messages[dst_step][dep_name] = {
                            "source_step": src_step,
                            "source_output": best_match[0],
                            "match_type": "specification_match",
                            "compatibility": best_match[2],
                        }
                        logger.info(
                            f"Matched {dst_step}.{dep_name} to {src_step}.{best_match[0]} (score: {best_match[2]:.2f})"
                        )

        # Log final assignments for all steps
        logger.info("=" * 80)
        logger.info("FINAL DEPENDENCY ASSIGNMENTS:")
        logger.info("=" * 80)
        for dst_step in sorted(self.step_messages.keys()):
            matches = self.step_messages[dst_step]
            if matches:
                logger.info(f"\n{dst_step}:")
                for dep_name, match in sorted(matches.items()):
                    logger.info(
                        f"  {dep_name:30s} ← {match['source_step']}.{match['source_output']} "
                        f"(score: {match['compatibility']:.2f}, type: {match['match_type']})"
                    )
        logger.info("=" * 80)

        # Validate that required inputs have matches
        logger.info("\nVALIDATING REQUIRED DEPENDENCIES:")
        validation_errors = []
        for dst_step, dst_builder in self.step_builders.items():
            if not hasattr(dst_builder, "spec") or not dst_builder.spec:
                continue

            for dep_name, dep_spec in dst_builder.spec.dependencies.items():
                if dep_spec.required:
                    if dep_name not in self.step_messages.get(dst_step, {}):
                        error_msg = f"MISSING REQUIRED: {dst_step}.{dep_name} (type: {dep_spec.dependency_type})"
                        logger.error(error_msg)
                        validation_errors.append(error_msg)
                    else:
                        match = self.step_messages[dst_step][dep_name]
                        logger.info(
                            f"✓ {dst_step}.{dep_name} matched to {match['source_step']}.{match['source_output']}"
                        )

        if validation_errors:
            logger.error(f"\nFound {len(validation_errors)} validation errors!")
            for error in validation_errors:
                logger.error(f"  - {error}")
        else:
            logger.info("\n✓ All required dependencies have matches")

    def _generate_outputs(self, step_name: str) -> Dict[str, Any]:
        """
        Generate outputs dictionary using step builder's specification.

        This implementation leverages the step builder's specification
        to generate appropriate outputs using the new _get_base_output_path method
        and Join() for proper ParameterString support.

        Args:
            step_name: Name of the step to generate outputs for

        Returns:
            Dictionary with output paths based on specification
        """
        builder = self.step_builders[step_name]

        # If builder has no specification, return empty dict
        if not hasattr(builder, "spec") or not builder.spec:
            logger.warning(
                f"Step {step_name} has no specification, returning empty outputs"
            )
            return {}

        # Get base S3 location using the new method that supports PIPELINE_EXECUTION_TEMP_DIR
        base_s3_loc = builder._get_base_output_path()

        # Generate outputs dictionary based on specification
        outputs = {}
        step_type = builder.spec.step_type.lower()

        # Check if config has job_type (e.g., training, validation, testing, calibration)
        job_type = getattr(builder.config, "job_type", None)

        # Use each output specification to generate standard output path
        for logical_name, output_spec in builder.spec.outputs.items():
            # Standard path pattern using Join instead of f-string to ensure proper parameter substitution
            from sagemaker.workflow.functions import Join

            if job_type:
                # Include job_type in path for steps that have it (e.g., DummyDataLoading, CradleDataLoading)
                outputs[logical_name] = Join(
                    on="/", values=[base_s3_loc, step_type, job_type, logical_name]
                )
            else:
                # Fallback without job_type for steps that don't have it
                outputs[logical_name] = Join(
                    on="/", values=[base_s3_loc, step_type, logical_name]
                )

            # Add debug log with type-safe handling using safe_value_for_logging
            logger.debug(
                f"Generated output for {step_name}.{logical_name}: {safe_value_for_logging(outputs[logical_name])}"
            )

        return outputs

    def _instantiate_step(self, step_name: str) -> Step:
        """
        Instantiate a pipeline step with appropriate inputs from dependencies.

        This method creates a step using the step builder's create_step method,
        delegating input extraction and output generation to the builder.

        Args:
            step_name: Name of the step to instantiate

        Returns:
            Instantiated SageMaker Pipeline Step
        """
        builder = self.step_builders[step_name]

        # Get dependency steps
        dependencies = []
        dag_dependencies = list(self.dag.get_dependencies(step_name))
        logger.info(
            f"[DEPENDS_ON] Step '{step_name}' has {len(dag_dependencies)} DAG predecessors: {dag_dependencies}"
        )

        for dep_name in dag_dependencies:
            if dep_name in self.step_instances:
                dependencies.append(self.step_instances[dep_name])
                logger.info(
                    f"[DEPENDS_ON]   ✓ Added dependency: {dep_name} (step name: {self.step_instances[dep_name].name})"
                )
            else:
                logger.warning(
                    f"[DEPENDS_ON]   ✗ Dependency {dep_name} not yet instantiated, skipping"
                )

        logger.info(
            f"[DEPENDS_ON] Final dependencies list for '{step_name}': {len(dependencies)} steps"
        )

        # Extract parameters from message dictionaries for backward compatibility
        inputs = {}
        if step_name in self.step_messages:
            for input_name, message in self.step_messages[step_name].items():
                src_step = message["source_step"]
                src_output = message["source_output"]
                if src_step in self.step_instances:
                    # Try to get the source step's builder to access its specifications
                    src_builder = self.step_builders.get(src_step)
                    output_spec = None

                    # Try to find the output spec for this output name
                    if (
                        src_builder
                        and hasattr(src_builder, "spec")
                        and src_builder.spec
                    ):
                        output_spec = src_builder.spec.get_output_by_name_or_alias(
                            src_output
                        )

                    if output_spec:
                        try:
                            # Create a PropertyReference object
                            prop_ref = PropertyReference(
                                step_name=src_step, output_spec=output_spec
                            )

                            # Use the enhanced to_runtime_property method to get an actual SageMaker Properties object
                            runtime_prop = prop_ref.to_runtime_property(
                                self.step_instances
                            )
                            inputs[input_name] = runtime_prop

                            logger.debug(
                                f"Created runtime property reference for {step_name}.{input_name} -> {src_step}.{output_spec.property_path}"
                            )
                        except Exception as e:
                            # Log the error and fall back to a safe string
                            logger.warning(
                                f"Error creating runtime property reference: {str(e)}"
                            )
                            s3_uri = f"s3://pipeline-reference/{src_step}/{src_output}"
                            inputs[input_name] = s3_uri
                            logger.warning(f"Using S3 URI fallback: {s3_uri}")
                    else:
                        # Create a safe string reference as a fallback
                        s3_uri = f"s3://pipeline-reference/{src_step}/{src_output}"
                        inputs[input_name] = s3_uri
                        logger.warning(
                            f"Could not find output spec for {src_step}.{src_output}, using S3 URI placeholder: {s3_uri}"
                        )

        # Generate outputs using the specification
        outputs = self._generate_outputs(step_name)

        # Create step with extracted inputs and outputs
        kwargs = {
            "inputs": inputs,
            "outputs": outputs,
            "dependencies": dependencies,
            "enable_caching": builder.config.enable_caching,
        }

        try:
            step = builder.create_step(**kwargs)
            logger.info(f"Built step {step_name}")

            # Note: Cradle data loading request collection removed as part of Phase 2 cleanup
            # This is now handled by the standalone execution document generator
            # (CradleDataLoadingHelper in cursus.mods.exe_doc.cradle_helper)

            return step
        except Exception as e:
            logger.error(f"Error building step {step_name}: {e}")
            raise ValueError(f"Failed to build step {step_name}: {e}") from e

    @classmethod
    def create_with_components(
        cls,
        dag: PipelineDAG,
        config_map: Dict[str, BasePipelineConfig],
        step_catalog: Optional[StepCatalog] = None,
        context_name: Optional[str] = None,
        **kwargs: Any,
    ) -> "PipelineAssembler":
        """
        Create pipeline assembler with managed components.

        This factory method creates a pipeline assembler with properly configured
        dependency components from the factory module.

        Args:
            dag: PipelineDAG instance defining the pipeline structure
            config_map: Mapping from step name to config instance
            step_catalog: StepCatalog for config-to-builder resolution
            context_name: Optional context name for registry
            **kwargs: Additional arguments to pass to the constructor

        Returns:
            Configured PipelineAssembler instance
        """
        components = create_pipeline_components(context_name)
        return cls(
            dag=dag,
            config_map=config_map,
            step_catalog=step_catalog,
            registry_manager=components["registry_manager"],
            dependency_resolver=components["resolver"],
            **kwargs,
        )

    def _get_registry_manager(self) -> RegistryManager:
        """Get the registry manager."""
        return self._registry_manager

    def _get_dependency_resolver(self) -> UnifiedDependencyResolver:
        """Get the dependency resolver."""
        return self._dependency_resolver

    def generate_pipeline(self, pipeline_name: str) -> Pipeline:
        """
        Build and return a SageMaker Pipeline object.

        This method builds the pipeline by:
        1. Propagating messages between steps using specification-based matching
        2. Instantiating steps in topological order
        3. Creating the pipeline with the instantiated steps

        Args:
            pipeline_name: Name of the pipeline

        Returns:
            SageMaker Pipeline object
        """
        logger.info(f"Generating pipeline: {pipeline_name}")
        start_time = time.time()

        # Reset step instances if we're regenerating the pipeline
        if self.step_instances:
            logger.info("Clearing existing step instances for pipeline regeneration")
            self.step_instances = {}

        # Propagate messages between steps
        self._propagate_messages()

        # Topological sort to determine build order
        try:
            build_order = self.dag.topological_sort()
            logger.info(f"Build order: {build_order}")
        except ValueError as e:
            logger.error(f"Error in topological sort: {e}")
            raise ValueError(f"Failed to determine build order: {e}") from e

        # Instantiate steps in topological order
        for step_name in build_order:
            try:
                step = self._instantiate_step(step_name)
                self.step_instances[step_name] = step
            except Exception as e:
                logger.error(f"Error instantiating step {step_name}: {e}")
                raise ValueError(f"Failed to instantiate step {step_name}: {e}") from e

        # Create the pipeline
        steps = [self.step_instances[name] for name in build_order]
        pipeline = Pipeline(
            name=pipeline_name,
            parameters=self.pipeline_parameters,
            steps=steps,
            sagemaker_session=self.sagemaker_session,
        )

        elapsed_time = time.time() - start_time
        logger.info(
            f"Generated pipeline {pipeline_name} with {len(steps)} steps in {elapsed_time:.2f} seconds"
        )

        return pipeline

    def generate_single_node_pipeline(
        self, target_node: str, manual_inputs: Dict[str, str], pipeline_name: str
    ) -> Pipeline:
        """
        Generate pipeline with single node and manual inputs.

        This method bypasses normal message propagation and dependency resolution,
        directly instantiating the target node with provided manual inputs.

        **Backward Compatibility**: This method is completely isolated and does not
        affect existing pipeline generation flow. The normal generate_pipeline()
        method remains unchanged.

        Args:
            target_node: Name of node to execute in isolation
            manual_inputs: Manual input paths (logical_name -> s3_uri)
            pipeline_name: Name for generated pipeline

        Returns:
            Single-node Pipeline

        Raises:
            ValueError: If target node not found in step builders

        Example:
            >>> manual_inputs = {
            ...     "input_path": "s3://bucket/run-123/preprocess/data/"
            ... }
            >>> pipeline = assembler.generate_single_node_pipeline(
            ...     target_node="train",
            ...     manual_inputs=manual_inputs,
            ...     pipeline_name="train-debug-001"
            ... )
        """
        logger.info(f"[SINGLE_NODE] Generating pipeline for: {target_node}")

        # Validate target node exists
        if target_node not in self.step_builders:
            raise ValueError(
                f"Target node '{target_node}' not found. "
                f"Available: {list(self.step_builders.keys())}"
            )

        # Clear step instances to ensure clean state
        self.step_instances = {}

        # Instantiate only the target step with manual inputs
        step = self._instantiate_step_with_manual_inputs(target_node, manual_inputs)
        self.step_instances[target_node] = step

        # Create minimal pipeline with single step
        pipeline = Pipeline(
            name=pipeline_name,
            parameters=self.pipeline_parameters,
            steps=[step],
            sagemaker_session=self.sagemaker_session,
        )

        logger.info(f"[SINGLE_NODE] Pipeline created: {pipeline_name}")
        return pipeline

    def _instantiate_step_with_manual_inputs(
        self, step_name: str, manual_inputs: Dict[str, str]
    ) -> Step:
        """
        Instantiate step with manually provided input paths.

        This method bypasses message-based input resolution and directly uses
        the provided manual input paths (S3 URIs). This is specifically for
        single-node execution mode and does not affect normal pipeline flow.

        **Backward Compatibility**: This is a new private method that does not
        modify any existing methods or class state beyond what's necessary for
        single-node execution.

        Args:
            step_name: Name of step to instantiate
            manual_inputs: Manual input paths (logical_name -> s3_uri)

        Returns:
            Instantiated SageMaker Step

        Raises:
            ValueError: If step creation fails
        """
        builder = self.step_builders[step_name]

        logger.info(f"[SINGLE_NODE] Instantiating '{step_name}' with manual inputs")
        for input_name, s3_path in manual_inputs.items():
            logger.info(f"[SINGLE_NODE]   {input_name}: {s3_path}")

        # No dependencies for isolated execution
        dependencies = []

        # Manual inputs are already S3 URIs - use directly
        inputs = dict(manual_inputs)

        # Generate outputs using specification (same as normal flow)
        outputs = self._generate_outputs(step_name)

        # Create step with manual inputs
        kwargs = {
            "inputs": inputs,
            "outputs": outputs,
            "dependencies": dependencies,
            "enable_caching": builder.config.enable_caching,
        }

        try:
            step = builder.create_step(**kwargs)
            logger.info(f"[SINGLE_NODE] Step created: {step_name}")
            return step

        except Exception as e:
            logger.error(f"[SINGLE_NODE] Error creating step: {e}")
            raise ValueError(f"Failed to create step '{step_name}': {e}") from e

    def analyze_pipeline_structure(self) -> None:
        """
        Analyze and print the complete pipeline structure including:
        1. Dependency graph between steps
        2. Input assignments with logical names, property paths, and destination paths

        This method uses internal assembler data (step_messages, step_builders, step_instances)
        to provide comprehensive insights into how the pipeline is structured.

        This should be called after generate_pipeline() to ensure all steps are instantiated.
        """
        if not self.step_instances:
            logger.warning("No step instances found. Call generate_pipeline() first.")
            return

        # Part 1: Dependency Graph
        print("=" * 80)
        print("PIPELINE DEPENDENCY GRAPH")
        print("=" * 80)

        for step_name, step in self.step_instances.items():
            step_type = step.__class__.__name__
            depends_on = getattr(step, "depends_on", []) or []

            print(f"\n{step_name}:")
            print(f"  Type: {step_type}")

            if depends_on:
                print(f"  Depends on ({len(depends_on)}):")
                for dep in depends_on:
                    dep_name = dep.name if hasattr(dep, "name") else str(dep)
                    print(f"    - {dep_name}")
            else:
                print(f"  Depends on: None (root step)")

        # Part 2: Input Assignments
        print("\n" + "=" * 80)
        print("INPUT ASSIGNMENTS")
        print("=" * 80)

        for step_name in sorted(self.step_messages.keys()):
            builder = self.step_builders.get(step_name)
            if not builder or not hasattr(builder, "spec") or not builder.spec:
                continue

            spec = builder.spec
            messages = self.step_messages[step_name]

            if not messages:
                continue

            print(f"\n{step_name}:")

            for input_name in sorted(messages.keys()):
                match = messages[input_name]

                # Get dependency spec for this input
                dep_spec = spec.dependencies.get(input_name)
                if not dep_spec:
                    continue

                req = "✓" if dep_spec.required else "○"
                dep_type = str(dep_spec.dependency_type)

                print(f"\n  {req} {input_name} ({dep_type})")
                print(f"      Match Type: {match.get('match_type', 'unknown')}")
                print(f"      Compatibility: {match.get('compatibility', 0):.2f}")
                print(f"      Source: {match['source_step']}.{match['source_output']}")

                # Get source property path from source builder's spec
                source_builder = self.step_builders.get(match["source_step"])
                if (
                    source_builder
                    and hasattr(source_builder, "spec")
                    and source_builder.spec
                ):
                    source_spec = source_builder.spec
                    output_spec = source_spec.get_output_by_name_or_alias(
                        match["source_output"]
                    )
                    if output_spec:
                        print(f"      Property Path: {output_spec.property_path}")

                # Get container path from contract
                contract = getattr(builder, "contract", None)
                if contract and hasattr(contract, "expected_input_paths"):
                    dest_path = contract.expected_input_paths.get(input_name, "N/A")
                    print(f"      Destination: {dest_path}")

        print("\n" + "=" * 80)
