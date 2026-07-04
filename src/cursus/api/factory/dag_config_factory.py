"""
DAG Configuration Factory

This module provides the main DAGConfigFactory class for interactive pipeline configuration
generation. It orchestrates the step-by-step workflow for collecting user inputs and
generating complete pipeline configurations.

Key Components:
- DAGConfigFactory: Main interactive factory class
- ConfigurationIncompleteError: Exception for incomplete configurations
- Interactive workflow management and validation
"""

from typing import Dict, List, Type, Any, Optional
from pydantic import BaseModel
import logging

from .configuration_generator import ConfigurationGenerator
from .field_extractor import extract_field_requirements, extract_non_inherited_fields

logger = logging.getLogger(__name__)


class ConfigurationIncompleteError(Exception):
    """Exception raised when essential configuration fields are missing."""

    pass


class DAGConfigFactory:
    """
    Interactive factory for step-by-step pipeline configuration generation.

    This class provides a user-friendly interface for creating pipeline configurations
    by guiding users through the process of setting base configurations and step-specific
    configurations in a structured workflow.

    Workflow:
    1. Analyze DAG to get config class mapping
    2. Collect base configurations first
    3. Guide user through step-specific configurations
    4. Generate final config instances with inheritance
    """

    def __init__(self, dag):
        """
        Initialize factory with DAG analysis using robust canonical step name extraction.

        Args:
            dag: Pipeline DAG object to create configurations for
        """
        logger.info("🔧 Initializing DAGConfigFactory...")

        self.dag = dag
        self.config_generator = None  # Initialized after base configs are set

        # EAGER initialization with progress feedback to avoid hanging
        logger.info("📊 Step 1/3: Mapping DAG nodes to config classes...")
        self._config_class_map = self._map_dag_to_config_classes_robust(dag)
        logger.info(
            f"✅ Mapped {len(self._config_class_map)} DAG nodes to config classes"
        )

        # Pre-compute and cache inheritance checks FIRST (before checking processing config needs)
        logger.info("📊 Step 2/3: Pre-computing inheritance relationships...")
        self._cached_inheritance_info: Dict[str, Dict[str, bool]] = {}
        self._precompute_inheritance_checks()
        logger.info(
            f"✅ Cached inheritance info for {len(self._cached_inheritance_info)} steps"
        )

        # Pre-compute processing config requirement (now can use inheritance cache)
        logger.info("📊 Step 3/3: Checking processing config requirements...")
        self._needs_processing_config = self._check_if_needs_processing_config()

        # Cache the actual field requirements to avoid expensive extraction on every call
        self._cached_processing_requirements = None
        if self._needs_processing_config:
            logger.info(
                "📊 Step 3a/3: Extracting processing config field requirements..."
            )
            self._cached_processing_requirements = (
                self._extract_processing_requirements()
            )
            logger.info(
                f"✅ Cached {len(self._cached_processing_requirements)} processing config fields"
            )
        else:
            logger.info("✅ Processing config not required")

        # Pre-compute and cache step-specific requirements for ALL steps to avoid hanging
        logger.info("📊 Step 3b/3: Pre-computing step requirements for all steps...")
        self._cached_step_requirements: Dict[str, List[Dict[str, Any]]] = {}
        self._precompute_all_step_requirements()
        logger.info(
            f"✅ Cached requirements for {len(self._cached_step_requirements)} steps"
        )

        self.base_config = None  # BasePipelineConfig instance
        self.base_processing_config = None  # BaseProcessingStepConfig instance
        self.step_configs: Dict[
            str, Dict[str, Any]
        ] = {}  # Raw inputs for serialization
        self.step_config_instances: Dict[str, BaseModel] = {}  # Validated instances

        logger.info("📊 Step 3/3: Factory initialization complete")
        logger.info(
            f"✅ DAGConfigFactory ready with {len(self._config_class_map)} steps"
        )

    def _map_dag_to_config_classes_robust(self, dag) -> Dict[str, Type[BaseModel]]:
        """
        Map DAG nodes to config classes using robust canonical step name extraction.

        This method implements the pattern: node_name = "{canonical_step_name}_{job_type}"
        and uses the existing registry system to map canonical names to config classes.

        Args:
            dag: Pipeline DAG object

        Returns:
            Dictionary mapping node names to configuration classes
        """
        config_class_map = {}

        # Get DAG nodes - handle different DAG implementations
        nodes = self._get_dag_nodes(dag)

        # Get available config classes from the step catalog system
        available_config_classes = self._get_available_config_classes()

        for node_name in nodes:
            try:
                # Extract canonical step name from node name using the established pattern
                canonical_step_name = self._extract_canonical_step_name(node_name)

                # Use registry system to map canonical name to config class
                config_class = self._resolve_canonical_name_to_config_class(
                    canonical_step_name, available_config_classes
                )

                if config_class:
                    config_class_map[node_name] = config_class
                    logger.info(
                        f"✅ Mapped '{node_name}' -> '{canonical_step_name}' -> {config_class.__name__}"
                    )
                else:
                    logger.warning(
                        f"❌ No config class found for canonical step name: {canonical_step_name} (from node: {node_name})"
                    )

            except Exception as e:
                logger.error(f"❌ Failed to map node '{node_name}': {e}")

        logger.info(
            f"Successfully mapped {len(config_class_map)}/{len(nodes)} DAG nodes to config classes"
        )
        return config_class_map

    def _extract_canonical_step_name(self, node_name: str) -> str:
        """
        Extract canonical step name from DAG node name using the established pattern.

        Pattern: node_name = "{canonical_step_name}_{job_type}"
        Examples:
        - "XGBoostModelEval_calibration" -> "XGBoostModelEval"
        - "CradleDataLoading_training" -> "CradleDataLoading"
        - "TabularPreprocessing_training" -> "TabularPreprocessing"

        Args:
            node_name: DAG node name

        Returns:
            Canonical step name
        """
        # Split on the last underscore to separate canonical name from job_type
        if "_" in node_name:
            parts = node_name.rsplit("_", 1)
            canonical_name = parts[0]
            job_type = parts[1]

            logger.debug(
                f"Extracted canonical name '{canonical_name}' and job type '{job_type}' from '{node_name}'"
            )
            return canonical_name
        else:
            # If no underscore, assume the whole name is the canonical step name
            logger.debug(
                f"No job type suffix found in '{node_name}', using as canonical name"
            )
            return node_name

    def _resolve_step_name_to_node(
        self, step_name: str, job_type: Optional[str] = None
    ) -> str:
        """Resolve a caller-supplied step name to the actual DAG node key.

        ``set_step_config`` accepts the step given WITH or WITHOUT a job_type suffix — the
        config class is found by BASE step name, and the job_type value is carried in kwargs.
        A DAG node is ``{canonical_step_name}[_{job_type}]``, so this maps the caller's name
        onto whichever node key the DAG actually declared:

        1. Exact node-key match (already suffixed, or a suffix-less node) → use as-is.
        2. Bare name + explicit ``job_type`` → try composed key ``{step_name}_{job_type}``.
        3. Bare name → the single node whose canonical base step name equals ``step_name``
           (unambiguous); if several nodes share the base, keep the name unchanged and let
           the caller disambiguate (the exact-key check downstream will raise clearly).

        Returns the resolved node key (or the original ``step_name`` if nothing better fits).
        """
        # 1. Exact match — nothing to do.
        if step_name in self._config_class_map:
            return step_name

        # 2. Bare name + job_type → composed suffix key.
        if job_type:
            composed = f"{step_name}_{job_type}"
            if composed in self._config_class_map:
                logger.debug(
                    f"Resolved step '{step_name}' + job_type='{job_type}' → node '{composed}'"
                )
                return composed

        # 3. Bare name → unique node whose base canonical step name matches.
        base_matches = [
            node
            for node in self._config_class_map
            if self._extract_canonical_step_name(node) == step_name
        ]
        if len(base_matches) == 1:
            logger.debug(
                f"Resolved bare step '{step_name}' → node '{base_matches[0]}' (unique base match)"
            )
            return base_matches[0]
        if len(base_matches) > 1:
            logger.warning(
                f"Step '{step_name}' is ambiguous — {len(base_matches)} DAG nodes share this base "
                f"({base_matches}); pass the full node name (e.g. with its job_type suffix)."
            )

        return step_name

    def _resolve_canonical_name_to_config_class(
        self,
        canonical_step_name: str,
        available_config_classes: Dict[str, Type[BaseModel]],
    ) -> Optional[Type[BaseModel]]:
        """
        Resolve canonical step name to config class using registry system.

        Args:
            canonical_step_name: Canonical step name (e.g., "XGBoostModelEval")
            available_config_classes: Available config classes from step catalog

        Returns:
            Config class or None if not found
        """
        try:
            # Method 1: Try direct registry lookup using existing system
            from ...registry.step_names import (
                get_config_step_registry,
                CONFIG_STEP_REGISTRY,
            )

            # Get the registry mapping
            config_registry = get_config_step_registry()

            # Try to find config class by canonical step name
            for config_class_name, registered_step_name in config_registry.items():
                if registered_step_name == canonical_step_name:
                    # Found a match, look for this config class in available classes
                    if config_class_name in available_config_classes:
                        return available_config_classes[config_class_name]

            # Method 2: Fallback to legacy CONFIG_STEP_REGISTRY
            for config_class_name, registered_step_name in CONFIG_STEP_REGISTRY.items():
                if registered_step_name == canonical_step_name:
                    if config_class_name in available_config_classes:
                        return available_config_classes[config_class_name]

            # Method 3: Try pattern-based matching as final fallback
            expected_config_class_name = f"{canonical_step_name}Config"
            if expected_config_class_name in available_config_classes:
                logger.debug(
                    f"Found config class using pattern matching: {expected_config_class_name}"
                )
                return available_config_classes[expected_config_class_name]

            # Method 4: Try case-insensitive matching
            canonical_lower = canonical_step_name.lower()
            for class_name, config_class in available_config_classes.items():
                if canonical_lower in class_name.lower():
                    logger.debug(
                        f"Found config class using case-insensitive matching: {class_name}"
                    )
                    return config_class

            return None

        except Exception as e:
            logger.warning(f"Error in registry lookup for '{canonical_step_name}': {e}")

            # Final fallback: pattern-based matching
            expected_config_class_name = f"{canonical_step_name}Config"
            if expected_config_class_name in available_config_classes:
                return available_config_classes[expected_config_class_name]

            return None

    def _get_available_config_classes(self) -> Dict[str, Type[BaseModel]]:
        """
        Get available config classes from the step catalog system.

        Returns:
            Dictionary mapping config class names to config classes
        """
        try:
            # Use the unified config manager to get config classes
            from ...core.config_fields.unified_config_manager import (
                get_unified_config_manager,
            )

            unified_manager = get_unified_config_manager()
            config_classes = unified_manager.get_config_classes()

            logger.debug(
                f"Retrieved {len(config_classes)} config classes from unified config manager"
            )
            return config_classes

        except Exception as e:
            logger.warning(f"Failed to get config classes from unified manager: {e}")

            # Fallback: use step catalog auto-discovery
            try:
                from ...step_catalog.config_discovery import ConfigAutoDiscovery
                from ...step_catalog import StepCatalog

                # Get package root from StepCatalog
                temp_catalog = StepCatalog(workspace_dirs=None)
                package_root = temp_catalog.package_root

                config_discovery = ConfigAutoDiscovery(
                    package_root=package_root, workspace_dirs=[]
                )

                discovered_classes = config_discovery.build_complete_config_classes()
                logger.info(
                    f"Discovered {len(discovered_classes)} config classes via auto-discovery"
                )
                return discovered_classes

            except Exception as discovery_e:
                logger.error(f"Failed to use config auto-discovery: {discovery_e}")
                return {}

    def _get_dag_nodes(self, dag) -> List[str]:
        """
        Extract node names from DAG object, handling different DAG implementations.

        Args:
            dag: Pipeline DAG object

        Returns:
            List of node names in the DAG
        """
        # Handle different DAG implementations
        if hasattr(dag, "nodes"):
            if callable(dag.nodes):
                return list(dag.nodes())
            else:
                return list(dag.nodes)
        elif hasattr(dag, "get_nodes"):
            return dag.get_nodes()
        elif hasattr(dag, "steps"):
            return (
                list(dag.steps.keys())
                if isinstance(dag.steps, dict)
                else list(dag.steps)
            )
        else:
            logger.warning(f"Unknown DAG structure: {type(dag)}")
            return []

    def get_config_class_map(self) -> Dict[str, Type[BaseModel]]:
        """
        Get mapping of DAG node names to config classes (not instances).

        Returns:
            Dictionary mapping node names to configuration classes
        """
        return self._config_class_map.copy()

    def get_base_config_requirements(self) -> List[Dict[str, Any]]:
        """
        Get base configuration requirements directly from Pydantic class definition.

        Extracts field requirements directly from BasePipelineConfig Pydantic class definition.

        Returns:
            List of field requirement dictionaries with format:
            {
                'name': str,           # Field name
                'type': str,           # Field type as string
                'description': str,    # Field description from Pydantic Field()
                'required': bool,      # True for required fields, False for optional
                'default': Any         # Default value (only for optional fields)
            }
        """
        try:
            # Import BasePipelineConfig using correct relative import
            from ...core.base.config_base import BasePipelineConfig

            return extract_field_requirements(BasePipelineConfig)
        except ImportError:
            logger.warning("BasePipelineConfig not found, returning empty requirements")
            return []

    def _check_if_needs_processing_config(self) -> bool:
        """
        Check if any step in the DAG requires processing configuration.

        This method is called once during initialization and cached to avoid
        repeated expensive inheritance checks.

        Returns:
            True if any step requires processing config, False otherwise
        """
        try:
            logger.debug("Checking if any step requires processing configuration...")

            # Check if any step requires processing configuration
            for config_class in self._config_class_map.values():
                if self._inherits_from_processing_config(config_class):
                    logger.debug("Found step requiring processing configuration")
                    return True

            logger.debug("No steps require processing configuration")
            return False

        except Exception as e:
            logger.warning(f"Error checking processing config requirement: {e}")
            return False

    def _extract_processing_requirements(self) -> List[Dict[str, Any]]:
        """
        Extract base processing configuration field requirements.

        This expensive operation is called once during initialization and cached.
        Extracts the 9 base processing fields that ProcessingStepConfigBase adds
        on top of BasePipelineConfig.

        Returns:
            List of field requirement dictionaries for processing-specific fields
        """
        try:
            from ...core.base.config_base import BasePipelineConfig
            from ...steps.configs.config_processing_step_base import (
                ProcessingStepConfigBase,
            )

            logger.debug("Extracting processing config field requirements...")
            requirements = extract_non_inherited_fields(
                ProcessingStepConfigBase, BasePipelineConfig
            )
            logger.debug(f"Extracted {len(requirements)} processing config fields")
            return requirements

        except ImportError as e:
            logger.warning(f"Could not import processing config classes: {e}")
            return []
        except Exception as e:
            logger.error(f"Error extracting processing requirements: {e}")
            return []

    def _precompute_all_step_requirements(self) -> None:
        """
        Pre-compute and cache step-specific requirements for all steps.

        This expensive operation runs once during initialization and caches results
        for all steps in the DAG. Each step's requirements are computed by calling
        _extract_step_specific_fields which performs ~200+ operations per step.

        By caching all results upfront, subsequent calls to get_step_requirements(),
        can_auto_configure_step(), and get_pending_steps() return instantly.
        """
        for step_name, config_class in self._config_class_map.items():
            try:
                logger.debug(f"Pre-computing requirements for {step_name}...")
                requirements = self._extract_step_specific_fields(config_class)
                self._cached_step_requirements[step_name] = requirements
                logger.debug(f"  ✓ Cached {len(requirements)} fields for {step_name}")
            except Exception as e:
                logger.warning(f"Failed to extract requirements for {step_name}: {e}")
                # Store empty list as fallback to avoid re-attempting
                self._cached_step_requirements[step_name] = []

    def _precompute_inheritance_checks(self) -> None:
        """
        Pre-compute and cache inheritance relationships for all config classes.

        Computes once during initialization whether each config class inherits from
        ProcessingStepConfigBase or BasePipelineConfig. This eliminates repeated
        import and issubclass() calls during get_pending_steps() execution.

        Cached results are stored per config class (not per step) since multiple
        steps may use the same config class.
        """
        # Track which classes we've already checked to avoid duplicates
        checked_classes = set()

        for step_name, config_class in self._config_class_map.items():
            class_name = config_class.__name__

            # Skip if we've already checked this class
            if class_name in checked_classes:
                # Copy from existing entry
                self._cached_inheritance_info[step_name] = (
                    self._cached_inheritance_info[
                        next(
                            k
                            for k, v in self._config_class_map.items()
                            if v.__name__ == class_name
                            and k in self._cached_inheritance_info
                        )
                    ]
                )
                continue

            checked_classes.add(class_name)

            try:
                logger.debug(f"Checking inheritance for {class_name}...")

                # Check both inheritance types
                inherits_processing = self._check_processing_inheritance(config_class)
                inherits_base = self._check_base_inheritance(config_class)

                # Cache the results
                self._cached_inheritance_info[step_name] = {
                    "inherits_processing": inherits_processing,
                    "inherits_base": inherits_base,
                }

                logger.debug(
                    f"  ✓ {class_name}: processing={inherits_processing}, base={inherits_base}"
                )
            except Exception as e:
                logger.warning(f"Failed to check inheritance for {class_name}: {e}")
                # Store False as safe fallback
                self._cached_inheritance_info[step_name] = {
                    "inherits_processing": False,
                    "inherits_base": False,
                }

    def _check_processing_inheritance(self, config_class: Type[BaseModel]) -> bool:
        """
        Internal helper to check ProcessingStepConfigBase inheritance.

        Used only during initialization for caching. After initialization,
        use _inherits_from_processing_config() which returns cached results.
        """
        try:
            from ...steps.configs.config_processing_step_base import (
                ProcessingStepConfigBase,
            )

            return issubclass(config_class, ProcessingStepConfigBase)
        except (ImportError, TypeError):
            return False

    def _check_base_inheritance(self, config_class: Type[BaseModel]) -> bool:
        """
        Internal helper to check BasePipelineConfig inheritance.

        Used only during initialization for caching. After initialization,
        use _inherits_from_base_config() which returns cached results.
        """
        try:
            from ...core.base.config_base import BasePipelineConfig

            return issubclass(config_class, BasePipelineConfig)
        except (ImportError, TypeError):
            return False

    def get_base_processing_config_requirements(self) -> List[Dict[str, Any]]:
        """
        Get base processing configuration requirements.

        Returns only the non-inherited fields specific to BaseProcessingStepConfig.
        Inherited fields from BasePipelineConfig can be obtained by calling get_base_config_requirements().

        Uses cached field requirements computed during initialization for instant response.

        Returns:
            List of field requirement dictionaries for processing-specific fields
        """
        # Return cached requirements (computed once during __init__)
        if self._cached_processing_requirements is not None:
            return self._cached_processing_requirements

        # If somehow cache wasn't initialized (shouldn't happen), return empty list
        logger.debug(
            "Processing requirements cache not initialized, returning empty list"
        )
        return []

    def set_base_config(self, **kwargs) -> None:
        """
        Set base pipeline configuration from user inputs.

        Args:
            **kwargs: Base configuration field values
        """
        try:
            from ...core.base.config_base import BasePipelineConfig

            self.base_config = BasePipelineConfig(**kwargs)

            # Initialize config generator once base config is set
            self.config_generator = ConfigurationGenerator(
                base_config=self.base_config,
                base_processing_config=self.base_processing_config,
            )

            logger.info("Base configuration set successfully")

        except ImportError:
            logger.error("BasePipelineConfig not available")
            raise ValueError("BasePipelineConfig class not found")
        except Exception as e:
            logger.error(f"Failed to set base configuration: {e}")
            raise ValueError(f"Invalid base configuration: {e}")

    def set_base_processing_config(self, **kwargs) -> None:
        """
        Set base processing configuration from user inputs.

        Args:
            **kwargs: Base processing configuration field values
        """
        try:
            from ...steps.configs.config_processing_step_base import (
                ProcessingStepConfigBase,
            )

            # Combine base config values with processing-specific values
            combined_kwargs = {}
            if self.base_config:
                combined_kwargs.update(
                    self.config_generator._extract_config_values(self.base_config)
                )
            combined_kwargs.update(kwargs)

            self.base_processing_config = ProcessingStepConfigBase(**combined_kwargs)

            # Update config generator with processing config
            if self.config_generator:
                self.config_generator.base_processing_config = (
                    self.base_processing_config
                )

            logger.info("Base processing configuration set successfully")

        except ImportError:
            logger.error("BaseProcessingStepConfig not available")
            raise ValueError("BaseProcessingStepConfig class not found")
        except Exception as e:
            logger.error(f"Failed to set base processing configuration: {e}")
            raise ValueError(f"Invalid base processing configuration: {e}")

    def get_pending_steps(self) -> List[str]:
        """
        Get list of steps that still need configuration.

        Steps with only tier 2+ (optional) fields besides inherited fields
        are considered auto-configurable and not pending.

        Returns:
            List of step names that haven't been configured yet and require user input
        """
        logger.debug("🔍 get_pending_steps() called")
        pending_steps = []

        for i, step_name in enumerate(self._config_class_map.keys(), 1):
            logger.debug(f"  [{i}/11] Checking step: {step_name}")

            if step_name in self.step_configs:
                logger.debug(f"    ✓ {step_name} already configured")
                continue  # Already configured

            # Check if step can be auto-configured (only has tier 2+ fields)
            logger.debug(f"    Checking if {step_name} can be auto-configured...")
            if self.can_auto_configure_step(step_name):
                logger.debug(f"    ✓ {step_name} can be auto-configured")
                continue  # Can be auto-configured, not pending

            logger.debug(f"    → {step_name} needs configuration")
            pending_steps.append(step_name)

        logger.debug(f"✅ get_pending_steps() complete: {len(pending_steps)} pending")
        return pending_steps

    def can_auto_configure_step(self, step_name: str) -> bool:
        """
        Check if a step can be auto-configured (only has tier 2+ fields besides inherited).

        Args:
            step_name: Name of the step to check

        Returns:
            True if step can be auto-configured, False if it requires user input
        """
        logger.debug(f"    → can_auto_configure_step({step_name}) called")

        if step_name not in self._config_class_map:
            logger.debug(f"      ✗ {step_name} not in config_class_map")
            return False

        config_class = self._config_class_map[step_name]
        logger.debug(f"      Config class: {config_class.__name__}")

        # Check if prerequisites are met
        logger.debug(f"      Validating prerequisites...")
        try:
            self._validate_prerequisites_for_step(step_name, config_class)
            logger.debug(f"      ✓ Prerequisites validated")
        except ValueError as e:
            logger.debug(f"      ✗ Prerequisites not met: {e}")
            return False  # Prerequisites not met, can't auto-configure

        # Get step-specific requirements (excluding inherited fields)
        logger.debug(f"      Getting step requirements...")
        step_requirements = self.get_step_requirements(step_name)
        logger.debug(f"      Got {len(step_requirements)} requirements")

        essential_step_fields = [
            req["name"] for req in step_requirements if req["required"]
        ]
        logger.debug(f"      Found {len(essential_step_fields)} essential fields")

        # If there are no essential step-specific fields, it can be auto-configured
        result = len(essential_step_fields) == 0
        logger.debug(f"    ← can_auto_configure_step({step_name}) = {result}")
        return result

    def get_step_requirements(self, step_name: str) -> List[Dict[str, Any]]:
        """
        Get step-specific requirements excluding inherited base config fields.

        Uses cached field requirements computed during initialization for instant response.
        This avoids the expensive field extraction process (~200+ operations) on every call.

        Args:
            step_name: Name of the step to get requirements for

        Returns:
            List of field requirement dictionaries for step-specific fields only
        """
        if step_name not in self._config_class_map:
            raise ValueError(f"Step '{step_name}' not found in DAG")

        # Return cached requirements (computed once during __init__)
        if step_name in self._cached_step_requirements:
            return self._cached_step_requirements[step_name]

        # Fallback: if somehow not cached (shouldn't happen), compute now
        logger.warning(
            f"Step requirements not cached for {step_name}, computing now..."
        )
        config_class = self._config_class_map[step_name]
        requirements = self._extract_step_specific_fields(config_class)
        self._cached_step_requirements[step_name] = requirements
        return requirements

    def set_step_config(self, step_name: str, **kwargs) -> BaseModel:
        """
        Set configuration for a specific step with immediate validation.

        Creates and validates the config instance immediately using the proper
        from_base_config pattern, providing early feedback to users.

        Args:
            step_name: Name of the step to configure
            **kwargs: Step-specific configuration field values

        Returns:
            The created and validated config instance

        Raises:
            ValueError: If configuration is invalid or prerequisites not met
        """
        # Accept a bare step name too: if the DAG node carries a job_type suffix
        # (e.g. node "PercentileModelCalibration_calibration") the caller may pass the
        # base step name "PercentileModelCalibration" together with job_type="calibration".
        # Resolve such a bare name to the actual DAG node key before lookup.
        step_name = self._resolve_step_name_to_node(step_name, kwargs.get("job_type"))

        if step_name not in self._config_class_map:
            raise ValueError(f"Step '{step_name}' not found in DAG")

        config_class = self._config_class_map[step_name]

        # Check prerequisites before attempting configuration
        self._validate_prerequisites_for_step(step_name, config_class)

        try:
            # Create instance immediately with validation using proper inheritance
            config_instance = self._create_config_instance_with_inheritance(
                config_class, kwargs
            )

            # Store both raw inputs (for serialization) and validated instance
            self.step_configs[step_name] = kwargs
            self.step_config_instances[step_name] = config_instance

            logger.info(
                f"✅ {step_name} configured successfully using {config_class.__name__}"
            )
            return config_instance

        except Exception as e:
            # Enhanced error message with context for better debugging
            error_context = self._build_error_context(
                step_name, config_class, kwargs, e
            )
            logger.error(f"❌ Configuration failed for {step_name}: {error_context}")
            raise ValueError(
                f"Configuration validation failed for {step_name}: {error_context}"
            )

    def is_dag_step(self, step_name: str, job_type: Optional[str] = None) -> bool:
        """Return True if ``step_name`` (bare or suffixed, optionally + job_type) is a DAG node.

        Lets callers replace the silent ``if "X" in pending_steps:`` guard with an explicit,
        resolution-aware check that honours the same bare→suffixed logic as set_step_config.
        """
        return self._resolve_step_name_to_node(step_name, job_type) in self._config_class_map

    def configure_step_if_present(self, step_name: str, **kwargs) -> Optional[BaseModel]:
        """Configure a step, but WARN (not raise) if it is not a DAG node — a non-silent
        replacement for the ubiquitous ``if "X" in pending_steps: set_step_config("X", ...)``
        notebook guard.

        The bare guard silently skips a mistyped/renamed step name because ``pending_steps`` is
        derived from the DAG nodes — hiding real typos (the campaign found leftover template cells
        in ~7 projects this way). This wrapper resolves the name (bare→suffixed included) and, when
        it matches no DAG node, logs a WARNING and returns ``None`` instead of quietly doing nothing.
        """
        if not self.is_dag_step(step_name, kwargs.get("job_type")):
            logger.warning(
                f"⚠️  set_step_config skipped: '{step_name}' resolves to no DAG node "
                f"(job_type={kwargs.get('job_type')!r}). If this is intentional dead code, remove "
                f"the cell; otherwise it is likely a typo/renamed step — the config will be MISSING."
            )
            return None
        return self.set_step_config(step_name, **kwargs)

    def auto_configure_step_if_possible(self, step_name: str) -> Optional[BaseModel]:
        """
        Auto-configure a step if it only has tier 2+ (optional) fields besides inherited fields.

        This method checks if a step can be configured with just the inherited base config
        fields, without requiring any tier 1 (essential) step-specific fields.

        Args:
            step_name: Name of the step to auto-configure

        Returns:
            The created config instance if auto-configuration succeeded, None otherwise
        """
        if step_name not in self._config_class_map:
            return None

        config_class = self._config_class_map[step_name]

        # Check if prerequisites are met
        try:
            self._validate_prerequisites_for_step(step_name, config_class)
        except ValueError:
            return None  # Prerequisites not met, can't auto-configure

        # Get step-specific requirements (excluding inherited fields)
        step_requirements = self.get_step_requirements(step_name)
        essential_step_fields = [
            req["name"] for req in step_requirements if req["required"]
        ]

        # If there are essential step-specific fields, we can't auto-configure
        if essential_step_fields:
            return None

        # Try to auto-configure with empty step inputs (only inherited fields)
        try:
            config_instance = self._create_config_instance_with_inheritance(
                config_class,
                {},  # Empty step inputs - only use inherited fields
            )

            # Store the auto-configured instance
            self.step_configs[step_name] = {}
            self.step_config_instances[step_name] = config_instance

            logger.info(
                f"✅ {step_name} auto-configured successfully (only tier 2+ fields)"
            )
            return config_instance

        except Exception as e:
            logger.debug(f"Auto-configuration failed for {step_name}: {e}")
            return None

    def get_configuration_status(self) -> Dict[str, bool]:
        """
        Check which configurations have been filled in.

        Returns:
            Dictionary mapping configuration names to completion status
        """
        status = {
            "base_config": self.base_config is not None,
            "base_processing_config": self.base_processing_config is not None
            or not self.get_base_processing_config_requirements(),
        }

        # Add step configuration status
        for step_name in self._config_class_map.keys():
            status[f"step_{step_name}"] = step_name in self.step_configs

        return status

    def validate_dag_config_alignment(self, raise_on_error: bool = True) -> List[str]:
        """Assert the DAG↔config invariant: every DAG node resolves to a config whose derived
        save-key equals the node name, and every configured instance keys back to a DAG node.

        This catches the class of latent drift the multi-pipeline validation campaign found —
        a DAG node whose step TYPE has no matching config (e.g. DAG ``BedrockBatchProcessing_training``
        vs config class ``BedrockProcessing``), which otherwise only surfaces at pipeline-compile
        time as an opaque "no config for node". We check it at generate/save time instead.

        The saved config key is ``{registry_step_name}[_{job_type}]`` (see config ``_derive_step_name``);
        it MUST equal the DAG node key the factory stored the instance under. A mismatch means the
        configured config class is the wrong step TYPE for that node.

        Args:
            raise_on_error: raise ValueError listing all mismatches (default); if False, return them.

        Returns:
            List of human-readable mismatch strings (empty when aligned).
        """
        errors: List[str] = []

        for node_name, instance in self.step_config_instances.items():
            # The key this config will serialize under (merge_and_save_configs uses this).
            derived = None
            if hasattr(instance, "_derive_step_name"):
                try:
                    derived = instance._derive_step_name()
                except Exception as e:  # noqa: BLE001 — a broken derive is itself a finding
                    errors.append(
                        f"node '{node_name}': could not derive its saved config key ({e})"
                    )
                    continue
            if derived is not None and derived != node_name:
                errors.append(
                    f"node '{node_name}': configured instance serializes under key "
                    f"'{derived}' (config class {type(instance).__name__}) — the DAG node key "
                    f"and the saved config key disagree, so the config is the wrong step TYPE "
                    f"for this node (DAG↔config mismatch)."
                )

        # Every DAG node must have mapped to a config class at all.
        for node_name in self._config_class_map:
            if self._config_class_map.get(node_name) is None:
                errors.append(f"node '{node_name}': no config class resolved for this DAG node.")

        if errors and raise_on_error:
            raise ValueError(
                "DAG↔config alignment check failed:\n  - " + "\n  - ".join(errors)
            )
        return errors

    def generate_all_configs(self) -> List[BaseModel]:
        """
        Generate final list of config instances.

        Automatically configures steps that only have tier 2+ fields, then validates
        that all essential steps are configured before generating final instances.

        Returns:
            List of configured instances ready for pipeline execution
        """
        # Auto-configure steps that only have tier 2+ fields
        auto_configured_count = self._auto_configure_eligible_steps()
        if auto_configured_count > 0:
            logger.info(
                f"✅ Auto-configured {auto_configured_count} steps with only tier 2+ fields"
            )

        # Check that all steps are configured (after auto-configuration)
        missing_steps = self.get_pending_steps()
        if missing_steps:
            raise ValueError(f"Missing configuration for steps: {missing_steps}")

        # Enforce the DAG↔config invariant before emitting configs, so a step-type mismatch
        # (DAG node vs configured config class) fails loudly here instead of at pipeline compile.
        self.validate_dag_config_alignment(raise_on_error=True)

        # If we have pre-validated instances for all steps, return them
        if len(self.step_config_instances) == len(self._config_class_map):
            configs = list(self.step_config_instances.values())
            logger.info(
                f"✅ Returning {len(configs)} pre-validated configuration instances"
            )
            return configs

        # Fallback: generate instances using the traditional approach
        # This handles cases where configs were loaded from state or set via old API
        logger.warning(
            "Some configs not pre-validated, falling back to traditional generation"
        )

        if not self.base_config:
            raise ValueError("Base configuration must be set before generating configs")

        if not self.config_generator:
            self.config_generator = ConfigurationGenerator(
                base_config=self.base_config,
                base_processing_config=self.base_processing_config,
            )

        try:
            configs = self.config_generator.generate_all_instances(
                config_class_map=self._config_class_map, step_configs=self.step_configs
            )

            logger.info(
                f"Successfully generated {len(configs)} configuration instances"
            )
            return configs

        except Exception as e:
            logger.error(f"Configuration generation failed: {e}")
            raise ValueError(f"Failed to generate configurations: {e}")

    def _auto_configure_eligible_steps(self) -> int:
        """
        Auto-configure all steps that are eligible (only have tier 2+ fields).

        Returns:
            Number of steps that were auto-configured
        """
        auto_configured_count = 0

        for step_name in self._config_class_map.keys():
            if step_name in self.step_configs:
                continue  # Already configured

            # Try to auto-configure this step
            if self.auto_configure_step_if_possible(step_name):
                auto_configured_count += 1

        return auto_configured_count

    def _validate_essential_fields(self) -> List[str]:
        """
        Validate that all essential (tier 1) fields are provided before config generation.

        This is a guardrail to ensure all required fields are present across:
        1. Base pipeline configuration
        2. Base processing configuration (if needed)
        3. All step-specific configurations

        Returns:
            List of validation error messages (empty if validation passes)
        """
        validation_errors = []

        # 1. Validate base configuration essential fields
        if not self.base_config:
            validation_errors.append(
                "Base pipeline configuration is required but not set"
            )
        else:
            # Check if all essential fields in base config are provided
            base_requirements = self.get_base_config_requirements()
            essential_base_fields = [
                req["name"] for req in base_requirements if req["required"]
            ]

            for field_name in essential_base_fields:
                field_value = getattr(self.base_config, field_name, None)
                if field_value is None or (
                    isinstance(field_value, str) and not field_value.strip()
                ):
                    validation_errors.append(
                        f"Essential base config field '{field_name}' is missing or empty"
                    )

        # 2. Validate base processing configuration if needed
        processing_requirements = self.get_base_processing_config_requirements()
        if processing_requirements:  # Processing config is needed
            if not self.base_processing_config:
                validation_errors.append(
                    "Base processing configuration is required but not set"
                )
            else:
                essential_processing_fields = [
                    req["name"] for req in processing_requirements if req["required"]
                ]

                for field_name in essential_processing_fields:
                    field_value = getattr(self.base_processing_config, field_name, None)
                    if field_value is None or (
                        isinstance(field_value, str) and not field_value.strip()
                    ):
                        validation_errors.append(
                            f"Essential processing config field '{field_name}' is missing or empty"
                        )

        # 3. Validate step-specific essential fields
        for step_name, config_class in self._config_class_map.items():
            if step_name not in self.step_configs:
                validation_errors.append(f"Step '{step_name}' configuration is missing")
                continue

            step_requirements = self.get_step_requirements(step_name)
            essential_step_fields = [
                req["name"] for req in step_requirements if req["required"]
            ]
            provided_step_fields = self.step_configs[step_name]

            for field_name in essential_step_fields:
                if field_name not in provided_step_fields:
                    validation_errors.append(
                        f"Essential field '{field_name}' missing for step '{step_name}'"
                    )
                else:
                    field_value = provided_step_fields[field_name]
                    if field_value is None or (
                        isinstance(field_value, str) and not field_value.strip()
                    ):
                        validation_errors.append(
                            f"Essential field '{field_name}' is empty for step '{step_name}'"
                        )

        return validation_errors

    def _extract_step_specific_fields(
        self, config_class: Type[BaseModel]
    ) -> List[Dict[str, Any]]:
        """
        Extract step-specific fields excluding inherited base config fields.

        Args:
            config_class: Step configuration class to extract fields from

        Returns:
            List of field requirement dictionaries for step-specific fields only
        """
        try:
            from ...core.base.config_base import BasePipelineConfig
            from ...steps.configs.config_processing_step_base import (
                ProcessingStepConfigBase,
            )

            # Determine the appropriate base class to exclude fields from
            if self._inherits_from_processing_config(config_class):
                # If step inherits from ProcessingStepConfigBase, exclude those fields
                base_class = ProcessingStepConfigBase
            else:
                # Otherwise, exclude BasePipelineConfig fields
                base_class = BasePipelineConfig

            return extract_non_inherited_fields(config_class, base_class)

        except ImportError:
            logger.warning("Base config classes not found, extracting all fields")
            return extract_field_requirements(config_class)

    def _inherits_from_processing_config(self, config_class: Type[BaseModel]) -> bool:
        """
        Check if config class inherits from ProcessingStepConfigBase.

        Uses cached inheritance information computed during initialization for instant response.

        Args:
            config_class: Configuration class to check

        Returns:
            True if class inherits from ProcessingStepConfigBase
        """
        # Find which step uses this config class and return cached result
        class_name = config_class.__name__
        for step_name, step_config_class in self._config_class_map.items():
            if step_config_class.__name__ == class_name:
                if step_name in self._cached_inheritance_info:
                    return self._cached_inheritance_info[step_name][
                        "inherits_processing"
                    ]

        # Fallback: if not cached (shouldn't happen after init), compute now
        logger.warning(f"Inheritance not cached for {class_name}, computing now...")
        return self._check_processing_inheritance(config_class)

    def get_factory_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the factory state.

        Returns:
            Dictionary with factory state summary
        """
        status = self.get_configuration_status()

        return {
            "dag_steps": len(self._config_class_map),
            "mapped_config_classes": list(self._config_class_map.keys()),
            "configuration_status": status,
            "completed_steps": len(
                [k for k, v in status.items() if k.startswith("step_") and v]
            ),
            "pending_steps": self.get_pending_steps(),
            "base_config_set": self.base_config is not None,
            "processing_config_set": self.base_processing_config is not None,
            "ready_for_generation": all(status.values()),
        }

    def save_partial_state(self, file_path: str) -> None:
        """
        Save current factory state for later restoration.

        Args:
            file_path: Path to save the state file
        """
        import json
        from pathlib import Path

        state = {
            "step_configs": self.step_configs,
            "base_config_dict": self.config_generator._extract_config_values(
                self.base_config
            )
            if self.base_config
            else None,
            "base_processing_config_dict": self.config_generator._extract_config_values(
                self.base_processing_config
            )
            if self.base_processing_config
            else None,
            "config_class_map": {
                k: v.__name__ for k, v in self._config_class_map.items()
            },
        }

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"Factory state saved to: {file_path}")

    def load_partial_state(self, file_path: str) -> None:
        """
        Load previously saved factory state.

        Args:
            file_path: Path to the saved state file
        """
        import json

        with open(file_path, "r") as f:
            state = json.load(f)

        # Restore step configs
        self.step_configs = state.get("step_configs", {})

        # Restore base configs if available
        if state.get("base_config_dict"):
            self.set_base_config(**state["base_config_dict"])

        if state.get("base_processing_config_dict"):
            self.set_base_processing_config(**state["base_processing_config_dict"])

        logger.info(f"Factory state loaded from: {file_path}")

    def _validate_prerequisites_for_step(
        self, step_name: str, config_class: Type[BaseModel]
    ) -> None:
        """
        Validate that required base configs are set before step configuration.

        Args:
            step_name: Name of the step being configured
            config_class: Configuration class for the step

        Raises:
            ValueError: If required base configurations are missing
        """
        logger.debug(f"        → _validate_prerequisites_for_step({step_name})")
        logger.debug(f"          Checking if inherits from processing config...")

        if self._inherits_from_processing_config(config_class):
            logger.debug(f"          ✓ Inherits from processing config")
            if not self.base_config:
                raise ValueError(
                    f"Step '{step_name}' requires base config to be set first"
                )
            if not self.base_processing_config:
                raise ValueError(
                    f"Step '{step_name}' requires base processing config to be set first"
                )
        else:
            logger.debug(f"          Checking if inherits from base config...")
            if self._inherits_from_base_config(config_class):
                logger.debug(f"          ✓ Inherits from base config")
                if not self.base_config:
                    raise ValueError(
                        f"Step '{step_name}' requires base config to be set first"
                    )
            else:
                logger.debug(f"          Standalone config class")

        logger.debug(f"        ← _validate_prerequisites_for_step() complete")

    def _create_config_instance_with_inheritance(
        self, config_class: Type[BaseModel], step_inputs: Dict[str, Any]
    ) -> BaseModel:
        """
        Create config instance using proper from_base_config pattern with inheritance.

        Args:
            config_class: Configuration class to instantiate
            step_inputs: Step-specific input values

        Returns:
            Configuration instance with proper inheritance applied
        """
        # Ensure config generator is available
        if not self.config_generator:
            self.config_generator = ConfigurationGenerator(
                base_config=self.base_config,
                base_processing_config=self.base_processing_config,
            )

        # Determine inheritance strategy and create instance
        if self._inherits_from_processing_config(config_class):
            return self._create_with_processing_inheritance(config_class, step_inputs)
        elif self._inherits_from_base_config(config_class):
            return self._create_with_base_inheritance(config_class, step_inputs)
        else:
            # Standalone configuration class
            return config_class(**step_inputs)

    def _create_with_processing_inheritance(
        self, config_class: Type[BaseModel], step_inputs: Dict[str, Any]
    ) -> BaseModel:
        """
        Create config instance with processing config inheritance using from_base_config.

        Args:
            config_class: Configuration class that inherits from ProcessingStepConfigBase
            step_inputs: Step-specific input values

        Returns:
            Configuration instance with processing inheritance applied
        """
        # Try from_base_config first (preferred method)
        if hasattr(config_class, "from_base_config"):
            try:
                return config_class.from_base_config(
                    self.base_processing_config, **step_inputs
                )
            except Exception as e:
                logger.warning(
                    f"from_base_config failed for {config_class.__name__}: {e}"
                )
                # Fall through to manual combination

        # Fallback: combine all inputs manually
        combined_inputs = {}
        if self.base_config:
            combined_inputs.update(
                self.config_generator._extract_config_values(self.base_config)
            )
        if self.base_processing_config:
            combined_inputs.update(
                self.config_generator._extract_config_values(
                    self.base_processing_config
                )
            )
        combined_inputs.update(step_inputs)

        return config_class(**combined_inputs)

    def _create_with_base_inheritance(
        self, config_class: Type[BaseModel], step_inputs: Dict[str, Any]
    ) -> BaseModel:
        """
        Create config instance with base config inheritance using from_base_config.

        Args:
            config_class: Configuration class that inherits from BasePipelineConfig
            step_inputs: Step-specific input values

        Returns:
            Configuration instance with base inheritance applied
        """
        # Try from_base_config first (preferred method)
        if hasattr(config_class, "from_base_config"):
            try:
                return config_class.from_base_config(self.base_config, **step_inputs)
            except Exception as e:
                logger.warning(
                    f"from_base_config failed for {config_class.__name__}: {e}"
                )
                # Fall through to manual combination

        # Fallback: combine inputs manually
        combined_inputs = {}
        if self.base_config:
            combined_inputs.update(
                self.config_generator._extract_config_values(self.base_config)
            )
        combined_inputs.update(step_inputs)

        return config_class(**combined_inputs)

    def _inherits_from_base_config(self, config_class: Type[BaseModel]) -> bool:
        """
        Check if config class inherits from BasePipelineConfig.

        Uses cached inheritance information computed during initialization for instant response.

        Args:
            config_class: Configuration class to check

        Returns:
            True if class inherits from BasePipelineConfig
        """
        # Find which step uses this config class and return cached result
        class_name = config_class.__name__
        for step_name, step_config_class in self._config_class_map.items():
            if step_config_class.__name__ == class_name:
                if step_name in self._cached_inheritance_info:
                    return self._cached_inheritance_info[step_name]["inherits_base"]

        # Fallback: if not cached (shouldn't happen after init), compute now
        logger.warning(f"Inheritance not cached for {class_name}, computing now...")
        return self._check_base_inheritance(config_class)

    def _build_error_context(
        self,
        step_name: str,
        config_class: Type[BaseModel],
        step_inputs: Dict[str, Any],
        error: Exception,
    ) -> str:
        """
        Build detailed error context for better debugging.

        Args:
            step_name: Name of the step that failed
            config_class: Configuration class that failed
            step_inputs: Input values that were provided
            error: The exception that occurred

        Returns:
            Detailed error context string
        """
        context_parts = [
            f"Step: {step_name}",
            f"Config Class: {config_class.__name__}",
            f"Has from_base_config: {hasattr(config_class, 'from_base_config')}",
            f"Inherits from processing: {self._inherits_from_processing_config(config_class)}",
            f"Inherits from base: {self._inherits_from_base_config(config_class)}",
            f"Step inputs: {list(step_inputs.keys())}",
            f"Error: {str(error)}",
        ]

        return " | ".join(context_parts)

    def update_step_config(self, step_name: str, **kwargs) -> BaseModel:
        """
        Update existing step configuration with new values.

        Args:
            step_name: Name of the step to update
            **kwargs: New configuration values to merge

        Returns:
            Updated and validated config instance

        Raises:
            ValueError: If step not configured yet or update fails
        """
        if step_name not in self.step_configs:
            raise ValueError(
                f"Step '{step_name}' not configured yet. Use set_step_config first."
            )

        # Merge with existing configuration
        updated_inputs = {**self.step_configs[step_name], **kwargs}

        # Use set_step_config to validate and update
        return self.set_step_config(step_name, **updated_inputs)

    def get_step_config_instance(self, step_name: str) -> Optional[BaseModel]:
        """
        Get the validated config instance for a step.

        Args:
            step_name: Name of the step

        Returns:
            The validated config instance, or None if not configured
        """
        return self.step_config_instances.get(step_name)

    def get_all_config_instances(self) -> Dict[str, BaseModel]:
        """
        Get all validated config instances.

        Returns:
            Dictionary mapping step names to validated config instances
        """
        return self.step_config_instances.copy()
