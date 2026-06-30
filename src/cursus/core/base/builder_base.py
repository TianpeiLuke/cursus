from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
import logging
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import Step
from sagemaker.workflow.steps import CacheConfig

# Import dependency resolver (with error handling for backward compatibility)
if TYPE_CHECKING:
    from ..deps.dependency_resolver import UnifiedDependencyResolver
    from ..deps.registry_manager import RegistryManager
    from ..deps.semantic_matcher import SemanticMatcher
    from ..deps.factory import create_dependency_resolver, create_pipeline_components
    from ..deps.property_reference import PropertyReference

    DEPENDENCY_RESOLVER_AVAILABLE = True
else:
    try:
        from ..deps.dependency_resolver import UnifiedDependencyResolver
        from ..deps.registry_manager import RegistryManager
        from ..deps.semantic_matcher import SemanticMatcher
        from ..deps.factory import (
            create_dependency_resolver,
            create_pipeline_components,
        )
        from ..deps.property_reference import PropertyReference

        DEPENDENCY_RESOLVER_AVAILABLE = True
    except ImportError:
        DEPENDENCY_RESOLVER_AVAILABLE = False
        # Create placeholder classes for runtime
        UnifiedDependencyResolver = Any
        RegistryManager = Any
        SemanticMatcher = Any
        PropertyReference = Any
        create_dependency_resolver = None
        create_pipeline_components = None
        logger = logging.getLogger(__name__)
        logger.warning("Dependency resolver not available, using traditional methods")

# Import for type hints only
if TYPE_CHECKING:
    from .step_interface import StepInterface
else:
    # Just for runtime use, won't affect type checking
    StepInterface = Any

# Import BasePipelineConfig for type hints only to break circular dependency
if TYPE_CHECKING:
    from .config_base import BasePipelineConfig
else:
    # Just for runtime use, won't affect type checking
    BasePipelineConfig = Any

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


class StepBuilderBase(ABC):
    """
    Base class for all step builders

    ## Safe Logging Methods

    To handle Pipeline variables safely in logs, use these methods:

    ```python
    # Instead of:
    logger.info(f"Using input path: {input_path}")  # May raise TypeError for Pipeline variables

    # Use:
    self.log_info("Using input path: %s", input_path)  # Handles Pipeline variables safely
    ```

    Standard Pattern for `input_names` and `output_names`:

    1. In **config classes**:
       ```python
       output_names = {"logical_name": "DescriptiveValue"}  # VALUE used as key in outputs dict
       input_names = {"logical_name": "ScriptInputName"}    # KEY used as key in inputs dict
       ```

    2. In **pipeline code**:
       ```python
       # Get output using VALUE from output_names
       output_value = step_a.config.output_names["logical_name"]
       output_uri = step_a.properties.ProcessingOutputConfig.Outputs[output_value].S3Output.S3Uri

       # Set input using KEY from input_names
       inputs = {"logical_name": output_uri}
       ```

    3. In **step builders**:
       ```python
       # For outputs - validate using VALUES
       value = self.config.output_names["logical_name"]
       if value not in outputs:
           raise ValueError(f"Must supply an S3 URI for '{value}'")

       # For inputs - validate using KEYS
       for logical_name in self.config.input_names.keys():
           if logical_name not in inputs:
               raise ValueError(f"Must supply an S3 URI for '{logical_name}'")
       ```

    Developers should follow this standard pattern when creating new step builders.
    The base class provides helper methods to enforce and simplify this pattern:

    - `_validate_inputs()`: Validates inputs using KEYS from input_names
    - `_validate_outputs()`: Validates outputs using VALUES from output_names
    - `_get_script_input_name()`: Maps logical name to script input name
    - `_get_output_destination_name()`: Maps logical name to output destination name
    - `_create_standard_processing_input()`: Creates standardized ProcessingInput
    - `_create_standard_processing_output()`: Creates standardized ProcessingOutput

    Property Path Registry:

    To bridge the gap between definition-time and runtime, step builders can register
    property paths that define how to access their outputs at runtime. This solves the
    issue where outputs are defined statically but only accessible via specific runtime paths.

    - `register_property_path()`: Registers a property path for a logical output name
    - `get_property_paths()`: Gets all registered property paths for this step
    """

    REGION_MAPPING: Dict[str, str] = {
        "NA": "us-east-1",
        "EU": "eu-west-1",
        "FE": "us-west-2",
    }

    #: The canonical registry step name for THIS builder (singular), e.g. "XGBoostTraining".
    #: This is the authoritative identity slot ``_get_step_name`` reads (FZ 31e1d3g3 Phase C1):
    #: every routed shell + ``TemplateStepBuilder`` sets it, the materializer stamps it on synthesized
    #: fileless builders, and once the per-step shell classes are deleted it is the ONLY reliable
    #: canonical key (the class name collapses to ``TemplateStepBuilder``). Declared here on the root
    #: so the base method that reads it owns the slot, and so it is not confused with ``STEP_NAMES``
    #: (PLURAL — the whole registry dict, below). ``None`` on a hand-written builder, which falls back
    #: to the legacy ``<Name>StepBuilder`` class-name convention.
    STEP_NAME: Optional[str] = None

    @property
    def STEP_NAMES(self) -> Dict[str, Any]:
        """
        Lazy load step names with workspace context awareness.

        This property now supports workspace-aware step name resolution by:
        1. Extracting workspace context from config or environment
        2. Using hybrid registry manager for workspace-specific step names
        3. Falling back to traditional registry if hybrid is unavailable
        4. Maintaining backward compatibility with existing code

        Returns:
            Dict[str, str]: Step names mapping for the current workspace context
        """
        if not hasattr(self, "_step_names"):
            try:
                # Get workspace context
                workspace_context = self._get_workspace_context()

                # Try to use hybrid registry manager first
                try:
                    from ...registry.hybrid.manager import HybridRegistryManager

                    hybrid_manager = HybridRegistryManager()

                    # Get step names using the actual available method
                    legacy_dict = hybrid_manager.create_legacy_step_names_dict(
                        workspace_context or "default"
                    )
                    self._step_names = legacy_dict

                    if workspace_context:
                        self.log_debug(
                            f"Loaded workspace-specific step names for context: {workspace_context}"
                        )
                    else:
                        self.log_debug("Loaded default step names from hybrid registry")

                except ImportError:
                    # Fallback to traditional registry
                    self.log_debug(
                        "Hybrid registry not available, falling back to traditional registry"
                    )
                    from ...registry.step_names import BUILDER_STEP_NAMES

                    self._step_names = BUILDER_STEP_NAMES  # type: ignore[assignment]

            except ImportError:
                # Final fallback if all imports fail
                self.log_warning("No registry available, using empty step names")
                self._step_names = {}

        return self._step_names

    def _get_workspace_context(self) -> Optional[str]:
        """
        Extract workspace context from configuration or environment variables.

        This method determines the current workspace context by checking:
        1. Config object for workspace-related attributes
        2. Environment variables for workspace identification
        3. Pipeline name as workspace identifier
        4. Returns None for default/global workspace

        Returns:
            Optional[str]: Workspace context identifier or None for default
        """
        # Check config for explicit workspace context
        if hasattr(self.config, "workspace_context") and self.config.workspace_context:
            return str(self.config.workspace_context)

        # Check config for workspace attribute
        if hasattr(self.config, "workspace") and self.config.workspace:
            return str(self.config.workspace)

        # Check environment variables
        import os

        workspace_env = os.environ.get("CURSUS_WORKSPACE_CONTEXT")
        if workspace_env:
            return workspace_env

        # Use pipeline name as workspace context if available
        if hasattr(self.config, "pipeline_name") and self.config.pipeline_name:
            return str(self.config.pipeline_name)

        # Check for project-specific context
        if hasattr(self.config, "project_name") and self.config.project_name:
            return str(self.config.project_name)

        # Return None for default/global workspace
        return None

    # Common properties that all steps might need
    COMMON_PROPERTIES = {
        "dependencies": "Optional list of dependent steps",
        "enable_caching": "Whether to enable caching for this step (default: True)",
    }

    # Standard output properties for training steps
    TRAINING_OUTPUT_PROPERTIES = {
        "training_job_name": "Name of the training job",
        "model_data": "S3 path to the model artifacts",
        "model_data_url": "S3 URL to the model artifacts",
    }

    # Standard output properties for model steps
    MODEL_OUTPUT_PROPERTIES = {
        "model_artifacts_path": "S3 path to model artifacts",
        "model": "SageMaker model object",
    }

    def __init__(
        self,
        config: BasePipelineConfig,
        spec: Optional[StepInterface] = None,  # New parameter
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        registry_manager: Optional[RegistryManager] = None,
        dependency_resolver: Optional[UnifiedDependencyResolver] = None,
    ):
        """
        Initialize base step builder.

        Args:
            config: Model configuration
            spec: Optional step specification for specification-driven implementation
            sagemaker_session: SageMaker session
            role: IAM role
            registry_manager: Optional registry manager for dependency injection
            dependency_resolver: Optional dependency resolver for dependency injection
        """
        self.config = config
        self.spec = spec  # Store the specification
        self.session = sagemaker_session
        self.role = role
        self._registry_manager = registry_manager
        self._dependency_resolver = dependency_resolver
        self.execution_prefix: Optional[Union[str, Any]] = (
            None  # Initialize execution prefix for PIPELINE_EXECUTION_TEMP_DIR support
        )

        # Get contract from specification if available, or directly from config
        self.contract = getattr(spec, "script_contract", None) if spec else None
        if not self.contract and hasattr(self.config, "script_contract"):
            self.contract = self.config.script_contract

        # Validate and set AWS region
        self.aws_region = self.REGION_MAPPING.get(self.config.region)
        if not self.aws_region:
            raise ValueError(
                f"Invalid region code: {self.config.region}. "
                f"Must be one of: {', '.join(self.REGION_MAPPING.keys())}"
            )

        # Validate specification-contract alignment if both are provided
        if (
            self.spec
            and self.contract
            and hasattr(self.spec, "validate_contract_alignment")
        ):
            result = self.spec.validate_contract_alignment()
            if not result.is_valid:
                raise ValueError(f"Spec-Contract alignment errors: {result.errors}")

        logger.info(
            f"Initializing {self.__class__.__name__} with region: {self.config.region}"
        )
        self.validate_configuration()

    def _sanitize_name_for_sagemaker(self, name: str, max_length: int = 63) -> str:
        """
        Sanitize a string to be a valid SageMaker resource name component.

        Args:
            name: Name to sanitize
            max_length: Maximum length of sanitized name

        Returns:
            Sanitized name
        """
        if not name:
            return "default-name"
        sanitized = "".join(c if c.isalnum() else "-" for c in str(name))
        sanitized = "-".join(filter(None, sanitized.split("-")))
        return sanitized[:max_length].rstrip("-")

    def _get_step_name(self, include_job_type: bool = True) -> str:
        """
        Get standard step name, optionally including job_type.

        Resolution order (FZ 31e1d3g3 Phase C1):
        1. The ``STEP_NAME`` class/instance attribute, when set — the AUTHORITATIVE source. Every
           routed shell (and the synthesized fileless builders) declares it, and ``TemplateStepBuilder``
           defines it. This is required for the factory end-state: once the 45 per-step shell classes
           are deleted, ``self.__class__.__name__`` collapses to ``TemplateStepBuilder`` (or the
           synthesized ``<Name>StepBuilder``), so the class name is no longer a reliable canonical key.
        2. Otherwise, fall back to the legacy convention: strip the ``StepBuilder`` suffix off the
           class name (``XGBoostTrainingStepBuilder`` -> ``XGBoostTraining``). For every hand-written
           builder this equals its ``STEP_NAME``, so the change is a behavior-preserving no-op today.

        Args:
            include_job_type: Whether to include job_type suffix if available in config

        Returns:
            The canonical step name, optionally with job_type suffix
        """
        # Prefer the declared STEP_NAME (set on every shell + TemplateStepBuilder) — robust to the
        # class-name collapse once the per-step shells are deleted.
        canonical_name = getattr(self, "STEP_NAME", None)
        if not canonical_name:
            class_name = self.__class__.__name__
            # If class name follows the standard pattern, extract the registry key
            if class_name.endswith("StepBuilder"):
                canonical_name = class_name[:-11]  # Remove "StepBuilder" suffix
            else:
                # Fallback for non-standard class names
                self.log_warning(
                    f"Class name '{class_name}' doesn't follow the convention. Using as is."
                )
                canonical_name = class_name

        # Validate that the extracted name exists in the registry
        if canonical_name not in self.STEP_NAMES:
            self.log_warning(f"Unknown step type: {canonical_name}. Using as is.")

        # Add job_type suffix if requested and available
        if (
            include_job_type
            and hasattr(self.config, "job_type")
            and self.config.job_type
        ):
            return f"{canonical_name}-{self.config.job_type.capitalize()}"

        return canonical_name

    def _generate_job_name(self, step_type: Optional[str] = None) -> str:
        """
        Generate a standardized job name for SageMaker processing/training jobs.

        This method automatically determines the step type from the class name
        if not provided, using the _get_step_name method. It adds a timestamp
        to ensure uniqueness across executions.

        Args:
            step_type: Optional type of step. If not provided, it will be
                      determined automatically using _get_step_name.

        Returns:
            Sanitized job name suitable for SageMaker
        """
        import time

        # If step_type is not provided, use our simplified _get_step_name method
        if step_type is None:
            step_type = self._get_step_name()

        # Generate a timestamp for uniqueness (unix timestamp in seconds)
        timestamp = int(time.time())

        # Build the job name
        if hasattr(self.config, "job_type") and self.config.job_type:
            job_name = f"{step_type}-{self.config.job_type.capitalize()}-{timestamp}"
        else:
            job_name = f"{step_type}-{timestamp}"

        # Sanitize and return
        return self._sanitize_name_for_sagemaker(job_name)

    def get_property_path(
        self, logical_name: str, format_args: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Get property path for an output using the specification.

        This method retrieves the property path for an output from the specification.
        It also supports template formatting if format_args are provided.

        Args:
            logical_name: Logical name of the output
            format_args: Optional dictionary of format arguments for template paths
                        (e.g., {'output_descriptor': 'data'} for paths with placeholders)

        Returns:
            Property path from specification, formatted with args if provided,
            or None if not found
        """
        property_path = None

        # Get property path from specification outputs
        if self.spec and hasattr(self.spec, "outputs"):
            for _, output_spec in self.spec.outputs.items():
                if (
                    output_spec.logical_name == logical_name
                    and output_spec.property_path
                ):
                    property_path = output_spec.property_path
                    break

        if not property_path:
            return None

        # If found and format args are provided, format the path
        if format_args:
            try:
                property_path = property_path.format(**format_args)
            except KeyError as e:
                logger.warning(
                    f"Missing format key {e} for property path template: {property_path}"
                )
            except Exception as e:
                logger.warning(f"Error formatting property path: {e}")

        return property_path

    def get_all_property_paths(self) -> Dict[str, str]:
        """
        Get all property paths defined in the specification.

        Returns:
            dict: Mapping from logical output names to runtime property paths
        """
        paths = {}
        if self.spec and hasattr(self.spec, "outputs"):
            for _, output_spec in self.spec.outputs.items():
                if output_spec.property_path:
                    paths[output_spec.logical_name] = output_spec.property_path

        return paths

    def log_info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Safely log info messages, handling Pipeline variables.

        Args:
            message: The log message
            *args, **kwargs: Values to format into the message
        """
        try:
            # Convert args to safe strings
            safe_args = [safe_value_for_logging(arg) for arg in args]

            # Log with safe values (logger.info doesn't accept **kwargs)
            logger.info(message, *safe_args)
        except Exception as e:
            logger.info(
                f"Original logging failed ({e}), logging raw message: {message}"
            )

    def log_debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Debug version of safe logging"""
        try:
            safe_args = [safe_value_for_logging(arg) for arg in args]
            logger.debug(message, *safe_args)
        except Exception as e:
            logger.debug(
                f"Original logging failed ({e}), logging raw message: {message}"
            )

    def log_warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Warning version of safe logging"""
        try:
            safe_args = [safe_value_for_logging(arg) for arg in args]
            logger.warning(message, *safe_args)
        except Exception as e:
            logger.warning(
                f"Original logging failed ({e}), logging raw message: {message}"
            )

    def log_error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Error version of safe logging"""
        try:
            safe_args = [safe_value_for_logging(arg) for arg in args]
            logger.error(message, *safe_args)
        except Exception as e:
            logger.error(
                f"Original logging failed ({e}), logging raw message: {message}"
            )

    def _get_cache_config(self, enable_caching: bool = True) -> CacheConfig:
        """
        Get cache configuration for step.
        ProcessingStep.to_request() can call .config safely.

        Args:
            enable_caching: Whether to enable caching

        Returns:
            Cache configuration dictionary
        """
        return CacheConfig(enable_caching=enable_caching, expire_after="P30D")

    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Build the container environment variables — config is the single source (FZ 31e1d3g).

        The step interface (``.step.yaml`` ``env_vars``) DECLARES which env vars the step uses; the
        config INSTANCE supplies the VALUES via ``config.get_environment_variables(declared_names)``.
        Composition (the one template, per the env single-source plan):
          1. ``config.get_environment_variables(<declared names>)`` — interface-declared names resolved
             against config (convention ``NAME`` -> ``self.name``, else the config's ``_env_overrides``).
             A config with a bespoke collector may ignore the names and return its full dict.
          2. interface defaults for any declared-optional var the config did not produce (so an
             unset optional still gets its ``.step.yaml`` default).
          3. ``config.env`` explicit overrides last.

        Per-step ``_get_environment_variables`` overrides are being retired in favor of this one
        method + the config collector. Steps not yet migrated keep their override (it wins via MRO).
        """
        import inspect as _inspect

        declared_required: List[str] = []
        declared_optional: Dict[str, str] = {}
        if getattr(self, "contract", None) is not None:
            declared_required = list(
                getattr(self.contract, "required_env_vars", []) or []
            )
            declared_optional = dict(
                getattr(self.contract, "optional_env_vars", {}) or {}
            )
        declared_names = declared_required + list(declared_optional)

        env_vars: Dict[str, str] = {}
        cfg_type = type(self.config)

        # A config exposes its env values through ONE of three collector shapes, in priority order:
        #   1. a BESPOKE ``get_environment_variables`` method defined on the config's own class
        #      (computed values, e.g. JSON/Join) — call it; pass declared names if it accepts them.
        #   2. a bespoke ``environment_variables`` PROPERTY defined on the config's own class.
        #   3. the inherited generic names-driven resolver on BasePipelineConfig (NAME -> self.name).
        # We prefer a config-OWNED collector over the inherited resolver so a config's bespoke env
        # logic is never silently bypassed.
        def _own(attr):
            # True only if a class STRICTLY BELOW BasePipelineConfig defines `attr` — i.e. a
            # config-specific collector, not the generic resolver that now lives on the base.
            for klass in cfg_type.__mro__:
                if klass.__name__ in ("BasePipelineConfig", "BaseModel", "object"):
                    break
                if attr in klass.__dict__:
                    return True
            return False

        produced = None
        if _own("get_environment_variables"):
            collector = self.config.get_environment_variables
            try:
                accepts_arg = any(
                    p.kind
                    in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
                    for p in _inspect.signature(collector).parameters.values()
                )
            except (ValueError, TypeError):
                accepts_arg = False
            produced = collector(declared_names) if accepts_arg else collector()
        elif _own("environment_variables"):
            produced = getattr(self.config, "environment_variables", None)
        else:
            # Inherited BasePipelineConfig resolver — interface names drive the keys.
            collector = getattr(self.config, "get_environment_variables", None)
            if callable(collector):
                produced = collector(declared_names)
        if isinstance(produced, dict):
            env_vars.update(produced)

        # Diagnostics: a declared-required var the config did not supply is a likely misconfig.
        for env_var in declared_required:
            if env_var not in env_vars:
                self.log_warning(
                    f"Required environment variable '{env_var}' not found in config"
                )

        # Interface defaults for declared-optional vars the config did not supply.
        for env_var, default_value in declared_optional.items():
            if env_var not in env_vars:
                env_vars[env_var] = default_value
                self.log_debug(
                    f"Using default value for optional environment variable '{env_var}': {default_value}"
                )

        # COMPUTED-S3-ENV pattern (FZ 31e1d3g3 Phase A3): env vars whose value is an S3 sub-path under
        # the pipeline execution prefix (base_output_path), declared in contract.computed_env_paths as
        # {ENV_VAR: [segment, ...]}. Set ENV_VAR = Join(base_output_path, *segments). This is the
        # declarative replacement for hand-written _get_environment_variables overrides that built
        # runtime S3 paths (e.g. Bedrock's BEDROCK_BATCH_INPUT/OUTPUT_S3_PATH).
        computed_env_paths = {}
        if getattr(self, "contract", None) is not None:
            raw = getattr(self.contract, "computed_env_paths", None)
            if isinstance(raw, dict):
                computed_env_paths = raw
        if computed_env_paths:
            from sagemaker.workflow.functions import Join

            base_output_path = self._get_base_output_path()
            for env_var, segments in computed_env_paths.items():
                env_vars[env_var] = Join(
                    on="/", values=[base_output_path, *list(segments)]
                )
                self.log_debug(
                    f"Computed S3-path env var '{env_var}' from base_output_path + {segments}"
                )

        # Explicit per-config overrides win last.
        explicit = getattr(self.config, "env", None)
        if isinstance(explicit, dict) and explicit:
            env_vars.update(explicit)

        return env_vars

    def _get_job_arguments(self) -> Optional[List[str]]:
        """
        Build the script's CLI arguments — config is the single source (FZ 31e1d3h).

        Delegates entirely to ``config.get_job_arguments()``: the base config returns ``None`` (no
        args), and a step-config that passes args to its script overrides it (the common
        ``--job_type`` case via ``self._job_type_arg()``; bespoke configs build their own list).
        This mirrors the env single-source model — config owns the VALUES.

        Per-step ``_get_job_arguments`` overrides are being retired in favor of config-side
        ``get_job_arguments()``; a step not yet migrated keeps its override (it wins via MRO).
        Returns ``None`` (not ``[]``) when there are no args — the contract the SDK expects.
        """
        getter = getattr(self.config, "get_job_arguments", None)
        if not callable(getter):
            return None
        produced = getter()
        return list(produced) if produced else None

    def _processing_instance_type(self, spec) -> str:
        """Resolve the processing instance type per the compute descriptor's ``instance_size_mode``.

        ``large_or_small`` (the near-universal pattern) picks large vs small by
        ``use_large_processing_instance``; otherwise a single fixed field.
        """
        if spec.instance_size_mode == "large_or_small":
            return (
                self.config.processing_instance_type_large
                if self.config.use_large_processing_instance
                else self.config.processing_instance_type_small
            )
        return self.config.processing_instance_type

    def _create_compute(
        self,
        output_path: Optional[str] = None,
        *,
        model_data: Optional[Any] = None,
        model_name: Optional[Any] = None,
    ) -> Any:
        """Build the step's compute object (processor / estimator / model / transformer) from the
        declarative ``contract.compute`` descriptor + config (FZ 31e1d3k).

        Every value is a config field; the descriptor only says WHICH SDK class and WHICH fields.
        This is the single template factory that replaces the near-identical per-step
        ``_create_processor`` / ``_create_estimator`` / ``_create_model`` / ``_create_transformer``
        overrides. A step keeps its own factory only if it does NOT declare ``compute.kind`` (then the
        handler uses the legacy hook).

        Extra runtime args are threaded per verb: ``output_path`` (estimator/transformer),
        ``model_data`` (model — from ModelCreationHandler), ``model_name`` (transformer — from
        TransformHandler). Each handler passes only what its verb produces.
        """
        spec = getattr(self.contract, "compute", None)
        kind = getattr(spec, "kind", None)
        if not kind:
            raise NotImplementedError(
                "No compute descriptor (contract.compute.kind) and no _create_processor/"
                "_create_estimator override for this step."
            )
        cfg = self.config
        if spec.framework_version_field:
            # getattr-with-default: several steps fall back to a per-step framework default when the
            # config lacks the field (e.g. processing_framework_version -> "1.0-1"/"1.2-1").
            if spec.framework_version_default is not None:
                fw = getattr(
                    cfg, spec.framework_version_field, spec.framework_version_default
                )
            else:
                fw = getattr(cfg, spec.framework_version_field)
        else:
            fw = None
        py = getattr(cfg, spec.py_version_field) if spec.py_version_field else None
        job_name = self._generate_job_name()
        env = self._get_environment_variables()

        if kind in ("sklearn", "xgboost", "framework", "script"):
            instance_type = self._processing_instance_type(spec)
            common = dict(
                role=self.role,
                instance_type=instance_type,
                instance_count=cfg.processing_instance_count,
                volume_size_in_gb=cfg.processing_volume_size,
                base_job_name=job_name,
                sagemaker_session=self.session,
                env=env,
            )
            if kind == "sklearn":
                from sagemaker.sklearn import SKLearnProcessor

                return SKLearnProcessor(framework_version=fw, **common)
            if kind == "xgboost":
                from sagemaker.xgboost import XGBoostProcessor

                return XGBoostProcessor(framework_version=fw, **common)
            if kind == "framework":
                from sagemaker.processing import FrameworkProcessor

                est = self._resolve_sdk_class(spec.sdk_class)
                fw_kwargs = dict(estimator_cls=est, framework_version=fw, **common)
                # Only pass py_version when declared — some factories (the SKLearn-backed framework
                # processors) omit it and rely on the SDK default; passing py_version=None errors.
                if py is not None:
                    fw_kwargs["py_version"] = py
                return FrameworkProcessor(**fw_kwargs)
            if kind == "script":
                # ScriptProcessor with the SAIS ECR-from-role image + KMS/network (EdxUploading).
                from sagemaker.processing import ScriptProcessor

                image_uri = (
                    f"{self.role.split(':')[4]}.dkr.ecr."
                    f"{cfg.aws_region or 'us-east-1'}.amazonaws.com/"
                    f"sais_python_lib_docker_image"
                )
                # STANDARDIZED: the old edx factory omitted base_job_name (the ONLY processor that
                # did — it let the SDK auto-name); the script kind now sets base_job_name=job_name
                # like every other processor, so edx's job naming matches the fleet.
                kwargs = dict(
                    image_uri=image_uri,
                    role=self.role,
                    instance_count=cfg.processing_instance_count,
                    instance_type=instance_type,
                    volume_size_in_gb=cfg.processing_volume_size,
                    command=["python3"],
                    sagemaker_session=self.session,
                    base_job_name=job_name,
                    env=env,
                )
                if spec.kms_network:
                    from mods_workflow_core.utils.constants import (
                        KMS_ENCRYPTION_KEY_PARAM,
                        PROCESSING_JOB_SHARED_NETWORK_CONFIG,
                    )

                    kwargs["volume_kms_key"] = KMS_ENCRYPTION_KEY_PARAM
                    kwargs["network_config"] = PROCESSING_JOB_SHARED_NETWORK_CONFIG
                return ScriptProcessor(**kwargs)

        if kind == "estimator":
            est_cls = self._resolve_sdk_class(spec.sdk_class)
            source_dir = cfg.effective_source_dir
            est_kwargs = dict(
                entry_point=cfg.training_entry_point,
                source_dir=source_dir,
                framework_version=fw,
                py_version=py,
                role=self.role,
                instance_type=cfg.training_instance_type,
                instance_count=cfg.training_instance_count,
                volume_size=cfg.training_volume_size,
                base_job_name=job_name,
                sagemaker_session=self.session,
                output_path=output_path,
                environment=env,
            )
            if spec.retrieve_image:
                from sagemaker import image_uris

                # Region locking is a TOGGLEABLE pattern (FZ 31e1d3k): when lock_training_region is
                # set (the SAIS platform restriction), pin to the locked region; otherwise use the
                # config's normal region (standard mode) — switchable via .step.yaml/config, no code.
                region = (
                    spec.locked_region
                    if spec.lock_training_region
                    else (cfg.aws_region or "us-east-1")
                )
                est_kwargs["image_uri"] = image_uris.retrieve(
                    framework="pytorch",
                    region=region,
                    version=fw,
                    py_version=py,
                    instance_type=cfg.training_instance_type,
                    image_scope="training",
                )
            return est_cls(**est_kwargs)

        if kind == "model":
            # CreateModel: a *Model (PyTorchModel/XGBoostModel) with an auto-retrieved INFERENCE
            # image. The ModelCreationHandler threads `model_data` in LAST.
            from sagemaker import image_uris

            model_cls = self._resolve_sdk_class(spec.sdk_class)
            # region locking is the SAME toggleable pattern as the estimator (the model image was
            # historically forced to us-east-1 — now an opt-in lock_training_region flag).
            region = (
                spec.locked_region
                if spec.lock_training_region
                else (cfg.aws_region or "us-east-1")
            )
            image_uri = image_uris.retrieve(
                framework=spec.framework_name,
                region=region,
                version=fw,
                py_version=py,
                instance_type=cfg.instance_type,
                image_scope="inference",
            )
            return model_cls(
                model_data=model_data,
                role=self.role,
                entry_point=cfg.entry_point,
                source_dir=cfg.effective_source_dir,
                framework_version=fw,
                py_version=py,
                image_uri=image_uri,
                sagemaker_session=self.session,
                env=env,
            )

        if kind == "transformer":
            # Batch Transform: a Transformer. The TransformHandler threads `model_name` + `output_path`
            # in LAST. No image, no role, no framework — a distinct, image-less compute shape.
            from sagemaker.transformer import Transformer

            return Transformer(
                model_name=model_name,
                instance_type=cfg.transform_instance_type,
                instance_count=cfg.transform_instance_count,
                output_path=output_path,  # SageMaker auto-assigns when None
                accept=cfg.accept,
                assemble_with=cfg.assemble_with,
                sagemaker_session=self.session,
            )

        raise ValueError(f"compute.kind {kind!r} not built by _create_compute")

    @staticmethod
    def _resolve_sdk_class(name: Optional[str]):
        """Map a compute ``sdk_class`` NAME (e.g. 'PyTorch') to the SDK class object (lazy import)."""
        if name == "PyTorch":
            from sagemaker.pytorch import PyTorch

            return PyTorch
        if name == "SKLearn":
            from sagemaker.sklearn import SKLearn

            return SKLearn
        if name == "XGBoost":
            from sagemaker.xgboost import XGBoost

            return XGBoost
        if name == "PyTorchModel":
            from sagemaker.pytorch import PyTorchModel

            return PyTorchModel
        if name == "XGBoostModel":
            from sagemaker.xgboost import XGBoostModel

            return XGBoostModel
        raise ValueError(f"unknown compute sdk_class {name!r}")

    def _is_pipeline_variable(self, value: Any) -> bool:
        """True if ``value`` is a SageMaker PipelineVariable (or quacks like one).

        Shared input-safety helper (FZ 31e1d3i) used by ``_detect_circular_references`` and the
        ProcessingHandler's ``circular_ref_check`` knob. Lazy import keeps builder_base free of a
        hard sagemaker.workflow dependency at module load.
        """
        try:
            from sagemaker.workflow.entities import PipelineVariable
        except Exception:
            PipelineVariable = ()  # type: ignore
        return isinstance(value, PipelineVariable) or (
            hasattr(value, "expr") and callable(getattr(value, "expr", None))
        )

    def _detect_circular_references(
        self, var: Any, visited: Optional[set] = None
    ) -> bool:
        """Detect circular references in PipelineVariable objects (FZ 31e1d3i).

        Re-homed from the 3 model eval/wiki builders (byte-identical there) so a step's input check
        is a base capability the ProcessingHandler can invoke via the ``circular_ref_check`` knob —
        no per-builder copy. Guards against infinite recursion / unresolvable wiring at build time.
        """
        if visited is None:
            visited = set()
        if id(var) in visited:
            return True
        if self._is_pipeline_variable(var):
            visited.add(id(var))
            for dep in getattr(var, "_dependencies", []):
                if self._detect_circular_references(dep, visited):
                    return True
        elif isinstance(var, dict):
            for key, value in var.items():
                if key == "Get":  # Skip Get references
                    continue
                if self._detect_circular_references(value, visited.copy()):
                    return True
        return False

    def set_execution_prefix(
        self, execution_prefix: Optional[Union[str, Any]] = None
    ) -> None:
        """
        Set the execution prefix for dynamic output path resolution.

        This method is called by PipelineAssembler to provide the execution prefix
        that step builders use for dynamic output path generation.

        Based on analysis of regional_xgboost.py, only PIPELINE_EXECUTION_TEMP_DIR
        is used by step builders for output paths. Other pipeline parameters
        (KMS_ENCRYPTION_KEY_PARAM, VPC_SUBNET, SECURITY_GROUP_ID) are used at
        the pipeline level, not in step builders.

        Args:
            execution_prefix: The execution prefix that can be either:
                           - ParameterString: PIPELINE_EXECUTION_TEMP_DIR from pipeline parameters
                           - str: config.pipeline_s3_loc as fallback
                           - None: No parameter found, will fall back to config.pipeline_s3_loc
        """
        self.execution_prefix = execution_prefix
        self.log_debug("Set execution prefix: %s", execution_prefix)

    def _get_base_output_path(self) -> Union[str, Any]:
        """
        Get base path for output destinations with PIPELINE_EXECUTION_TEMP_DIR support.

        This method checks for the execution_prefix (set by PipelineAssembler) and falls
        back to the traditional pipeline_s3_loc from config.

        Returns:
            The base path for output destinations. Returns a ParameterString if
            execution_prefix was set from PIPELINE_EXECUTION_TEMP_DIR, otherwise
            returns the string value from config.pipeline_s3_loc.
        """
        # Check if execution_prefix has been set by PipelineAssembler
        if hasattr(self, "execution_prefix") and self.execution_prefix is not None:
            self.log_info("Using execution_prefix for base output path")
            return self.execution_prefix

        # Fall back to pipeline_s3_loc from config (current behavior)
        base_path = self.config.pipeline_s3_loc
        self.log_debug(
            "No execution_prefix set, using config.pipeline_s3_loc for base output path"
        )
        return base_path

    def validate_configuration(self) -> None:
        """
        Validate builder-context configuration requirements (optional hook).

        No-op by default. The Pydantic config class is the authority for config validation —
        required fields, ``@field_validator`` / ``@model_validator`` constraints, and defaults are
        all enforced at config construction, BEFORE the builder runs. A config that constructs is
        valid by definition, so most builders need no override here (FZ 31e1d3e). Override ONLY to
        assert an invariant the config genuinely cannot express — one involving builder context
        (``self.role`` / ``self.session`` / ``self.spec`` / resolved dependencies) or a cross-field
        rule not yet on the config model.
        """
        return None

    def get_required_dependencies(self) -> List[str]:
        """
        Get list of required dependency logical names from specification.

        This method provides direct access to the required dependencies defined in
        the step specification.

        Returns:
            List of logical names for required dependencies

        Raises:
            ValueError: If specification is not provided
        """
        if not self.spec or not hasattr(self.spec, "dependencies"):
            raise ValueError(
                "Step specification is required for dependency information"
            )

        return [d.logical_name for _, d in self.spec.dependencies.items() if d.required]

    def get_optional_dependencies(self) -> List[str]:
        """
        Get list of optional dependency logical names from specification.

        This method provides direct access to the optional dependencies defined in
        the step specification.

        Returns:
            List of logical names for optional dependencies

        Raises:
            ValueError: If specification is not provided
        """
        if not self.spec or not hasattr(self.spec, "dependencies"):
            raise ValueError(
                "Step specification is required for dependency information"
            )

        return [
            d.logical_name for _, d in self.spec.dependencies.items() if not d.required
        ]

    def get_outputs(self) -> Dict[str, Any]:
        """
        Get output specifications directly from the step specification.

        This method provides direct access to the outputs defined in the
        step specification, returning the complete OutputSpec objects.

        Returns:
            Dictionary mapping output names to their OutputSpec objects

        Raises:
            ValueError: If specification is not provided
        """
        if not self.spec or not hasattr(self.spec, "outputs"):
            raise ValueError("Step specification is required for output information")

        return {o.logical_name: o for _, o in self.spec.outputs.items()}

    @abstractmethod
    def _get_inputs(self, inputs: Dict[str, Any]) -> Any:
        """
        Get inputs for the step.

        This is a unified method that all derived classes must implement.
        Each derived class will return the appropriate input type for its step:
        - ProcessingInput list for ProcessingStep
        - Training channels dict for TrainingStep
        - Model location for ModelStep
        etc.

        Args:
            inputs: Dictionary mapping logical names to input sources

        Returns:
            Appropriate inputs object for the step type
        """
        pass

    @abstractmethod
    def _get_outputs(self, outputs: Dict[str, Any]) -> Any:
        """
        Get outputs for the step.

        This is a unified method that all derived classes must implement.
        Each derived class will return the appropriate output type for its step:
        - ProcessingOutput list for ProcessingStep
        - Output path for TrainingStep
        - Model output info for ModelStep
        etc.

        Args:
            outputs: Dictionary mapping logical names to output destinations

        Returns:
            Appropriate outputs object for the step type
        """
        pass

    def _get_context_name(self) -> str:
        """
        Get the context name to use for registry operations.

        Returns:
            Context name based on pipeline name or default
        """
        if hasattr(self.config, "pipeline_name") and self.config.pipeline_name:
            return self.config.pipeline_name
        return "default"

    def _get_registry_manager(self) -> RegistryManager:
        """
        Get or create a registry manager.

        Returns:
            Registry manager instance
        """
        if not hasattr(self, "_registry_manager") or self._registry_manager is None:
            self._registry_manager = RegistryManager()
            self.log_debug("Created new registry manager")
        return self._registry_manager

    def _get_registry(self) -> Any:
        """
        Get the appropriate registry for this step.

        Returns:
            Registry instance for the current context
        """
        registry_manager = self._get_registry_manager()
        context_name = self._get_context_name()
        return registry_manager.get_registry(context_name)

    def _get_dependency_resolver(self) -> UnifiedDependencyResolver:
        """
        Get or create a dependency resolver.

        Returns:
            Dependency resolver instance
        """
        if (
            not hasattr(self, "_dependency_resolver")
            or self._dependency_resolver is None
        ):
            registry = self._get_registry()
            semantic_matcher = SemanticMatcher()
            self._dependency_resolver = create_dependency_resolver(
                registry, semantic_matcher
            )
            self.log_debug(
                f"Created new dependency resolver for context '{self._get_context_name()}'"
            )
        return self._dependency_resolver

    def extract_inputs_from_dependencies(
        self, dependency_steps: List[Step]
    ) -> Dict[str, Any]:
        """
        Extract inputs from dependency steps using the UnifiedDependencyResolver.

        Args:
            dependency_steps: List of dependency steps

        Returns:
            Dictionary of inputs extracted from dependency steps

        Raises:
            ValueError: If dependency resolver is not available or specification is not provided
        """
        if not DEPENDENCY_RESOLVER_AVAILABLE:
            raise ValueError(
                "UnifiedDependencyResolver not available. Make sure pipeline_deps module is installed."
            )

        if not self.spec:
            raise ValueError(
                "Step specification is required for dependency extraction."
            )

        # Get step name
        step_name = self.__class__.__name__.replace("Builder", "Step")

        # Use the injected resolver or create one
        resolver = self._get_dependency_resolver()
        resolver.register_specification(step_name, self.spec)

        # Register dependencies and enhance them with metadata
        available_steps: List[str] = []
        self._enhance_dependency_steps_with_specs(
            resolver, dependency_steps, available_steps
        )

        # One method call handles what used to require multiple matching methods
        resolved = resolver.resolve_step_dependencies(step_name, available_steps)

        # Convert results to SageMaker properties
        return {
            name: prop_ref.to_sagemaker_property()
            for name, prop_ref in resolved.items()
        }

    def _enhance_dependency_steps_with_specs(
        self, resolver: Any, dependency_steps: List[Step], available_steps: List[str]
    ) -> None:
        """
        Enhance dependency steps with specifications and additional metadata.

        This method extracts specifications from dependency steps and adds them to the resolver.
        It also extracts additional metadata to help with dependency resolution for steps
        that don't have specifications.

        Args:
            resolver: The UnifiedDependencyResolver instance
            dependency_steps: List of dependency steps
            available_steps: List to populate with step names
        """
        from .step_interface import (
            StepInterface,
            SpecSection,
            OutputDecl,
            ContractSection,
        )
        from .enums import DependencyType

        for i, dep_step in enumerate(dependency_steps):
            # Get step name
            dep_name = getattr(dep_step, "name", f"Step_{i}")
            available_steps.append(dep_name)

            # Try to get specification from step
            dep_spec = None
            if hasattr(dep_step, "_spec"):
                dep_spec = getattr(dep_step, "_spec")
            elif hasattr(dep_step, "spec"):
                dep_spec = getattr(dep_step, "spec")

            if dep_spec:
                resolver.register_specification(dep_name, dep_spec)
                logger.debug(
                    f"Registered specification for dependency step '{dep_name}'"
                )
                continue

            # If no specification, try to create a minimal one
            try:
                # For model artifacts from training steps
                if hasattr(dep_step, "properties") and hasattr(
                    dep_step.properties, "ModelArtifacts"
                ):
                    minimal_spec = StepInterface(
                        step_type=dep_name,
                        contract=ContractSection(),
                        spec=SpecSection(
                            outputs={
                                "model": OutputDecl(
                                    description="Model artifacts",
                                    type=DependencyType.MODEL_ARTIFACTS,
                                    property_path="properties.ModelArtifacts.S3ModelArtifacts",
                                )
                            },
                        ),
                    )
                    resolver.register_specification(dep_name, minimal_spec)
                    logger.info(f"Created minimal model spec for {dep_name}")

                # For processing outputs
                elif (
                    hasattr(dep_step, "properties")
                    and hasattr(dep_step.properties, "ProcessingOutputConfig")
                    and hasattr(dep_step.properties.ProcessingOutputConfig, "Outputs")
                ):
                    outputs = {}
                    processing_outputs = (
                        dep_step.properties.ProcessingOutputConfig.Outputs
                    )

                    # Handle dictionary-like outputs
                    if hasattr(processing_outputs, "items"):
                        try:
                            for key, output in processing_outputs.items():
                                if hasattr(output, "S3Output") and hasattr(
                                    output.S3Output, "S3Uri"
                                ):
                                    outputs[key] = OutputDecl(
                                        description=f"Output {key}",
                                        type=DependencyType.PROCESSING_OUTPUT,
                                        property_path=f"properties.ProcessingOutputConfig.Outputs['{key}'].S3Output.S3Uri",
                                    )
                        except (AttributeError, TypeError):
                            pass

                    # Handle list-like outputs
                    elif hasattr(processing_outputs, "__getitem__"):
                        try:
                            for i, output in enumerate(processing_outputs):
                                if hasattr(output, "S3Output") and hasattr(
                                    output.S3Output, "S3Uri"
                                ):
                                    key = f"output_{i}"
                                    outputs[key] = OutputDecl(
                                        description=f"Output at index {i}",
                                        type=DependencyType.PROCESSING_OUTPUT,
                                        property_path=f"properties.ProcessingOutputConfig.Outputs[{i}].S3Output.S3Uri",
                                    )
                        except (IndexError, TypeError, AttributeError):
                            pass

                    if outputs:
                        minimal_spec = StepInterface(
                            step_type=dep_name,
                            contract=ContractSection(),
                            spec=SpecSection(
                                outputs=outputs,
                            ),
                        )
                        resolver.register_specification(dep_name, minimal_spec)
                        logger.info(
                            f"Created minimal processing spec for {dep_name} with {len(outputs)} outputs"
                        )

            except Exception as e:
                logger.debug(
                    f"Error creating minimal specification for {dep_name}: {e}"
                )

    @abstractmethod
    def create_step(self, **kwargs: Any) -> Step:
        """
        Create pipeline step.

        This method should be implemented by all step builders to create a SageMaker pipeline step.
        It accepts a dictionary of keyword arguments that can be used to configure the step.

        Common parameters that all step builders should handle:
        - dependencies: Optional list of steps that this step depends on
        - enable_caching: Whether to enable caching for this step (default: True)

        Step-specific parameters should be extracted from kwargs as needed.

        Args:
            **kwargs: Keyword arguments for configuring the step

        Returns:
            SageMaker pipeline step
        """
        pass
