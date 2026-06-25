"""
Step Interface Loader

Loads unified .step.yaml files and emits ScriptContract + StepSpecification objects.
This is the single entry point for all step interface data — replaces direct imports
from steps/specs/ and steps/contracts/.

Usage:
    from cursus.steps.interfaces import load_step_interface

    contract, spec = load_step_interface("TabularPreprocessing")
    contract, spec = load_step_interface("CradleDataLoading", job_type="calibration")
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from ...core.base.specification_base import StepSpecification, DependencySpec, OutputSpec
from ...core.base.contract_base import ScriptContract
from ...core.base.enums import DependencyType, NodeType

logger = logging.getLogger(__name__)

INTERFACES_DIR = Path(__file__).parent

# Cache loaded interfaces
_cache: Dict[str, Any] = {}


def _step_name_to_filename(step_name: str) -> str:
    """Convert StepName to filename: PyTorchTraining → pytorch_training"""
    import re
    # Handle known abbreviations that shouldn't be split
    replacements = {
        'PyTorch': 'Pytorch',
        'XGBoost': 'Xgboost',
        'LightGBMMT': 'Lightgbmmt',
        'LightGBM': 'Lightgbm',
    }
    s = step_name
    for old, new in replacements.items():
        s = s.replace(old, new)
    s = re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()
    return s


def load_step_interface(
    step_name: str,
    job_type: Optional[str] = None,
) -> Tuple[ScriptContract, StepSpecification]:
    """
    Load a step's unified interface from YAML.

    Args:
        step_name: PascalCase step name (e.g., "TabularPreprocessing", "CradleDataLoading")
        job_type: Optional job_type variant (e.g., "training", "calibration")

    Returns:
        Tuple of (ScriptContract, StepSpecification)
    """
    cache_key = f"{step_name}:{job_type or 'default'}"
    if cache_key in _cache:
        return _cache[cache_key]

    filename = _step_name_to_filename(step_name) + ".step.yaml"
    filepath = INTERFACES_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"No interface file for '{step_name}': {filepath}")

    with open(filepath) as f:
        data = yaml.safe_load(f)

    # Resolve variant if job_type specified
    spec_data = data.get("spec", {})
    contract_data = data.get("contract", {})

    if job_type and "variants" in data:
        variant = data["variants"].get(job_type)
        if variant:
            # Merge variant spec over base spec
            if "spec" in variant:
                spec_data = {**spec_data, **variant["spec"]}
            if "contract" in variant:
                contract_data = {**contract_data, **variant["contract"]}

    try:
        contract = _build_contract(contract_data, data)
    except Exception as e:
        # Training steps use /opt/ml/input, /opt/ml/model — not processing paths
        # Create a minimal contract without path validation
        contract = None
        logger.debug(f"Contract validation skipped for {step_name}: {e}")

    spec = _build_spec(spec_data, contract, data)

    _cache[cache_key] = (contract, spec)
    return contract, spec


def _build_contract(contract_data: Dict, full_data: Dict) -> ScriptContract:
    """Build ScriptContract from YAML contract section."""
    inputs = contract_data.get("inputs", {})
    outputs = contract_data.get("outputs", {})
    env = contract_data.get("env_vars", {})

    return ScriptContract(
        entry_point=contract_data.get("entry_point", ""),
        expected_input_paths={
            name: info.get("path", f"/opt/ml/processing/input/{name.lower()}")
            for name, info in inputs.items()
        },
        expected_output_paths={
            name: info.get("path", f"/opt/ml/processing/output/{name.lower()}")
            for name, info in outputs.items()
        },
        expected_arguments=contract_data.get("arguments", {}),
        required_env_vars=env.get("required", []),
        optional_env_vars=env.get("optional", {}),
        framework_requirements=contract_data.get("framework_requirements", {}),
        description=contract_data.get("description", full_data.get("step_type", "")),
    )


def _build_spec(spec_data: Dict, contract: ScriptContract, full_data: Dict) -> StepSpecification:
    """Build StepSpecification from YAML spec section."""
    # Parse node_type
    node_type_str = full_data.get("node_type", "internal")
    node_type_map = {"internal": NodeType.INTERNAL, "source": NodeType.SOURCE, "sink": NodeType.SINK}
    node_type = node_type_map.get(node_type_str, NodeType.INTERNAL)

    # Type mapping for dependencies and outputs
    dep_type_map = {
        "processing_output": DependencyType.PROCESSING_OUTPUT,
        "model_artifacts": DependencyType.MODEL_ARTIFACTS,
        "training_data": DependencyType.TRAINING_DATA,
        "hyperparameters": DependencyType.HYPERPARAMETERS,
        "payload_samples": DependencyType.PAYLOAD_SAMPLES,
        "custom_property": DependencyType.CUSTOM_PROPERTY,
    }

    # Parse dependencies
    deps = {}
    for name, info in spec_data.get("dependencies", {}).items():
        dep_type_str = info.get("type", "processing_output")
        deps[name] = DependencySpec(
            logical_name=name,
            dependency_type=dep_type_map.get(dep_type_str, DependencyType.PROCESSING_OUTPUT),
            required=info.get("required", True),
            compatible_sources=info.get("compatible_sources", []),
            semantic_keywords=info.get("semantic_keywords", []),
            data_type=info.get("data_type", "S3Uri"),
            description=info.get("description", ""),
        )

    # Parse outputs
    outputs = {}
    for name, info in spec_data.get("outputs", {}).items():
        out_type_str = info.get("type", "processing_output")
        outputs[name] = OutputSpec(
            logical_name=name,
            aliases=info.get("aliases", []),
            output_type=dep_type_map.get(out_type_str, DependencyType.PROCESSING_OUTPUT),
            property_path=info.get("property_path", f"properties.ProcessingOutputConfig.Outputs['{name}'].S3Output.S3Uri"),
            description=info.get("description", ""),
        )

    return StepSpecification(
        step_type=full_data.get("step_type", ""),
        node_type=node_type,
        script_contract=contract,
        dependencies=deps,
        outputs=outputs,
    )


def list_available_interfaces() -> List[str]:
    """List all available step interface names."""
    return [f.stem.replace(".step", "") for f in INTERFACES_DIR.glob("*.step.yaml")]
