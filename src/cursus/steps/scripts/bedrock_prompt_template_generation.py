"""
Bedrock Prompt Template Generation Script — declarative meta-prompt assembler.

Assembles the prompt-config bundle into ONE standardized ``prompts.json`` prompt ruleset in the
``{ruleset, rules}`` shape that BedrockProcessing / BedrockBatchProcessing consume — the SAME
contract the knowledge-routing producer emits, so the two producers share one contract. The step's job: validate the four config blanks -> assemble -> emit one prompt ruleset.

Design (a meta-prompt template with named slots + a validated input contract):
  * ONE meta-prompt template with named slots ({SYSTEM_PROMPT} lives on ruleset.system_prompt;
    {RULES} / {INPUT_EVIDENCE} / {OUTPUT_CONSTRAINTS} on ruleset.user_prompt_template).
  * The OUTPUT SCHEMA is embedded INSIDE the prompt (the {OUTPUT_CONSTRAINTS} block the model reads)
    AND carried as ``ruleset.output_schema`` (the machine half BedrockProcessing turns into the
    forced-tool schema). No separate validation_schema artifact is needed downstream.
  * Assembly is a str.replace / str.join over the config bundle — NO tone register table, NO
    placeholder-example guessing, NO self-output prose validator (the removed ~1,200 LOC of
    ceremony; the fill-in-the-blank contract is enforced by the config Pydantic models at authoring
    time, not by re-validating the generated prose).

Container contract (bedrock_prompt_template_generation.step.yaml):
  inputs   prompt_configs=/opt/ml/processing/input/prompt_configs (OPTIONAL — bundled defaults)
  outputs  prompt_templates=/opt/ml/processing/output/templates (the ONE prompts.json ruleset),
           template_metadata=/opt/ml/processing/output/metadata,
  env      TEMPLATE_TASK_TYPE, TEMPLATE_STYLE, VALIDATION_LEVEL, INPUT_PLACEHOLDERS,
           INCLUDE_EXAMPLES, TEMPLATE_VERSION
"""

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Container path constants
CONTAINER_PATHS = {
    "INPUT_PROMPT_CONFIGS_DIR": "/opt/ml/processing/input/prompt_configs",
    "OUTPUT_TEMPLATES_DIR": "/opt/ml/processing/output/templates",
    "OUTPUT_METADATA_DIR": "/opt/ml/processing/output/metadata",
    "OUTPUT_SCHEMA_DIR": "/opt/ml/processing/output/schema",
}

# Default system prompt configuration
DEFAULT_SYSTEM_PROMPT_CONFIG = {
    "role_definition": "expert analyst",
    "expertise_areas": ["data analysis", "classification", "pattern recognition"],
    "responsibilities": [
        "analyze data accurately",
        "classify content systematically",
        "provide clear reasoning",
    ],
    "behavioral_guidelines": [
        "be precise",
        "be objective",
        "be thorough",
        "be consistent",
    ],
    "tone": "professional",
}

# Default output format configuration
DEFAULT_OUTPUT_FORMAT_CONFIG = {
    "format_type": "structured_json",
    "required_fields": ["category", "confidence", "key_evidence", "reasoning"],
    "field_descriptions": {
        "category": "The classified category name (must be exactly one of the defined categories)",
        "confidence": "Confidence score between 0.0 and 1.0 indicating certainty of classification",
        "key_evidence": "Specific evidence from input data that aligns with the selected category conditions and does NOT match any category exceptions. Reference exact content that supports the classification decision.",
        "reasoning": "Clear explanation of the decision-making process, showing how the evidence supports the selected category while considering why other categories were rejected",
    },
    "validation_requirements": [
        "category must match one of the predefined category names exactly",
        "confidence must be a number between 0.0 and 1.0",
        "key_evidence must align with category conditions and avoid category exceptions",
        "key_evidence must reference specific content from the input data",
        "reasoning must explain the logical connection between evidence and category selection",
    ],
    "evidence_validation_rules": [
        "Evidence MUST align with at least one condition for the selected category",
        "Evidence MUST NOT match any exceptions listed for the selected category",
        "Evidence should reference specific content from the input data",
        "Multiple pieces of supporting evidence strengthen the classification",
    ],
}

# Default instruction configuration
DEFAULT_INSTRUCTION_CONFIG = {
    "include_analysis_steps": True,
    "include_decision_criteria": True,
    "include_reasoning_requirements": True,
    "step_by_step_format": True,
    "include_evidence_validation": True,
}

# ==========================================================================================
# THE META-PROMPT TEMPLATE — the single standardized prompt shape.
# {RULES} / {INPUT_EVIDENCE} / {OUTPUT_CONSTRAINTS} are the named slots the assembler fills. The
# system layer is carried separately on ruleset.system_prompt (BedrockProcessing's _build_system_prompt
# reads it); this template is the USER layer. Rules are embedded STATICALLY (this producer does not
# route per-record (unlike the knowledge-routing producer), so every record sees the full rule set.
# ==========================================================================================
META_PROMPT_TEMPLATE = """## Classification Rules

Evaluate the input against each rule below. Select exactly one category.

{RULES}

## Evidence

{INPUT_EVIDENCE}

## Required Output

{OUTPUT_CONSTRAINTS}
"""


# ==========================================================================================
# ASSEMBLY — pure, deterministic functions (the "shell"); the config bundle is the "data".
# ==========================================================================================


def build_system_prompt(
    system_config: Dict[str, Any], instruction_config: Dict[str, Any]
) -> str:
    """Assemble the system layer from system_prompt.json — a deterministic join, no tone register.

    Role + expertise + responsibilities + behavioral guidelines. (The former tone-adjustment table
    that rewrote the opener per 'professional/casual/technical/formal' register is removed — a
    classification-prompt generator does not need register switching.)
    """
    role = system_config.get("role_definition", "expert analyst")
    expertise = system_config.get("expertise_areas", []) or []
    responsibilities = system_config.get("responsibilities", []) or []
    guidelines = system_config.get("behavioral_guidelines", []) or []

    parts: List[str] = []
    if expertise:
        parts.append(
            f"You are an {role} with extensive knowledge in {', '.join(expertise)}."
        )
    else:
        parts.append(f"You are an {role}.")
    if responsibilities:
        parts.append("Your task is to " + ", ".join(responsibilities) + ".")
    if guidelines:
        parts.append("Always " + ", ".join(guidelines) + " in your analysis.")
    return " ".join(parts).strip()


def render_rule_block(rule: Dict[str, Any], index: int, include_examples: bool) -> str:
    """Render one rule as a fixed <RULE i> block.

    Emits name + description + conditions + exclusions + key_elements (+ examples if enabled),
    each only when present — mirrors BedrockProcessing's own _render_routed_rules block so the
    static-embed and the per-record-routed renderings are consistent.
    """
    name = rule.get("name") or rule.get("rule_name") or f"Rule_{index}"
    lines = [f"<RULE {index}: {name}>"]
    if rule.get("description"):
        lines.append(f"Description: {rule['description']}")
    conditions = rule.get("conditions") or []
    if conditions:
        lines.append("Conditions (ALL must be met):")
        lines.extend(f"  {i}. {c}" for i, c in enumerate(conditions, 1))
    key_indicators = rule.get("key_indicators") or rule.get("key_elements") or []
    if key_indicators:
        lines.append("Key Indicators:")
        lines.extend(f"  - {k}" for k in key_indicators)
    exceptions = rule.get("exceptions") or rule.get("exclusions") or []
    if exceptions:
        lines.append("Exceptions (if ANY match, this rule does NOT apply):")
        lines.extend(f"  - {x}" for x in exceptions)
    if include_examples and rule.get("examples"):
        lines.append("Examples:")
        lines.extend(f"  - {e}" for e in rule["examples"])
    lines.append(f"</RULE {index}>")
    return "\n".join(lines)


def render_rules_section(
    categories: List[Dict[str, Any]], include_examples: bool
) -> str:
    """Render all rules into the {RULES} slot, in priority order (lower priority number first)."""
    ordered = sorted(categories, key=lambda c: c.get("priority", 1))
    return "\n\n".join(
        render_rule_block(rule, i, include_examples)
        for i, rule in enumerate(ordered, 1)
    )


def render_input_evidence(input_placeholders: List[str]) -> str:
    """Render the INPUT schema as labelled evidence slots the record fills (requirement #2, input)."""
    placeholders = input_placeholders or ["input_data"]
    return "\n\n".join(f"{p}:\n{{{p}}}" for p in placeholders)


def build_output_constraints_text(
    output_format_config: Dict[str, Any], category_names: List[str]
) -> str:
    """Render the OUTPUT SCHEMA as the prose block embedded IN the prompt (requirement #2, output).

    This is the human-readable half of the schema — the exact fields, their descriptions, the
    category enum, and the validation rules — assembled from output_format.json (fed into the
    {OUTPUT_CONSTRAINTS} slot).
    """
    required_fields = output_format_config.get(
        "required_fields", ["category", "confidence", "key_evidence", "reasoning"]
    )
    field_descriptions = output_format_config.get("field_descriptions", {}) or {}
    parts = [
        "Respond with a single JSON object containing exactly these fields:",
        "",
    ]
    for field in required_fields:
        desc = field_descriptions.get(field, f"the {field} value")
        parts.append(f"- {field}: {desc}")
    parts.append("")
    parts.append(
        "The `category` (or abuse-vector) field MUST be exactly one of the rule names above: "
        + ", ".join(category_names)
        + "."
    )
    for key, header in (
        ("validation_requirements", "Validation Requirements"),
        ("evidence_validation_rules", "Evidence Validation"),
    ):
        items = output_format_config.get(key) or []
        if items:
            parts.append("")
            parts.append(f"{header}:")
            parts.extend(f"- {item}" for item in items)
    example_output = output_format_config.get("example_output")
    if example_output:
        parts.append("")
        parts.append("Example Output:")
        if isinstance(example_output, list):
            parts.extend(str(line) for line in example_output)
        else:
            parts.append(str(example_output))
    return "\n".join(parts)


def build_output_schema(
    output_format_config: Dict[str, Any], category_names: List[str]
) -> Dict[str, Any]:
    """Build the machine output JSON schema (the forced-tool half), enum-locked to the rule names.

    Precedence mirrors the former script: an explicit ``json_schema`` in output_format.json wins;
    a config that IS itself a JSON schema (has ``type``) is used as-is; otherwise a schema is
    derived from ``required_fields`` + ``field_descriptions``. The ``category`` field's enum is
    always (re)populated with the rule names.
    """
    if output_format_config.get("json_schema"):
        schema = json.loads(
            json.dumps(output_format_config["json_schema"])
        )  # deep copy
    elif "type" in output_format_config:
        schema = json.loads(json.dumps(output_format_config))
    else:
        required_fields = output_format_config.get(
            "required_fields", ["category", "confidence", "key_evidence", "reasoning"]
        )
        field_descriptions = output_format_config.get("field_descriptions", {}) or {}
        properties: Dict[str, Any] = {}
        for field in required_fields:
            if field in ("confidence", "confidence_score"):
                properties[field] = {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": field_descriptions.get(
                        field, "Confidence score between 0.0 and 1.0"
                    ),
                }
            elif field == "category":
                properties[field] = {
                    "type": "string",
                    "enum": list(category_names),
                    "description": field_descriptions.get(
                        field, "The classified category name"
                    ),
                }
            else:
                properties[field] = {
                    "type": "string",
                    "description": field_descriptions.get(field, f"The {field} value"),
                }
        schema = {
            "type": "object",
            "properties": properties,
            "required": required_fields,
            "additionalProperties": False,
        }
    # Always (re)populate the category enum from the rule names.
    props = schema.get("properties", {})
    if "category" in props and props["category"].get("type") == "string":
        props["category"]["enum"] = list(category_names)
    return schema


def build_rules_list(categories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map category_definitions.json entries to the ``rules[]`` shape the Bedrock processing step
    reads: rule_name / description / conditions / exclusions /
    key_elements / priority_tier / abuse_flag / abuse_vector. (name->rule_name,
    exceptions->exclusions, key_indicators->key_elements, priority->priority_tier.)
    """
    ordered = sorted(categories, key=lambda c: c.get("priority", 1))
    rules: List[Dict[str, Any]] = []
    for rule in ordered:
        rules.append(
            {
                "rule_name": rule.get("name") or rule.get("rule_name"),
                "description": rule.get("description", ""),
                "key_elements": rule.get("key_indicators")
                or rule.get("key_elements")
                or [],
                "conditions": rule.get("conditions") or [],
                "exclusions": rule.get("exceptions") or rule.get("exclusions") or [],
                "priority_tier": rule.get("priority", rule.get("priority_tier")),
                "abuse_flag": rule.get("abuse_flag"),
                "abuse_vector": rule.get("abuse_vector"),
                # Carry validation_rules through for downstream renderers that want them.
                "validation_rules": rule.get("validation_rules") or [],
            }
        )
    return rules


def assemble_prompt_ruleset(
    categories: List[Dict[str, Any]],
    system_config: Dict[str, Any],
    output_format_config: Dict[str, Any],
    instruction_config: Dict[str, Any],
    input_placeholders: List[str],
    include_examples: bool,
) -> Dict[str, Any]:
    """Assemble the ONE ``prompts.json`` prompt ruleset in the ``{ruleset, rules}`` contract.

    ``ruleset.user_prompt_template`` embeds the rules + input evidence + OUTPUT SCHEMA inline via the
    META_PROMPT_TEMPLATE slots (requirement #2). ``ruleset.output_schema`` carries the machine schema.
    ``rules[]`` is the rule list. Identical downstream contract to the knowledge-routing producer.
    """
    category_names = [c.get("name") or c.get("rule_name") for c in categories]
    system_prompt = build_system_prompt(system_config, instruction_config)
    rules_section = render_rules_section(categories, include_examples)
    input_evidence = render_input_evidence(input_placeholders)
    output_constraints = build_output_constraints_text(
        output_format_config, category_names
    )

    user_prompt_template = (
        META_PROMPT_TEMPLATE.replace("{RULES}", rules_section)
        .replace("{INPUT_EVIDENCE}", input_evidence)
        .replace("{OUTPUT_CONSTRAINTS}", output_constraints)
    )

    ruleset: Dict[str, Any] = {
        "system_prompt": system_prompt,
        "input_placeholders": input_placeholders,
        "user_prompt_template": user_prompt_template,
        "output_schema": build_output_schema(output_format_config, category_names),
    }
    # Optional analysis/classification guidance — BedrockProcessing's _build_system_prompt folds
    # these into the system layer if present (kept for parity with the knowledge-routing shape).
    classification_guidelines = instruction_config.get("classification_guidelines")
    if classification_guidelines:
        ruleset["classification_guidelines"] = classification_guidelines

    return {"ruleset": ruleset, "rules": build_rules_list(categories)}


# ==========================================================================================
# CONFIG LOADING (kept verbatim — the file-based bundle contract with the config class)
# ==========================================================================================


def load_config_from_json_file(
    config_path: str,
    config_name: str,
    default_config: Dict[str, Any],
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """Load configuration from JSON file with fallback to defaults (merged over defaults)."""
    config_file = Path(config_path) / f"{config_name}.json"
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                log(f"Loaded {config_name} config from {config_file}")
                return {**default_config, **config}
        except Exception as e:
            log(
                f"Failed to load {config_name} config from {config_file}: {e}. Using defaults."
            )
            return default_config
    log(f"{config_name} config file not found at {config_file}. Using defaults.")
    return default_config


def load_category_definitions(
    prompt_configs_path: str, log: Callable[[str], None]
) -> List[Dict[str, Any]]:
    """Load category definitions (the rule list) from the prompt configs directory."""
    config_dir = Path(prompt_configs_path)
    if not config_dir.exists():
        log(f"Prompt configs directory not found: {prompt_configs_path}")
        return []
    categories_file = config_dir / "category_definitions.json"
    if categories_file.exists():
        try:
            with open(categories_file, "r", encoding="utf-8") as f:
                categories = json.load(f)
                log(f"Loaded category definitions from {categories_file}")
                return categories if isinstance(categories, list) else [categories]
        except Exception as e:
            log(f"Failed to load category definitions from {categories_file}: {e}")
            return []
    log(f"Category definitions file not found: {categories_file}")
    return []


# ==========================================================================================
# MAIN
# ==========================================================================================


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Assemble the config bundle into one prompts.json prompt ruleset.

    Args:
        input_paths: {'prompt_configs': dir with system_prompt/category_definitions/output_format/instruction .json}
        output_paths: {'prompt_templates', 'template_metadata'}
        environ_vars: TEMPLATE_*, INPUT_PLACEHOLDERS, INCLUDE_EXAMPLES, TEMPLATE_VERSION
        job_args: parsed CLI args
        logger: optional logging function (defaults to print)

    Returns:
        A small summary dict (rules_assembled / output files).
    """
    log = logger or print

    try:
        prompt_configs_path = input_paths.get("prompt_configs")
        if not prompt_configs_path:
            raise ValueError("No prompt_configs input path provided")

        # --- Load the four config blanks (the fill-in-the-blank data; validated at authoring time
        #     by the config Pydantic models). ---
        categories = load_category_definitions(prompt_configs_path, log)
        if not categories:
            raise ValueError("No category definitions found in prompt configs")
        system_config = load_config_from_json_file(
            prompt_configs_path, "system_prompt", DEFAULT_SYSTEM_PROMPT_CONFIG, log
        )
        output_format_config = load_config_from_json_file(
            prompt_configs_path, "output_format", DEFAULT_OUTPUT_FORMAT_CONFIG, log
        )
        instruction_config = load_config_from_json_file(
            prompt_configs_path, "instruction", DEFAULT_INSTRUCTION_CONFIG, log
        )

        input_placeholders = json.loads(
            environ_vars.get("INPUT_PLACEHOLDERS", '["input_data"]')
        )
        include_examples = (
            environ_vars.get("INCLUDE_EXAMPLES", "true").lower() == "true"
        )
        template_version = environ_vars.get("TEMPLATE_VERSION", "1.0")

        # --- Assemble the ONE prompt ruleset ({ruleset, rules}, schema embedded). ---
        log(f"Assembling prompt ruleset from {len(categories)} rules...")
        prompt_ruleset = assemble_prompt_ruleset(
            categories=categories,
            system_config=system_config,
            output_format_config=output_format_config,
            instruction_config=instruction_config,
            input_placeholders=input_placeholders,
            include_examples=include_examples,
        )
        category_names = [r["rule_name"] for r in prompt_ruleset["rules"]]

        # --- Write outputs. prompts.json IS the whole contract (system + rules + schema). ---
        templates_path = Path(output_paths["prompt_templates"])
        metadata_path = Path(output_paths["template_metadata"])
        for p in (templates_path, metadata_path):
            p.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        prompts_file = templates_path / "prompts.json"
        with open(prompts_file, "w", encoding="utf-8") as f:
            json.dump(prompt_ruleset, f, indent=2, ensure_ascii=False)
        log(f"Saved prompt ruleset ({len(category_names)} rules) to: {prompts_file}")

        # Lightweight metadata (no self-output prose validation — the input contract is the gate).
        metadata_file = metadata_path / f"template_metadata_{timestamp}.json"
        metadata_output = {
            "template_version": template_version,
            "task_type": environ_vars.get("TEMPLATE_TASK_TYPE", "classification"),
            "template_style": environ_vars.get("TEMPLATE_STYLE", "structured"),
            "category_count": len(category_names),
            "category_names": category_names,
            "input_placeholders": input_placeholders,
            "generation_timestamp": datetime.now().isoformat(),
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_output, f, indent=2, ensure_ascii=False, default=str)
        log(f"Saved template metadata to: {metadata_file}")

        results = {
            "success": True,
            "rules_assembled": len(category_names),
            "template_version": template_version,
            "output_files": {
                "prompts": str(prompts_file),
                "metadata": str(metadata_file),
            },
            "generation_timestamp": datetime.now().isoformat(),
        }
        log(f"Prompt ruleset assembly completed: {len(category_names)} rules")
        return results

    except Exception as e:
        log(f"Prompt ruleset assembly failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Bedrock prompt template generation script"
        )
        parser.add_argument(
            "--include-examples",
            action="store_true",
            help="Include examples in template",
        )
        parser.add_argument(
            "--template-version", default="1.0", help="Template version identifier"
        )
        args = parser.parse_args()

        input_paths = {"prompt_configs": CONTAINER_PATHS["INPUT_PROMPT_CONFIGS_DIR"]}
        output_paths = {
            "prompt_templates": CONTAINER_PATHS["OUTPUT_TEMPLATES_DIR"],
            "template_metadata": CONTAINER_PATHS["OUTPUT_METADATA_DIR"],
        }
        environ_vars = {
            "TEMPLATE_TASK_TYPE": os.environ.get(
                "TEMPLATE_TASK_TYPE", "classification"
            ),
            "TEMPLATE_STYLE": os.environ.get("TEMPLATE_STYLE", "structured"),
            "VALIDATION_LEVEL": os.environ.get("VALIDATION_LEVEL", "standard"),
            "INPUT_PLACEHOLDERS": os.environ.get(
                "INPUT_PLACEHOLDERS", '["input_data"]'
            ),
            "INCLUDE_EXAMPLES": os.environ.get(
                "INCLUDE_EXAMPLES", str(args.include_examples).lower()
            ),
            "TEMPLATE_VERSION": os.environ.get(
                "TEMPLATE_VERSION", args.template_version
            ),
        }

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _logger = logging.getLogger(__name__)
        _logger.info("Starting prompt ruleset assembly with parameters:")
        _logger.info(f"  Task Type: {environ_vars['TEMPLATE_TASK_TYPE']}")
        _logger.info(f"  Input Placeholders: {environ_vars['INPUT_PLACEHOLDERS']}")
        _logger.info(f"  Include Examples: {environ_vars['INCLUDE_EXAMPLES']}")
        _logger.info(f"  Template Version: {environ_vars['TEMPLATE_VERSION']}")

        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
            logger=_logger.info,
        )
        _logger.info(
            f"Prompt ruleset assembly completed successfully. Results: {result}"
        )
        sys.exit(0)

    except Exception as e:
        logging.error(f"Error in prompt ruleset assembly script: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)
