#!/usr/bin/env python
"""
Slipbox Knowledge Routing — Cursus ProcessingStep script (PROPOSAL scaffold).

Hosts the DKS knowledge+ruleset corpus and runs the internal
compile → index → route pipeline, emitting to the downstream BedrockProcessing step:
  - prompt_ruleset : the compiled prompt ruleset (prompts.json, in the {ruleset, rules}
                     envelope the consumer expects; output schema embedded in ruleset)
  - routed_records : the input records + selected_rule_names + routing_confidence

Pipeline stages (each ports a named DKS-router source function — see the TODOs):
  COMPILE  read knowledge_corpus/rule_*.md      -> prompts.json (in memory)
           [ports compile_prompt_ruleset.compile_rules]
  INDEX    read knowledge_corpus/pattern_*.md (+ behavior_*.md)
           -> SentenceTransformer.encode -> in-memory routing index
           [ports build_routing_index.py:150]; the encoder is overridden to the
           offline embedding_model input path so no HuggingFace-hub download occurs.
  ROUTE    read records parquet -> build_query_text -> cosine-match
           -> activation top-k -> routed rule names + routing_confidence
           [ports routing.UnifiedPatternRouter.route_batch (routing.py:217)
            + scoring.score_rules_by_activation (scoring.py:64)]

Internal consistency gate: the set of rules linked from the routing index MUST be
a subset of the compiled rule_names in prompts.json (otherwise routing could emit a
rule name the ruleset does not define).

NOTE (PROPOSAL scaffold): the ported DKS-router logic below is a faithful skeleton
with explicit TODOs pointing at the source functions. The Cursus contract surface
— the ``main(input_paths, output_paths, environ_vars, job_args)`` signature, the
I/O container paths, the env-var reads, and the ``__main__`` argparse — is complete
and correct so that validate/preflight pass and the step is constructible.
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

# ============================================================================
# COMPILE — knowledge_corpus/rule_*.md -> prompts.json (in memory)
# ============================================================================


def compile_prompt_ruleset(
    knowledge_dir: str, log: Callable[[str], None]
) -> Dict[str, Any]:
    """
    Compile the DKS ``rule_*.md`` corpus into an in-memory prompt ruleset.

    Ports ``compile_prompt_ruleset.compile_rules`` from the DKS pipeline.

    Returns a dict shaped like the emitted ``prompts.json`` — the ``{ruleset, rules}``
    envelope the downstream Bedrock consumer's ``_adapt_ruleset_templates`` expects
    (``ruleset`` = the shared prompt layer + embedded ``output_schema``; ``rules`` = a
    LIST of per-rule dicts each carrying at least ``rule_name``):
        {
          "ruleset": {"system_prompt": str, "input_placeholders": [...],
                      "output_schema": {...}},
          "rules": [{"rule_name": str, "description": str, "metadata": {...}}, ...],
          "rule_names": [rule_name, ...],   # convenience mirror of rules[*].rule_name
        }

    Args:
        knowledge_dir: Path to the mounted DKS knowledge corpus.
        log: Logging function.

    Returns:
        The compiled prompt-ruleset dict.
    """
    knowledge_path = Path(knowledge_dir)
    rule_files = sorted(knowledge_path.glob("rule_*.md"))
    log(f"[COMPILE] Found {len(rule_files)} rule_*.md files in {knowledge_dir}")

    # rules is a LIST (the consumer contract), each entry keyed by rule_name.
    rules: List[Dict[str, Any]] = []
    for rule_file in rule_files:
        rule_name = rule_file.stem  # e.g. 'rule_return_abuse_high_velocity'
        text = rule_file.read_text(encoding="utf-8")
        # TODO(compile_prompt_ruleset.compile_rules): parse the rule markdown front
        #   matter + body into structured fields (description/key_elements/conditions/
        #   exclusions/priority_tier/...); port the section-splitting from the DKS
        #   compile_prompt_ruleset.py. For now the whole rule body is the description.
        rules.append(
            {
                "rule_name": rule_name,
                "description": text,
                "metadata": {"source": rule_file.name},
            }
        )

    rule_names = sorted(r["rule_name"] for r in rules)

    # The output schema travels INSIDE the ruleset (ruleset.output_schema), so the
    # downstream consumer sources it there instead of from a separate channel. Lock the
    # classification field's enum to the compiled rule names (the routed categories).
    # TODO(compile_prompt_ruleset.build_tool_schema): enrich the per-field schema from
    #   the parsed rule metadata; this is the minimal enum-locked classification schema.
    output_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": rule_names,
                "description": "The single best-matching rule / category name.",
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning": {
                "type": "string",
                "description": "Reasoning for the classification.",
            },
        },
        "required": ["category"],
    }

    ruleset = {
        "ruleset": {
            "system_prompt": (
                "Classify each record into exactly one of the routed rules below."
            ),
            "input_placeholders": [],
            "output_schema": output_schema,
        },
        "rules": rules,
        "rule_names": rule_names,
    }
    log(f"[COMPILE] Compiled {len(rule_names)} rules into prompt ruleset")
    return ruleset


# ============================================================================
# INDEX — pattern_*/behavior_* -> SentenceTransformer.encode -> routing index
# ============================================================================


def _configure_cpu_threads(environ_vars: Dict[str, str], log: Callable) -> None:
    """Size torch's intra-op thread pool to the box's vCPUs so the batched encode uses them.

    The route stage is embedding-bound and ``SentenceTransformer.encode`` parallelizes one
    batched forward pass across cores via torch's intra-op threads. On a big CPU instance a
    container-default thread count of 1-4 leaves most cores idle, so raise it to the vCPU count.
    Env-overridable via ``ROUTING_NUM_THREADS`` (0/empty = auto = os.cpu_count()).
    """
    raw = environ_vars.get("ROUTING_NUM_THREADS", "").strip()
    try:
        n = int(raw) if raw else 0
    except ValueError:
        n = 0
    if n <= 0:
        n = os.cpu_count() or 1
    try:
        import torch

        torch.set_num_threads(n)
        log(f"[INDEX] torch intra-op threads set to {n} (vCPUs={os.cpu_count()})")
    except (
        ImportError
    ):  # pragma: no cover - container ships torch (SentenceTransformer dep)
        log("[INDEX] torch not importable; skipping thread-count configuration")


def _load_encoder(embedding_model_dir: Optional[str], model_name: str, log: Callable):
    """
    Load a SentenceTransformer encoder, overriding to the offline weights path when
    the ``embedding_model`` input is mounted (so no HuggingFace-hub download occurs).

    Ports the encoder-construction half of ``build_routing_index.py:150``.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover - container has the dep
        raise RuntimeError(
            "sentence-transformers is required for the INDEX stage. "
            "Install with: pip install sentence-transformers"
        ) from e

    if (
        embedding_model_dir
        and Path(embedding_model_dir).exists()
        and any(Path(embedding_model_dir).iterdir())
    ):
        log(f"[INDEX] Loading OFFLINE encoder from {embedding_model_dir}")
        return SentenceTransformer(embedding_model_dir)

    log(
        f"[INDEX] Offline embedding_model not mounted; loading encoder by name "
        f"'{model_name}' (may require network access)"
    )
    return SentenceTransformer(model_name)


def build_routing_index(
    knowledge_dir: str,
    embedding_model_dir: Optional[str],
    model_name: str,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """
    Read ``pattern_*.md`` (+ ``behavior_*.md``) and encode them into an in-memory
    routing index.

    Ports ``build_routing_index`` (build_routing_index.py:150).

    Returns a dict shaped like:
        {
          "pattern_names": [str, ...],
          "embeddings": np.ndarray (n_patterns, dim),
          "linked_rules": {pattern_name: [rule_name, ...], ...},
        }

    Args:
        knowledge_dir: Path to the mounted DKS knowledge corpus.
        embedding_model_dir: Optional path to offline encoder weights.
        model_name: Fallback SentenceTransformer model name.
        log: Logging function.

    Returns:
        The in-memory routing index dict.
    """
    import numpy as np

    knowledge_path = Path(knowledge_dir)
    pattern_files = sorted(knowledge_path.glob("pattern_*.md")) + sorted(
        knowledge_path.glob("behavior_*.md")
    )
    log(
        f"[INDEX] Found {len(pattern_files)} pattern_*/behavior_* files in "
        f"{knowledge_dir}"
    )

    encoder = _load_encoder(embedding_model_dir, model_name, log)

    pattern_names: List[str] = []
    pattern_texts: List[str] = []
    linked_rules: Dict[str, List[str]] = {}
    for pattern_file in pattern_files:
        pattern_name = pattern_file.stem
        text = pattern_file.read_text(encoding="utf-8")
        # TODO(build_routing_index.py:150): parse the pattern markdown into its
        #   description text + the list of rule_* names it links to (front-matter
        #   'linked_rules:' / body reference parsing in the DKS build_routing_index).
        pattern_names.append(pattern_name)
        pattern_texts.append(text)
        linked_rules[pattern_name] = []

    if pattern_texts:
        embeddings = encoder.encode(
            pattern_texts, convert_to_numpy=True, normalize_embeddings=True
        )
    else:
        embeddings = np.zeros((0, 0))

    index = {
        "pattern_names": pattern_names,
        "embeddings": embeddings,
        "linked_rules": linked_rules,
        "_encoder": encoder,  # kept in-memory for query encoding during ROUTE
    }
    log(f"[INDEX] Built routing index over {len(pattern_names)} patterns")
    return index


# ============================================================================
# ROUTE — records -> query text -> cosine-match -> activation top-k
# ============================================================================


def build_query_text(row: pd.Series) -> str:
    """
    Build the query text for a single record used to match against the pattern index.

    Ports the query-assembly half of ``routing.UnifiedPatternRouter.route_batch``
    (routing.py:217).
    """
    # TODO(routing.py:217): assemble the query text from the domain-relevant record
    #   fields exactly as the DKS UnifiedPatternRouter.build_query_text does (field
    #   selection + concatenation order matters for routing parity).
    return " ".join(str(v) for v in row.to_dict().values())


def score_rules_by_activation(
    query_embedding,
    index: Dict[str, Any],
    threshold: float,
    top_k: int,
) -> Tuple[List[str], float]:
    """
    Score rules by activation and return the top-k routed rule names + confidence.

    Ports ``scoring.score_rules_by_activation`` (scoring.py:64).

    Args:
        query_embedding: The (dim,) normalized query embedding.
        index: The in-memory routing index from ``build_routing_index``.
        threshold: Minimum cosine similarity for a pattern to activate its rules.
        top_k: Maximum number of routed rules to keep.

    Returns:
        (routed_rule_names, routing_confidence)
    """
    import numpy as np

    embeddings = index["embeddings"]
    if getattr(embeddings, "size", 0) == 0:
        return [], 0.0

    # Cosine similarity (embeddings are normalized -> dot product).
    sims = np.asarray(embeddings) @ np.asarray(query_embedding)

    # TODO(scoring.score_rules_by_activation, scoring.py:64): accumulate per-rule
    #   activation from every pattern whose similarity >= threshold (a rule's
    #   activation is the aggregate of its linked patterns' similarities), then rank
    #   rules by activation and keep the top_k. The block below is the faithful
    #   skeleton of that aggregation.
    rule_activation: Dict[str, float] = {}
    for i, pattern_name in enumerate(index["pattern_names"]):
        sim = float(sims[i])
        if sim < threshold:
            continue
        for rule_name in index["linked_rules"].get(pattern_name, []):
            rule_activation[rule_name] = max(rule_activation.get(rule_name, 0.0), sim)

    if not rule_activation:
        return [], float(sims.max()) if sims.size else 0.0

    ranked = sorted(rule_activation.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    routed_rule_names = [name for name, _ in ranked]
    routing_confidence = float(ranked[0][1]) if ranked else 0.0
    return routed_rule_names, routing_confidence


def route_records(
    records_dir: str,
    index: Dict[str, Any],
    threshold: float,
    top_k: int,
    log: Callable[[str], None],
    encode_batch_size: int = 256,
) -> pd.DataFrame:
    """
    Read the input records and route each one to a set of rule names + confidence.

    Ports ``routing.UnifiedPatternRouter.route_batch`` (routing.py:217).

    Args:
        records_dir: Path to the mounted records (parquet shards).
        index: The in-memory routing index.
        threshold: Activation threshold.
        top_k: Max routed rules per record.
        log: Logging function.

    Returns:
        The records DataFrame with two added columns:
          - selected_rule_names : list[str]  (the routed rule names; this is the column
            name the downstream Bedrock consumer reads by default —
            ``BEDROCK_ROUTED_RULES_COLUMN``, default ``selected_rule_names``)
          - routing_confidence : float
    """
    records_path = Path(records_dir)
    shards = sorted(records_path.glob("*.parquet")) + sorted(
        records_path.glob("part-*")
    )
    shards = [s for s in shards if s.is_file()]
    if not shards:
        raise RuntimeError(f"No parquet record shards found under {records_dir}")

    log(f"[ROUTE] Reading {len(shards)} record shard(s) from {records_dir}")
    df = pd.concat([pd.read_parquet(s) for s in shards], axis=0, ignore_index=True)
    log(f"[ROUTE] Loaded {df.shape[0]} records with {df.shape[1]} columns")

    encoder = index["_encoder"]
    query_texts = [build_query_text(row) for _, row in df.iterrows()]
    if query_texts:
        # One batched, multi-core encode over ALL query texts. batch_size is a throughput
        # knob (larger on a many-vCPU box improves encode throughput up to a memory ceiling).
        query_embeddings = encoder.encode(
            query_texts,
            batch_size=encode_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    else:
        query_embeddings = []

    routed_names: List[List[str]] = []
    confidences: List[float] = []
    for i in range(len(df)):
        names, conf = score_rules_by_activation(
            query_embeddings[i], index, threshold, top_k
        )
        routed_names.append(names)
        confidences.append(conf)

    df["selected_rule_names"] = routed_names
    df["routing_confidence"] = confidences
    log(f"[ROUTE] Routed {len(df)} records (top_k={top_k}, threshold={threshold})")
    return df


# ============================================================================
# CONSISTENCY GATE
# ============================================================================


def assert_index_rules_subset_of_ruleset(
    index: Dict[str, Any], ruleset: Dict[str, Any], log: Callable[[str], None]
) -> None:
    """
    Internal consistency gate: every rule the routing index can emit MUST be defined
    in the compiled prompt ruleset (index linked_rules ⊆ prompts.json rule_names).
    """
    ruleset_rule_names = set(ruleset["rule_names"])
    index_rule_names = {
        rule_name for linked in index["linked_rules"].values() for rule_name in linked
    }
    missing = index_rule_names - ruleset_rule_names
    if missing:
        raise RuntimeError(
            "Routing index references rules absent from the compiled prompt ruleset: "
            f"{sorted(missing)}"
        )
    log(
        f"[GATE] Consistency OK: {len(index_rule_names)} indexed rules ⊆ "
        f"{len(ruleset_rule_names)} compiled rules"
    )


# ============================================================================
# WRITE
# ============================================================================


def write_prompt_ruleset(
    ruleset: Dict[str, Any], output_dir: str, log: Callable[[str], None]
) -> None:
    """Write prompts.json in the ``{ruleset, rules}`` envelope the Bedrock consumer expects.

    The consumer's gate requires BOTH a top-level ``ruleset`` object AND a ``rules`` list
    (``bedrock_processing.py`` ``load_prompt_templates`` → ``_adapt_ruleset_templates``);
    the output schema travels inside ``ruleset.output_schema`` so no separate schema
    channel is needed.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prompts_path = output_path / "prompts.json"
    prompts_path.write_text(
        json.dumps(
            {"ruleset": ruleset["ruleset"], "rules": ruleset["rules"]}, indent=2
        ),
        encoding="utf-8",
    )

    log(f"[WRITE] Wrote prompt ruleset ({{ruleset, rules}}) to {prompts_path}")


def write_routed_records(
    df: pd.DataFrame, output_dir: str, log: Callable[[str], None]
) -> None:
    """Write the routed records (records + selected_rule_names + confidence) as parquet."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    out_file = output_path / "routed_records.parquet"
    # selected_rule_names is a list column — serialize to JSON strings for portability
    # across the downstream Bedrock reader (which json.loads it back).
    df = df.copy()
    df["selected_rule_names"] = df["selected_rule_names"].apply(json.dumps)
    df.to_parquet(out_file, index=False)
    log(f"[WRITE] Wrote {df.shape[0]} routed records to {out_file}")


# ============================================================================
# MAIN
# ============================================================================


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Main logic for slipbox knowledge routing, refactored for testability.

    Args:
        input_paths: Dict of input container paths keyed by logical name
            ('records', 'knowledge_corpus', 'embedding_model').
        output_paths: Dict of output container paths keyed by logical name
            ('prompt_ruleset', 'routed_records').
        environ_vars: Dict of environment variables.
        job_args: Parsed command-line arguments (carries --job_type).
        logger: Optional logging function (defaults to print).

    Returns:
        A small summary dict describing what was compiled/indexed/routed.
    """
    log = logger or print

    # --- Extract parameters from arguments and environment variables ---
    job_type = job_args.job_type
    scoring_mode = environ_vars.get("ROUTING_SCORING_MODE", "activation")
    threshold = float(environ_vars.get("ROUTING_THRESHOLD", "0.30"))
    top_k = int(environ_vars.get("ROUTING_TOP_K", "7"))
    model_name = environ_vars.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    # Throughput knobs: encode batch size for the batched route pass, and the torch intra-op
    # thread count (0/empty = auto = os.cpu_count()). Both feed the embedding-bound encode.
    encode_batch_size = int(environ_vars.get("ROUTING_ENCODE_BATCH_SIZE", "256"))

    # --- Extract paths ---
    records_dir = input_paths["records"]
    knowledge_dir = input_paths["knowledge_corpus"]
    embedding_model_dir = input_paths.get("embedding_model")

    prompt_ruleset_out = output_paths["prompt_ruleset"]
    routed_records_out = output_paths["routed_records"]

    log("[INFO] Slipbox knowledge routing settings:")
    log(f"  Job Type: {job_type}")
    log(f"  Scoring Mode: {scoring_mode}")
    log(f"  Threshold: {threshold}")
    log(f"  Top-K: {top_k}")
    log(f"  Embedding Model Name: {model_name}")
    log(f"  Encode Batch Size: {encode_batch_size}")
    log(f"  Records Dir: {records_dir}")
    log(f"  Knowledge Corpus Dir: {knowledge_dir}")
    log(f"  Embedding Model Dir: {embedding_model_dir or 'not mounted'}")

    # Size torch's intra-op thread pool to the vCPUs before any encode.
    _configure_cpu_threads(environ_vars, log)

    # --- COMPILE ---
    ruleset = compile_prompt_ruleset(knowledge_dir, log)

    # --- INDEX ---
    index = build_routing_index(knowledge_dir, embedding_model_dir, model_name, log)

    # --- CONSISTENCY GATE ---
    assert_index_rules_subset_of_ruleset(index, ruleset, log)

    # --- ROUTE ---
    routed_df = route_records(
        records_dir, index, threshold, top_k, log, encode_batch_size=encode_batch_size
    )

    # --- WRITE ---
    write_prompt_ruleset(ruleset, prompt_ruleset_out, log)
    write_routed_records(routed_df, routed_records_out, log)

    return {
        "rules_compiled": len(ruleset["rule_names"]),
        "patterns_indexed": len(index["pattern_names"]),
        "records_routed": int(routed_df.shape[0]),
    }


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--job_type",
            type=str,
            required=True,
            choices=["training", "validation", "testing", "calibration"],
            help="One of ['training','validation','testing','calibration']",
        )
        args = parser.parse_args()

        # Define standard SageMaker container paths as constants (match the contract).
        INPUT_RECORDS_DIR = "/opt/ml/processing/input/data"
        INPUT_KNOWLEDGE_DIR = "/opt/ml/processing/input/knowledge"
        INPUT_MODEL_DIR = "/opt/ml/processing/input/model"
        OUTPUT_RULESET_DIR = "/opt/ml/processing/output/ruleset"
        OUTPUT_DATA_DIR = "/opt/ml/processing/output/data"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)

        logger.info("Starting slipbox knowledge routing with parameters:")
        logger.info(f"  Job Type: {args.job_type}")
        logger.info(f"  Records Dir: {INPUT_RECORDS_DIR}")
        logger.info(f"  Knowledge Corpus Dir: {INPUT_KNOWLEDGE_DIR}")
        logger.info(f"  Embedding Model Dir: {INPUT_MODEL_DIR}")
        logger.info(f"  Prompt Ruleset Out: {OUTPUT_RULESET_DIR}")
        logger.info(f"  Routed Records Out: {OUTPUT_DATA_DIR}")

        input_paths = {
            "records": INPUT_RECORDS_DIR,
            "knowledge_corpus": INPUT_KNOWLEDGE_DIR,
            "embedding_model": INPUT_MODEL_DIR,
        }

        output_paths = {
            "prompt_ruleset": OUTPUT_RULESET_DIR,
            "routed_records": OUTPUT_DATA_DIR,
        }

        environ_vars = {
            "ROUTING_SCORING_MODE": os.environ.get(
                "ROUTING_SCORING_MODE", "activation"
            ),
            "ROUTING_THRESHOLD": os.environ.get("ROUTING_THRESHOLD", "0.30"),
            "ROUTING_TOP_K": os.environ.get("ROUTING_TOP_K", "7"),
            # Throughput knobs: encode batch size for the batched route pass, and the torch
            # intra-op thread count (empty/0 = auto = os.cpu_count()).
            "ROUTING_ENCODE_BATCH_SIZE": os.environ.get(
                "ROUTING_ENCODE_BATCH_SIZE", "256"
            ),
            "ROUTING_NUM_THREADS": os.environ.get("ROUTING_NUM_THREADS", ""),
            "EMBEDDING_MODEL_NAME": os.environ.get(
                "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
            ),
        }

        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
            logger=logger.info,
        )

        logger.info(f"Slipbox knowledge routing completed successfully: {result}")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error in slipbox knowledge routing script: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)
