export const meta = {
  name: 'cursus-author-step',
  description: 'Drive DAG-driven agent-tool-driven creation of a cursus 2.0 (Design B) step. Input is a user PipelineDAG in which exactly ONE node is a NEW, unregistered step sitting BETWEEN a producer and a consumer. The workflow: locate the undefined node + climb the reuse->extend->new gap ladder; ALIGN both edges (the new dependency-spec to the producer output-spec, the new output-spec to the consumer dependency-spec) with a computed >=0.5 resolution score, editing the consumer/producer specs for room; derive container-path-constrained contract paths; author the .step.yaml + config + script (fixed main signature); then validate, preflight, and (for data-carrying steps) verify execution. Gates are non-skippable pipeline stages. Implements FZ 31e1d3f5a + the FZ 29 mods-migration SOP.',
  phases: [
    { title: 'Resolve', detail: 'DAG-in: locate the undefined node between producer+consumer; decompose node->step_type+_jobtype; climb the reuse->extend->new gap ladder; bind handler + exemplar' },
    { title: 'Challenge', detail: 'verdict-override on the gap rung: a skeptic runs catalog.step_info/steps.io to try to REVERSE the rung (esp. a wrong cheaper reuse/extend/delete that would silently short-circuit); reverses on the same enum' },
    { title: 'AlignEdges', detail: 'the spec-alignment gate: arity (1 PropertyReference per DependencySpec); GATE-1 enum (dependency_type/output_type exact-or-in-matrix, else the edge hard-zeros); edit CONSUMER compatible_sources + OUTPUT aliases for editing room; GATE-2 compute the 6-component score for BOTH edges, refuse <0.5' },
    { title: 'Guide', detail: 'per-axis field guidance + author.config_constraints (allowed_values + case + required-no-default) + the container-path roots per step type' },
    { title: 'Author', detail: 'agent Writes the .step.yaml + config class + script by exemplar-PLUS-required-divergences (never a silent exemplar clone) AND edits the producer/consumer .step.yaml specs so they accept the new step' },
    { title: 'Validate', detail: 'validate.step_interface (contract<->spec + container paths) + author.check_script (both directions incl. --job_type + the {job_type} subdir layout) + py_compile/yaml parse oracle + divergence-fidelity gate (every required delta-over-exemplar is present) + re-resolve both edges, bounded fix loops' },
    { title: 'Preflight', detail: 'author.preflight_config (values) + author.preflight_step (offline constructibility = CI merge gate) + compile.validate/preview on the whole DAG' },
    { title: 'Gaps', detail: 'completeness critic over the FULL dag_path: does an additive compatible_sources edit regress another consumer? cross-branch (train vs eval) column-survival trace (habit 4)? -> green_with_open_gaps not silent-ship' },
    { title: 'Synthesize', detail: 'report what was authored + which producer/consumer specs were edited + the edge scores + the publish path' },
  ],
}

// ---------------------------------------------------------------------------
// WORK LIST. Each request is a DAG EDGE context: the NEW node plus its producer + consumer.
//   { intent, name?, producer_node, consumer_node, dag_path? }
// Pass an array to author several new nodes; pass one object for one; with no args a single
// illustrative request is used. producer_node / consumer_node are the ADJACENT DAG node strings
// (e.g. "TabularPreprocessing_training" -> [NEW] -> "XGBoostTraining"); dag_path is the file that
// defines create_<project>_dag() so the agent can read the real edges. A source node (no producer)
// or sink node (no consumer) is expressed by omitting that side.
// ---------------------------------------------------------------------------
const REQUESTS = (args && Array.isArray(args) && args.length)
  ? args
  : (args && args.intent ? [args] : [{
      intent: 'a post-training probability calibration step',
      name: 'BetaCalibration',
      producer_node: 'XGBoostModelEval_calibration',
      consumer_node: 'ModelMetricsComputation',
    }])

// The six behavior axes a .step.yaml declares. The Guide phase gathers guidance per axis the step uses.
const AXES = ['env_vars', 'job_arguments', 'inputs', 'outputs', 'compute', 'dependency']

// HARNESS CAPABILITY: does the host TOOL-FORCE structured output? The Claude Code `Workflow` host does
// (it forces a StructuredOutput tool call, so a large all-at-once schema returns as the exact object),
// so CC keeps Resolve as ONE tool-forced PLAN_SCHEMA turn. The Kiro runtime canNOT tool-force — it only
// appends the schema and re-prompts — and on the frozen kiro-cli 2.5.0 snapshot even Opus 4.8 answered
// the big 15-field schema with a per-node ARRAY the harness could not coerce. So ONLY the non-tool-
// forcing host (Kiro, which sets __workflowHost.toolForcesSchemaOutput=false on the sandbox) splits
// Resolve into three small single-object turns. Under Claude Code __workflowHost is undefined -> we
// treat it as tool-forcing -> the CC path is byte-for-byte the original single-turn Resolve (unchanged).
const HOST_TOOL_FORCES_SCHEMA =
  (typeof __workflowHost === 'undefined') || !!__workflowHost.toolForcesSchemaOutput

// CLI-REROUTE FOR NON-TOOL-FORCING HOSTS (kiro-cli, esp. the frozen SAIS 2.5.0 that loads NO MCP
// servers — SAIS Run 10). The relay-tool-result phases tell the model to CALL author.*/validate.*/
// steps.io/catalog.* MCP tools; on 2.5.0 those tools do not exist, so the model fabricates JSON (the
// AlignEdges/validate/resolve/preflight failures in Runs 5-11). But the built-in SHELL/execute tool
// WORKS on 2.5.0 (Author writes files, the parse phase runs py_compile), and `cursus` is installed.
// So on the non-tool-forcing host we reroute each phase to the equivalent `cursus` CLI command run via
// the shell tool. cliJson(cmd) returns the MANDATORY clean-JSON recipe: the sagemaker.config INFO lines
// print to STDOUT (they survive 2>/dev/null), so a `sed -n '/^{/,$p'` filter to the first `{`-at-col-0
// is REQUIRED — a bare 2>/dev/null feeds noise+JSON to the parser and fails. `--format json` is also
// mandatory (Run 11 showed the model reaching for the CLI but forgetting it and capturing text/usage).
function cliJson(cmd) {
  return "PYTHONNOUSERSITE=1 python3 -m cursus.cli " + cmd + " --format json 2>/dev/null | sed -n '/^{/,$p'"
}
// Per-phase relay instruction, branched by host. On a tool-forcing host (Claude Code) the ORIGINAL
// 'call the MCP tool' text is emitted verbatim (byte-for-byte unchanged). On the non-tool-forcing host
// the shell-CLI instruction replaces it. `mcpText` = the original instruction; `cliText` = the reroute.
function relay(mcpText, cliText) {
  return HOST_TOOL_FORCES_SCHEMA ? mcpText : cliText
}

const HOWTO = [
  'cursus 2.0 (Design B): a step = ONE .step.yaml interface + ONE <StepName>Config class + ONE script.',
  'NO builder file (synthesized via a PatternHandler), NO separate contract/spec files (sections of the',
  'YAML), NO manual registry edit (STEP_NAMES is derived — writing the .step.yaml IS the registration).',
  'Tools (call as MCP tools; if the cursus MCP server is unreachable use the `cursus` CLI equivalent,',
  'NEVER hand-edit the registry): author.checklist / author.rules / author.config_constraints /',
  'author.preflight_config / author.preflight_step / author.check_script, strategies.for_step_type /',
  'strategies.knobs, steps.io / steps.patterns, catalog.step_info / catalog.resolve_step /',
  'catalog.config_fields (CLI: `cursus catalog fields`), validate.deps_resolve / validate.deps_explain,',
  'compile.validate / compile.preview, dag.validate_integrity, validate.step_interface.',
  '',
  'THE DAG IS THE INPUT. A new step is NEVER authored alone: it is the undefined node BETWEEN a producer',
  'and a consumer in a user DAG. Its interface is CONSTRAINED by both edges:',
  '  - its dependency-spec must ALIGN with the producer output-spec (the producer->NEW edge resolves);',
  '  - its output-spec must ALIGN with the consumer dependency-spec (the NEW->consumer edge resolves);',
  '  - the producer/consumer specs must be EDITABLE so they accept the new step (add the new step type to',
  '    the consumer dependency compatible_sources; add an alias) — additive, backward-compatible only.',
  'An edge RESOLVES when the 6-component compatibility score is >=0.5. The 6 components (from',
  'dependency_resolver._calculate_compatibility): dependency_type-vs-output_type 40% (exact) / 20%',
  '(compatible-in-matrix) / 0% (else); data_type 20% (exact) / 10% (compatible); semantic-name-with-aliases',
  '25%; exact-logical-name-or-alias bonus 5%; source-compatibility 10% (consumer compatible_sources lists',
  'the job-type-NORMALIZED producer base step_type) / 5% (compatible_sources empty) / 0% (wrong entry);',
  'keyword 5%. A mismatched dependency_type/output_type HARD-ZEROS the 40% and usually the edge — fix enums',
  'BEFORE names. compatible_sources lives on the CONSUMER dependency; you EDIT THE CONSUMER, never mutate',
  'the producer output_type to force an edge.',
  '',
  'THE 4-STEP HABIT (this caught 19 of 21 real migration bugs — apply it as you author each section):',
  '(1) READ THE VALIDATOR, NOT THE DOCSTRING: when you author a config @field_validator enum/Literal, make the',
  '    allowed set + case EXPLICIT and copy any value you reuse from an exemplar VERBATIM from its validator code —',
  '    docstrings / design notes / example JSON lie about case-insensitivity (output_format is case-sensitive',
  '    CSV/TSV/Parquet; cluster_type is STANDARD/SMALL/MEDIUM/LARGE, not XLARGE). Match the declared Python TYPE',
  '    (a Dict field takes a dict, NOT a JSON string).',
  '(2) TRACE config-field -> builder -> env var -> script-read -> downstream-column: the contract.job_arguments the',
  '    builder passes MUST be argparse-d in the script; every required contract.env_var MUST be read in main(); and an',
  '    output VARIABLE name your step emits is the byte-for-byte column/field a downstream step reads.',
  '(3) PLACEHOLDER-IS-A-BUG: do not leave a silent TODO/sample value that bypasses a required field. A',
  '    reuse_class=model_dependent MODEL section must be a NUMBERED open-section checklist, not a stub that quietly',
  '    no-ops; a builder-required field (Field(...) with no default) stays required even if "the simple script does',
  '    not use it yet" (use a greppable placeholder sentinel, never omit it).',
  '(4) A RESOLVED EDGE IS NOT A RUNTIME PROOF: score>=0.5 / validate.step_interface green proves the WIRE connects,',
  '    NOT that container subdirs, filenames, or COLUMN names agree. Always run the separate {job_type}-subdir check',
  '    (producer emits <job_type>/... vs consumer scans that dir or rglobs) and a producer-output-column vs',
  '    consumer-read-column trace on BOTH DAG branches (train AND calibration/eval) — renames do not propagate across',
  '    parallel branches. Preflight-green is necessary but still a hypothesis until the step runs.',
].join('\n')

// Container-path roots per SageMaker step type (SOP step 7). main() indexes contract LOGICAL names
// only; __main__ is the ONLY place these literals appear and is copied verbatim from the exemplar.
const CONTAINER_PATHS = [
  'Processing: inputs under /opt/ml/processing/input, outputs under /opt/ml/processing/output.',
  'Training: inputs under /opt/ml/input/data (with train/val/test subdirs when job_type splits),',
  '  the model under /opt/ml/model (auto-packaged into model.tar.gz), eval under /opt/ml/output/data.',
  'Contract dict KEYS == spec logical_names byte-for-byte; VALUES are paths under these roots.',
].join(' ')

// --- structured outputs the script branches on ---
const PLAN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['step_name', 'snake_name', 'sagemaker_step_type', 'step_assembly', 'node_type', 'framework', 'reuse_class', 'bound_handler', 'exemplar_step', 'needed_axes', 'gap_rung', 'is_new_step', 'divergences_from_exemplar', 'producer', 'consumer'],
  properties: {
    step_name: { type: 'string', description: 'PascalCase; IS the .step.yaml step_type + the canonical registry name + the base of the DAG node string' },
    snake_name: { type: 'string', description: 'snake_case file stem for the 3 artifacts' },
    sagemaker_step_type: { type: 'string' },
    step_assembly: { type: 'string', description: 'code | step_args | delegation | "" (non-Processing)' },
    node_type: { type: 'string', enum: ['source', 'internal', 'sink', 'singular'] },
    framework: { type: 'string' },
    reuse_class: { type: 'string', enum: ['shared', 'model_dependent', 'user_template'] },
    bound_handler: { type: 'string', description: 'from author.checklist / strategies.for_step_type' },
    exemplar_step: { type: 'string', description: 'same-phase existing step to copy section shapes from' },
    needed_axes: { type: 'array', items: { type: 'string', enum: AXES } },
    // The gap-triage ladder result (SOP step 2). is_new_step=false means a higher rung fit and NO new
    // step type is authored (the workflow reports the cheaper resolution instead).
    gap_rung: { type: 'string', enum: ['reuse_config_only', 'extend_optional_dep', 'delete_node_artifact_exists', 'new_step'], description: 'the lowest rung that fits; higher rungs rejected with reason' },
    gap_rung_reason: { type: 'string' },
    is_new_step: { type: 'boolean', description: 'true only when gap_rung == new_step' },
    // The concrete features that make this a NEW step rather than a reuse of the exemplar — the SAME
    // facts that justified rejecting reuse_config_only. These are REQUIRED additions/changes over the
    // exemplar shape; the Author phase must NOT drop them when it copies the exemplar. Example that
    // motivated this field: TSATabularPreprocessing needed a second output `preprocessor` the generic
    // TabularPreprocessing exemplar lacks — the reason it is a new step — and Author dropped it because
    // it copied the one-output exemplar. Each entry names the axis + the delta (e.g.
    // "outputs: ADD a `preprocessor` output (fitted scaler .pkl) — the exemplar has only processed_data").
    divergences_from_exemplar: { type: 'array', items: { type: 'string' }, description: 'the REQUIRED deltas over the exemplar that define this step; empty ONLY if the step is byte-shape-identical to the exemplar (then it should not have been a new_step)' },
    // The adjacent nodes and their decomposition (SOP step 1). Empty producer => source; empty consumer => sink.
    producer: { type: 'object', additionalProperties: false, required: ['node', 'base_step_type'], properties: { node: { type: 'string' }, base_step_type: { type: 'string' }, job_type: { type: 'string' } } },
    consumer: { type: 'object', additionalProperties: false, required: ['node', 'base_step_type'], properties: { node: { type: 'string' }, base_step_type: { type: 'string' }, job_type: { type: 'string' } } },
  },
}

// RESOLVE IS SPLIT into three small, single-purpose schemas (locate -> triage -> identity) whose
// results merge into one `plan` conforming to PLAN_SCHEMA above. WHY: asking for the full 15-field
// PLAN_SCHEMA in ONE turn makes a model decompose it — on the frozen kiro-cli 2.5.0 snapshot even a
// strong model (Opus 4.8) returned a MULTI-element array ([{...},{...}], one plan per DAG node) that
// the Kiro runtime cannot coerce to the single object wanted, and it stayed stuck across re-prompts
// (empirically confirmed on SAIS, plus a per-turn timeout on the oversized turn). Small, unambiguous
// per-decision schemas each naturally yield ONE object. Under the Claude Code host this is 3
// tool-forced turns instead of 1 — a modest, harness-portable cost that also decides each sub-question
// on its own rather than cramming one schema. The sub-schemas reuse PLAN_SCHEMA.properties so the
// merged object stays byte-shape-identical to what downstream stages already read.
const LOCATE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['node_type', 'producer', 'consumer'],
  properties: {
    node_type: PLAN_SCHEMA.properties.node_type,
    producer: PLAN_SCHEMA.properties.producer,
    consumer: PLAN_SCHEMA.properties.consumer,
  },
}
const TRIAGE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['gap_rung', 'is_new_step'],
  properties: {
    gap_rung: PLAN_SCHEMA.properties.gap_rung,
    gap_rung_reason: PLAN_SCHEMA.properties.gap_rung_reason,
    is_new_step: PLAN_SCHEMA.properties.is_new_step,
  },
}
const IDENTITY_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['step_name', 'snake_name', 'sagemaker_step_type', 'step_assembly', 'framework', 'reuse_class', 'bound_handler', 'exemplar_step', 'needed_axes', 'divergences_from_exemplar'],
  properties: {
    step_name: PLAN_SCHEMA.properties.step_name,
    snake_name: PLAN_SCHEMA.properties.snake_name,
    sagemaker_step_type: PLAN_SCHEMA.properties.sagemaker_step_type,
    step_assembly: PLAN_SCHEMA.properties.step_assembly,
    framework: PLAN_SCHEMA.properties.framework,
    reuse_class: PLAN_SCHEMA.properties.reuse_class,
    bound_handler: PLAN_SCHEMA.properties.bound_handler,
    exemplar_step: PLAN_SCHEMA.properties.exemplar_step,
    needed_axes: PLAN_SCHEMA.properties.needed_axes,
    divergences_from_exemplar: PLAN_SCHEMA.properties.divergences_from_exemplar,
  },
}

// The edge-alignment result (AlignEdges phase). One entry per edge (producer->NEW, NEW->consumer).
const ALIGN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['arity_ok', 'edges', 'consumer_edits', 'ready'],
  properties: {
    arity_ok: { type: 'boolean', description: 'the consumer has exactly one DependencySpec per intended data producer (no silent ordering-only demotion)' },
    arity_note: { type: 'string' },
    edges: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['edge', 'dependency_type', 'output_type', 'type_ok', 'data_type_ok', 'projected_score', 'resolves'],
      properties: {
        edge: { type: 'string', description: 'producer->NEW or NEW->consumer' },
        dependency_type: { type: 'string' }, output_type: { type: 'string' },
        type_ok: { type: 'boolean', description: 'dependency_type/output_type exact-or-compatible (GATE-1)' },
        data_type_ok: { type: 'boolean' },
        projected_score: { type: 'number', description: 'the 6-component score you expect for this edge' },
        resolves: { type: 'boolean', description: 'projected_score >= 0.5' },
        fragile: { type: 'boolean', description: 'projected_score < 0.7' },
      } } },
    // What must be edited on the producer/consumer .step.yaml to give the new step room (additive only).
    consumer_edits: { type: 'array', items: { type: 'string' }, description: 'e.g. "add BetaCalibration to XGBoostModelEval dependency X compatible_sources"; "add alias eval_output on the NEW output"' },
    ready: { type: 'boolean', description: 'both edges resolve (>=0.5) after the planned edits, or the missing side is a source/sink' },
  },
}

// ALIGNEDGES IS DECOMPOSED for the non-tool-forcing host: instead of one turn returning ALIGN_SCHEMA
// (whose `edges` is an array-of-objects that triggers the multi-element-array bias on kiro-cli 2.5.0 —
// the AlignEdges failure in SAIS Runs 5-6), score ONE edge per turn with this single-object schema,
// plus a tiny arity turn, then assemble the ALIGN_SCHEMA-shaped object in plain code. Each turn yields
// ONE object, so no per-edge array. Merged shape is byte-identical to what Guide/Author/Synthesize read.
// NOTE: `edge` is NOT required — the runtime dictates WHICH edge each turn scores (the prompt says
// "ALIGN EXACTLY ONE EDGE: <edge>"), so the model needn't echo it and the assembly injects it after
// parse. Dropping it from `required` eliminated the dominant "$.edge missing" schema failure (SAIS
// Run 13, where the model returned valid JSON but omitted the one field the runtime already knows).
const EDGE_ALIGN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['dependency_type', 'output_type', 'type_ok', 'data_type_ok', 'projected_score', 'resolves', 'consumer_edits'],
  properties: {
    edge: { type: 'string', description: 'producer->NEW or NEW->consumer (optional — the runtime sets it)' },
    dependency_type: { type: 'string' }, output_type: { type: 'string' },
    type_ok: { type: 'boolean', description: 'dependency_type/output_type exact-or-compatible (GATE-1)' },
    data_type_ok: { type: 'boolean' },
    projected_score: { type: 'number', description: 'the 6-component score you expect for THIS one edge' },
    resolves: { type: 'boolean', description: 'projected_score >= 0.5 (or true for a missing source/sink side)' },
    fragile: { type: 'boolean', description: 'projected_score < 0.7' },
    consumer_edits: { type: 'array', items: { type: 'string' }, description: 'the additive spec edits THIS edge needs (compatible_sources / alias); [] if none' },
  },
}
const ARITY_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['arity_ok'],
  properties: {
    arity_ok: { type: 'boolean', description: 'the consumer has exactly one DependencySpec per intended data producer (no silent ordering-only demotion)' },
    arity_note: { type: 'string' },
  },
}

// Verdict-Override Challenge on the gap-ladder rung (before the is_new_step branch). A single agent
// picks the rung in Resolve with no second opinion; a wrong CHEAPER rung (reuse/extend/delete)
// silently short-circuits with the need unmet. This lets a skeptic REVERSE the rung on the same enum.
const CHALLENGE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds', 'final_rung'],
  properties: {
    holds: { type: 'boolean', description: 'true if the original rung survives scrutiny' },
    challenges: { type: 'array', items: { type: 'string' } },
    final_rung: { type: 'string', enum: ['reuse_config_only', 'extend_optional_dep', 'delete_node_artifact_exists', 'new_step'], description: 'the same enum as gap_rung — so the challenge authoritatively reverses, not merely annotates' },
    correction: { type: 'string' },
  },
}

// Executable parse oracle for the authored artifacts (SOP: a green gate must mean the files at least
// PARSE, not merely that a validator tool returned ok). The agent MUST run the commands and set the
// booleans from their exit codes — the hardest thing to fake plausibly is a syntax/parse failure.
const PARSE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['py_compile_ok', 'yaml_loads_ok'],
  properties: {
    py_compile_ok: { type: 'boolean', description: 'from `python3 -m py_compile <script.py> <config.py>` exit code' },
    yaml_loads_ok: { type: 'boolean', description: 'from `python3 -c "import yaml; yaml.safe_load(open(<yaml>))"` exit code' },
    detail: { type: 'string' },
  },
}

// Fix agents report which file they touched + whether it still compiles, so a fixer that injects a
// syntax error escalates instead of silently burning the next retry on a broken file.
const FIX_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['file_edited', 'py_compile_ok'],
  properties: {
    file_edited: { type: 'string' },
    change_summary: { type: 'string' },
    py_compile_ok: { type: 'boolean', description: 'run `python3 -m py_compile <file_edited>` after editing a .py; set from its exit code (true for a .step.yaml-only edit)' },
  },
}

// Completeness critic over the FULL DAG (not just the 3-node triple). An additive compatible_sources
// edit can regress ANOTHER consumer's pre-existing edge; a parallel branch can be mis-wired; the
// habit-4 cross-branch column trace has no executable backing. This hunts those absences.
const GAPS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['full_dag_scanned', 'edit_collateral', 'cross_branch_column_mismatches'],
  properties: {
    full_dag_scanned: { type: 'boolean', description: 'true only if the FULL dag_path node set was resolved, not the 3-node triple' },
    edit_collateral: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['step', 'note'], properties: { step: { type: 'string' }, before_score: { type: 'number' }, after_score: { type: 'number' }, note: { type: 'string' } } }, description: 'existing consumers whose edge score dropped because of the additive compatible_sources edit' },
    uncovered_consumers: { type: 'array', items: { type: 'string' } },
    cross_branch_column_mismatches: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['branch', 'detail'], properties: { branch: { type: 'string' }, detail: { type: 'string' }, severity: { type: 'string' } } }, description: 'habit-4: producer output column vs consumer read column, on BOTH train + eval/calibration branches' },
    sequencing_risks: { type: 'array', items: { type: 'string' } },
  },
}

// `axis` is NOT required — the runtime dictates WHICH axis each Guide turn covers (the loop passes it)
// and injects it after parse, so the model needn't echo it (same lever as EDGE_ALIGN_SCHEMA.edge; the
// dropped-`axis` failure was 0/6 in SAIS Run 13/14). restrictions[] backfills to [] if omitted.
const GUIDE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['recommended', 'exemplar_snippet'],
  properties: {
    axis: { type: 'string', description: 'the behavior axis (optional — the runtime sets it)' },
    recommended: { type: 'string', description: 'the concrete field values to use for this axis' },
    restrictions: { type: 'array', items: { type: 'string' }, description: 'legal-value / closed-enum restrictions that apply' },
    exemplar_snippet: { type: 'string', description: 'the matching section shape copied from the exemplar' },
  },
}

const VALIDATE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['ok', 'errors', 'warnings'],
  properties: { ok: { type: 'boolean' }, errors: { type: 'array', items: { type: 'string' } }, warnings: { type: 'array', items: { type: 'string' } } },
}

// Divergence-fidelity gate: each REQUIRED delta the Resolve phase said makes this a new step must be
// present in the authored artifact. Catches the copy-the-exemplar-and-drop-the-distinguishing-feature
// bug (e.g. authoring TSATabularPreprocessing but omitting the `preprocessor` output that was its
// whole reason to exist). present:false on any entry = the step is (partly) just the exemplar clone.
const DIVERGENCE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['all_present', 'checks'],
  properties: {
    all_present: { type: 'boolean' },
    checks: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['divergence', 'present'], properties: { divergence: { type: 'string' }, present: { type: 'boolean' }, evidence: { type: 'string', description: 'where in the authored artifact the delta landed (or why it is missing)' } } } },
  },
}

// Cross-edge resolution after the specs are written (Validate phase): does the real resolver score
// both edges >= 0.5 now that producer/NEW/consumer specs all exist?
const RESOLVE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['both_edges_resolve', 'edges'],
  properties: {
    both_edges_resolve: { type: 'boolean' },
    edges: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['edge', 'score', 'resolves'], properties: { edge: { type: 'string' }, score: { type: 'number' }, resolves: { type: 'boolean' }, note: { type: 'string' } } } },
  },
}

// Per-edge re-resolution result (the single-object form of one RESOLVE_SCHEMA.edges entry). Used on the
// non-tool-forcing host to score ONE edge per turn, avoiding the edges[] array-of-objects bias.
// `edge` not required — the runtime dictates it per turn and injects it after parse (see EDGE_ALIGN_SCHEMA note).
const EDGE_RESOLVE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['score', 'resolves'],
  properties: { edge: { type: 'string' }, score: { type: 'number' }, resolves: { type: 'boolean' }, note: { type: 'string' } },
}

const PREFLIGHT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['constructible', 'gates'],
  properties: {
    constructible: { type: 'boolean' },
    gates: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['name', 'passed', 'detail'], properties: { name: { type: 'string' }, passed: { type: 'boolean' }, detail: { type: 'string' } } } },
  },
}

// author.check_script returns either status:'skipped' (script-less/SDK) or status:'checked' with
// passed + the flat issues[]. Schema tolerates both.
const CHECK_SCRIPT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'passed'],
  properties: {
    status: { type: 'string' },
    passed: { type: 'boolean' },
    issues: { type: 'array', items: { type: 'object', additionalProperties: true, required: ['severity', 'category', 'message'], properties: { severity: { type: 'string' }, category: { type: 'string' }, message: { type: 'string' }, recommendation: { type: 'string' } } } },
  },
}

function edgeStr(req) {
  const p = req.producer_node ? req.producer_node : '(source)'
  const c = req.consumer_node ? req.consumer_node : '(sink)'
  return p + ' -> [' + (req.name || 'NEW') + '] -> ' + c
}

// COMBINED single-turn Resolve — used ONLY on a tool-forcing host (Claude Code). The `Workflow` tool
// forces the exact PLAN_SCHEMA object, so all of SOP steps 1-2b fit in one turn. This is the ORIGINAL
// Resolve prompt, unchanged, so the Claude Code path is identical to before the Kiro split was added.
function resolvePrompt(req) {
  return [
    HOWTO, '',
    'TASK (SOP steps 1-2): resolve the routing decision for the ONE undefined DAG node (do NOT write files yet).',
    'USER INTENT: ' + req.intent + (req.name ? '   PROPOSED NAME: ' + req.name : ''),
    'DAG EDGE: ' + edgeStr(req) + (req.dag_path ? '   (DAG defined in ' + req.dag_path + ' — read it for the real edges)' : ''),
    '',
    'STEP 1 — locate + decompose. For the producer node "' + (req.producer_node || '(none: this is a SOURCE node)') + '"',
    'and the consumer node "' + (req.consumer_node || '(none: this is a SINK node)') + '", decompose each DAG node string',
    'into <registered base step_type> + optional <_jobtype suffix>: the BASE substring is what the resolver matches',
    'against compatible_sources; the FULL node string becomes the config key (1 node = 1 config key). Confirm each',
    'ADJACENT base step_type is registered via catalog.step_info / catalog.resolve_step (they must already exist — only',
    'the middle node is new). A _jobtype variant of an existing step is NOT a new step_type.',
    '',
    'STEP 2 — GAP TRIAGE LADDER. Climb in order, STOP at the first rung that fits, and set gap_rung + gap_rung_reason:',
    '  (a) reuse_config_only: an existing step_type already does this with only different config values? (no new step)',
    '  (b) extend_optional_dep: adding an OPTIONAL DependencySpec + one contract line to an EXISTING step (required=False,',
    '      backward-compatible, no builder change) covers it? (no new step)',
    '  (c) delete_node_artifact_exists: the artifact is already produced upstream/externally, so the node can be DELETED?',
    '  (d) new_step: only if (a)-(c) all fail — author a genuinely NEW step (full .step.yaml + config + script).',
    'Also DEMOTE any candidate whose output is invariant across runs (deterministic from static inputs) to a config field.',
    'Set is_new_step = (gap_rung == new_step). If not a new step, still return the plan with the cheaper resolution named.',
    '',
    'STEP 2b — CAPTURE THE DIVERGENCES (do this WHENEVER new_step). The very reason reuse_config_only was',
    'rejected IS a concrete list of features the exemplar lacks — record them in divergences_from_exemplar as',
    'REQUIRED deltas the Author phase must add on top of the exemplar shape (it must NOT silently copy the',
    'exemplar and drop them). One entry per delta, naming the AXIS + the change, e.g. "outputs: ADD a',
    '`preprocessor` output (the fitted scaler .pkl) — the exemplar emits only processed_data" or "env_vars: ADD',
    'LABEL_FIELD_2". If new_step but you cannot name a single divergence, the reuse-ladder was mis-applied — go back',
    'to rung (a). (For a non-new-step rung, divergences_from_exemplar = [].)',
    '',
    'If new_step: call author.checklist(sagemaker_step_type[, step_assembly]) for the ordered SOP + the bound handler +',
    'the exemplar; strategies.for_step_type to confirm the handler/knobs; author.rules("naming") + author.rules("reuse_class").',
    'needed_axes = only the behavior axes this step actually uses. Return the PLAN (incl. divergences_from_exemplar).',
  ].join('\n')
}

// RESOLVE-SPLIT: THREE SEQUENTIAL TURNS (locate -> triage -> identity), each with a small single-object
// schema, rather than one turn asking for the whole 15-field PLAN_SCHEMA. Used ONLY on a non-tool-
// forcing host (the Kiro runtime), where a big all-at-once schema made even Opus 4.8 emit a per-node
// array. Each prompt below states its ONE decision and its exact return shape, so the model has no
// reason to emit a per-node array. The three results are merged into one PLAN_SCHEMA-shaped `plan`.

// Turn 1 — locate + decompose the two ADJACENT nodes (no gap decision, no identity yet).
function locatePrompt(req) {
  return [
    HOWTO, '',
    'TASK (SOP step 1) — LOCATE ONLY. Decompose the two ADJACENT DAG nodes around the ONE undefined middle node.',
    'Do NOT decide the gap rung, do NOT name the new step, do NOT write files. Return ONLY the locate result.',
    'USER INTENT: ' + req.intent + (req.name ? '   PROPOSED NAME: ' + req.name : ''),
    'DAG EDGE: ' + edgeStr(req) + (req.dag_path ? '   (DAG defined in ' + req.dag_path + ' — read it for the real edges)' : ''),
    '',
    'For the producer node "' + (req.producer_node || '(none: this is a SOURCE node)') + '" and the consumer node "' +
      (req.consumer_node || '(none: this is a SINK node)') + '", decompose each DAG node string into <registered base',
    'step_type> + optional <_jobtype suffix>: the BASE substring is what the resolver matches against compatible_sources;',
    'the FULL node string becomes the config key (1 node = 1 config key). Confirm each ADJACENT base step_type is',
    'registered via catalog.step_info / catalog.resolve_step (they must already exist — only the middle node is new). A',
    '_jobtype variant of an existing step is NOT a new step_type. Decide node_type of the MIDDLE new node from its edges:',
    'source (no producer) | sink (no consumer) | internal (both) | singular (neither).',
    '',
    'Return ONE JSON object {node_type, producer:{node,base_step_type,job_type?}, consumer:{node,base_step_type,job_type?}}.',
    'For a missing side (source/sink) set that side to {node:"", base_step_type:""}.',
  ].join('\n')
}

// Turn 2 — the gap-triage ladder ONLY (uses the located producer/consumer from turn 1).
function triagePrompt(req, locate) {
  return [
    HOWTO, '',
    'TASK (SOP step 2) — GAP TRIAGE ONLY. The nodes are already located: producer=' +
      JSON.stringify(locate.producer) + ', consumer=' + JSON.stringify(locate.consumer) + ', node_type=' + locate.node_type + '.',
    'USER INTENT: ' + req.intent + (req.name ? '   PROPOSED NAME: ' + req.name : '') + '   EDGE: ' + edgeStr(req) + '.',
    'Decide the ONE gap rung. Do NOT name the step or gather its axes yet. Return ONLY the triage result.',
    '',
    'GAP TRIAGE LADDER — climb in order, STOP at the first rung that fits, and set gap_rung + gap_rung_reason:',
    '  (a) reuse_config_only: an existing step_type already does this with only different config values? (no new step)',
    '  (b) extend_optional_dep: adding an OPTIONAL DependencySpec + one contract line to an EXISTING step (required=False,',
    '      backward-compatible, no builder change) covers it? (no new step)',
    '  (c) delete_node_artifact_exists: the artifact is already produced upstream/externally, so the node can be DELETED?',
    '  (d) new_step: only if (a)-(c) all fail — author a genuinely NEW step (full .step.yaml + config + script).',
    'Use catalog.step_info / catalog.resolve_step / steps.io on the closest existing step types to CHECK each cheaper',
    'rung for real. Also DEMOTE any candidate whose output is invariant across runs (deterministic from static inputs)',
    'to a config field. Set is_new_step = (gap_rung == "new_step").',
    '',
    'Return ONE JSON object {gap_rung, gap_rung_reason, is_new_step}.',
  ].join('\n')
}

// Turn 3 — the new-step IDENTITY (name/type/handler/exemplar/axes/divergences). Only reached when
// triage said new_step, so it always yields a single object with real divergences.
function identityPrompt(req, locate, triage) {
  return [
    HOWTO, '',
    'TASK (SOP step 2b + checklist) — NEW-STEP IDENTITY. Triage already decided gap_rung="' + triage.gap_rung +
      '" (new_step), producer=' + JSON.stringify(locate.producer) + ', consumer=' + JSON.stringify(locate.consumer) + '.',
    'USER INTENT: ' + req.intent + (req.name ? '   PROPOSED NAME: ' + req.name : '') + '   EDGE: ' + edgeStr(req) + '.',
    'Name and characterize the ONE new step. Do NOT write files. Return ONLY the identity result.',
    '',
    'Call author.checklist(sagemaker_step_type[, step_assembly]) for the ordered SOP + the bound handler + the exemplar;',
    'strategies.for_step_type to confirm the handler/knobs; author.rules("naming") + author.rules("reuse_class").',
    'step_name is PascalCase (IS the .step.yaml step_type + registry name + base of the DAG node string); snake_name is',
    'its snake_case file stem. needed_axes = only the behavior axes this step actually uses (subset of ' + JSON.stringify(AXES) + ').',
    '',
    'CAPTURE THE DIVERGENCES. The very reason reuse_config_only was rejected IS a concrete list of features the exemplar',
    'lacks — record them in divergences_from_exemplar as REQUIRED deltas the Author phase must add on top of the exemplar',
    'shape (it must NOT silently copy the exemplar and drop them). One entry per delta, naming the AXIS + the change, e.g.',
    '"outputs: ADD a `preprocessor` output (the fitted scaler .pkl) — the exemplar emits only processed_data" or "env_vars:',
    'ADD LABEL_FIELD_2". If you cannot name a single divergence, the reuse-ladder was mis-applied — this should not have',
    'been new_step. divergences_from_exemplar must be NON-empty.',
    '',
    'Return ONE JSON object {step_name, snake_name, sagemaker_step_type, step_assembly, framework, reuse_class,',
    'bound_handler, exemplar_step, needed_axes, divergences_from_exemplar}.',
  ].join('\n')
}

function alignPrompt(plan, req) {
  return [
    HOWTO, '',
    'TASK (SOP steps 3-6): ALIGN the two DAG edges for new step "' + plan.step_name + '" BEFORE writing the interface.',
    'EDGE: ' + edgeStr(req) + '. Bound handler: ' + plan.bound_handler + '. Exemplar: ' + plan.exemplar_step + '.',
    'Read the REAL specs with steps.io: steps.io("' + (plan.producer.base_step_type || plan.producer.node) + '"' +
      (plan.producer.job_type ? ', job_type="' + plan.producer.job_type + '"' : '') + ') for the producer OUTPUT spec, and',
    'steps.io("' + (plan.consumer.base_step_type || plan.consumer.node) + '"' +
      (plan.consumer.job_type ? ', job_type="' + plan.consumer.job_type + '"' : '') + ') for the consumer DEPENDENCY spec.',
    '',
    'STEP 3 — ARITY. The resolver assigns exactly ONE PropertyReference per DependencySpec; extra incoming data edges',
    'silently demote to ordering-only. Confirm the number of intended DATA producers into the new node (and into the',
    'consumer) equals the number of its data DependencySpec entries. If >1 producers of DIFFERENT step types, plan an',
    'optional DATA_SECONDARY dependency; if the SAME step type twice (a non-deterministic tie) collapse to one producer',
    'emitting a sidecar (e.g. reference_counts.json) read from the single channel. Set arity_ok + arity_note.',
    '',
    'STEP 4 — GATE-1 ENUM ALIGNMENT (do this FIRST; a mismatch HARD-ZEROS the 40% and usually the edge):',
    'For the producer->NEW edge, set the NEW dependency_type exact-or-compatible with the producer output_type; for the',
    'NEW->consumer edge, set the NEW output_type so the consumer dependency_type accepts it. Compatibility matrix:',
    'MODEL_ARTIFACTS->{MODEL_ARTIFACTS}; TRAINING_DATA<->PROCESSING_OUTPUT (bidirectional, compatible=0.20);',
    'HYPERPARAMETERS->{HYPERPARAMETERS,CUSTOM_PROPERTY}; PAYLOAD_SAMPLES->{PAYLOAD_SAMPLES,PROCESSING_OUTPUT}. Match both',
    'data_types (usually S3Uri). Set type_ok/data_type_ok per edge.',
    '',
    'STEP 5 — EDITING ROOM (additive, backward-compatible ONLY). The lever is the CONSUMER dependency compatible_sources:',
    'plan to ADD "' + plan.step_name + '" (the job-type-normalized BASE step type) to the consumer dependency compatible_sources',
    '(+0.10 vs +0.05-empty vs 0.0-wrong-entry), and ADD the counterpart logical_name as an ALIAS on the relevant OUTPUT spec',
    '(buys the 5% exact bonus + up to 25% semantic). For the producer->NEW edge, put the producer base step_type in the NEW',
    'dependency compatible_sources (or leave it EMPTY for +0.05 rather than a wrong entry). NEVER mutate an existing step',
    'output_type/dependency_type to force an edge. List every planned edit in consumer_edits.',
    '',
    'STEP 6 — GATE-2 SCORE (compute, do not eyeball). For BOTH edges compute the 6-component score',
    '(type 40 / data 20 / semantic-with-aliases 25 / exact-or-alias 5 / source-compat 10-or-5-or-0 / keyword 5). Use',
    'validate.deps_explain(name1, name2) for the 25% semantic sub-score between the two logical names while you iterate.',
    'Require each edge projected_score >= 0.5; flag < 0.7 as fragile; a type-compatible-only (0.20) edge MUST earn >= 0.30',
    'from the other components. If borderline, add alias/compatible_sources/keyword levers until it clears. A missing',
    'producer (source) or consumer (sink) side is not scored (mark that edge resolves=true, note "source"/"sink").',
    'Set ready = both real edges resolve after the planned edits. Return the ALIGN result — do NOT write files yet.',
  ].join('\n')
}

// The real (non-source/sink) edges for a request, as {edge, side} — used to drive one align/resolve
// turn per edge on the decomposed path. producer->NEW exists iff there is a producer; NEW->consumer
// iff there is a consumer.
function realEdges(plan, req) {
  const edges = []
  if (req.producer_node) edges.push({ edge: 'producer->NEW', side: 'producer' })
  if (req.consumer_node) edges.push({ edge: 'NEW->consumer', side: 'consumer' })
  return edges
}

// DECOMPOSED AlignEdges — one edge scored per turn (single object), for the non-tool-forcing host.
function edgeAlignPrompt(plan, req, edge) {
  const isProducerEdge = edge === 'producer->NEW'
  const near = isProducerEdge ? plan.producer : plan.consumer
  return [
    HOWTO, '',
    'TASK (SOP steps 4-6) — ALIGN EXACTLY ONE EDGE: "' + edge + '" for new step "' + plan.step_name + '" (' + edgeStr(req) + ').',
    'Score ONLY this edge and return ONE object. Do NOT score the other edge, do NOT write files, do NOT return a list.',
    'Bound handler: ' + plan.bound_handler + '. Exemplar: ' + plan.exemplar_step + '. This edge\'s adjacent node: ' + JSON.stringify(near) + '.',
    relay(
      'Read the REAL spec with steps.io("' + (near.base_step_type || near.node) + '"' + (near.job_type ? ', job_type="' + near.job_type + '"' : '') + ').',
      'Read the REAL spec by running this via your shell/execute tool and reading its JSON:\n  ' +
        cliJson('steps io ' + (near.base_step_type || near.node) + (near.job_type ? ' --job-type ' + near.job_type : '')) + '\n') +
    (isProducerEdge ? ' Use the producer OUTPUT spec (aligns to the NEW dependency-spec).' : ' Use the consumer DEPENDENCY spec (the NEW output-spec must satisfy it).'),
    '',
    'GATE-1 ENUM (do FIRST; a mismatch HARD-ZEROS the 40%): ' +
      (isProducerEdge ? 'set the NEW dependency_type exact-or-compatible with the producer output_type.'
                      : 'set the NEW output_type so the consumer dependency_type accepts it.') +
    ' Matrix: MODEL_ARTIFACTS->{MODEL_ARTIFACTS}; TRAINING_DATA<->PROCESSING_OUTPUT (0.20); ' +
    'HYPERPARAMETERS->{HYPERPARAMETERS,CUSTOM_PROPERTY}; PAYLOAD_SAMPLES->{PAYLOAD_SAMPLES,PROCESSING_OUTPUT}. Match data_types (usually S3Uri).',
    'EDITING ROOM (additive only): ' +
      (isProducerEdge ? 'put the producer base step_type in the NEW dependency compatible_sources (or leave EMPTY for +0.05 rather than a wrong entry).'
                      : 'plan to ADD "' + plan.step_name + '" to the consumer dependency compatible_sources (+0.10) and ADD the counterpart logical_name as an ALIAS on the NEW output.') +
    ' NEVER mutate an existing step output_type/dependency_type. List THIS edge\'s edits in consumer_edits.',
    'GATE-2 SCORE (compute, do not eyeball): the 6-component score (type 40 / data 20 / semantic-with-aliases 25 / exact-or-alias 5 / source-compat 10-or-5-or-0 / keyword 5).',
    'Use validate.deps_explain(name1,name2) for the 25% semantic sub-score. Require projected_score >= 0.5; flag < 0.7 fragile; a type-compatible-only (0.20) edge must earn >= 0.30 elsewhere.',
    '',
    'Return ONE JSON object {edge:"' + edge + '", dependency_type, output_type, type_ok, data_type_ok, projected_score, resolves, fragile, consumer_edits:[...]}.',
  ].join('\n')
}
function arityPrompt(plan, req) {
  return [
    HOWTO, '',
    'TASK (SOP step 3) — ARITY CHECK ONLY for new step "' + plan.step_name + '" (' + edgeStr(req) + '). Return ONE object.',
    'The resolver assigns exactly ONE PropertyReference per DependencySpec; extra incoming data edges silently demote to ordering-only.',
    'Confirm the number of intended DATA producers into the new node (and into the consumer) equals the number of its data DependencySpec entries.',
    'If >1 producers of DIFFERENT step types, an optional DATA_SECONDARY dependency is needed; if the SAME step type twice, collapse to one producer emitting a sidecar.',
    'Return ONE JSON object {arity_ok, arity_note}.',
  ].join('\n')
}

// DECOMPOSED Validate/re-resolve — score ONE edge per turn, for the non-tool-forcing host. Assembled
// into RESOLVE_SCHEMA shape in plain code. NOTE: validate.deps_resolve has NO cursus CLI (the resolver
// is MCP-only; `cursus validate deps-resolve` is a PHANTOM that hard-errors), so on the non-tool-forcing
// host this edge is REASONED from the two steps' `steps io` JSON, not resolver-verified. This green gate
// is therefore relaxed on the frozen host and re-verified by the real UnifiedDependencyResolver at CR.
function edgeResolvePrompt(plan, req, edge, nodes) {
  const near = edge === 'producer->NEW' ? plan.producer : plan.consumer
  return [
    relay(
      'The new step .step.yaml now exists, so the real resolver can score its edges. Call ' +
        'validate.deps_resolve(step_names=' + JSON.stringify(nodes) + ').',
      'The resolver has NO CLI on this host (validate.deps_resolve is MCP-only; there is NO `cursus validate ' +
        'deps-resolve` command — do NOT try it). Instead REASON about this one edge: use your shell tool to run\n  ' +
        cliJson('steps io ' + (plan.step_name)) + '\nand\n  ' + cliJson('steps io ' + (near.base_step_type || near.node)) +
        '\nread the dependency-spec + output-spec (logical_name, dependency_type/output_type, data_type, compatible_sources, ' +
        'aliases) from the two JSON blobs, and compute the 6-component score by the HOWTO rule (type 40 / data 20 / ' +
        'semantic-name+alias 25 / exact-or-alias 5 / source-compat 10-or-5-or-0 / keyword 5).'),
    'Report ONLY the single edge "' + edge + '" of ' + edgeStr(req) + ': its resolution score and whether it resolves (>=0.5).',
    'Return ONE JSON object {edge:"' + edge + '", score, resolves, note' + (HOST_TOOL_FORCES_SCHEMA ? '' : ':"reasoned from steps io JSON; no resolver CLI on this host"') + '}. Do NOT return the other edge, do NOT return a list.',
  ].join('\n')
}

function guidePrompt(plan, axis, align) {
  return [
    HOWTO, '',
    'TASK (SOP steps 7,10): gather field guidance for the "' + axis + '" axis of new ' + plan.sagemaker_step_type + ' step "' + plan.step_name + '".',
    'Bound handler: ' + plan.bound_handler + '. Exemplar to copy shape from: ' + plan.exemplar_step + '.',
    'The two edges are already aligned; honor that: ' + JSON.stringify((align && align.edges) || []) + '.',
    relay(
      'Call author.rules for the relevant topic (naming/packaging/sdk_carveout/closure as applicable),\n' +
        'strategies.knobs(axis, strategy) for legal knob values + defaults, and\n' +
        'steps.io / steps.patterns / catalog.config_fields on the exemplar for the wired shape.',
      'Use your shell/execute tool to gather the wired shape (read each command\'s JSON):\n  ' +
        cliJson('strategies knobs --axis ' + axis + ' --name ' + plan.bound_handler) + '\n  ' +
        cliJson('steps io ' + plan.exemplar_step) + '\n  ' +
        cliJson('steps patterns ' + plan.exemplar_step) + '\n  ' +
        cliJson('catalog fields ' + plan.exemplar_step) + '\n' +
        '(author.rules has no CLI — reason naming/packaging rules from the exemplar shape.)'),
    'For inputs/outputs: the dependency/output spec logical names + types + aliases + compatible_sources MUST match the',
    'AlignEdges decision above; the contract path for each is under the container roots — ' + CONTAINER_PATHS,
    relay(
      'For the config-class field VALUES (esp. env_vars / compute), ALSO call author.config_constraints(' + plan.exemplar_step + ')\n' +
        'to get each field allowed_values + case_sensitive + the required_no_default list — so you write a LEGAL enum value\n' +
        '(e.g. output_format CSV/TSV/Parquet) and supply every required field, not a guessed one.',
      'For the config-class field VALUES: `catalog fields ' + plan.exemplar_step + '` JSON (above) gives each field\'s ' +
        'required flag + default (default "PydanticUndefined" == required-no-default) — use it for the required set. ' +
        'author.config_constraints has NO CLI, so for allowed_values + case-sensitivity reason from the exemplar ' +
        '.step.yaml + the config class @field_validator (HABIT-1: read the validator, not the docstring — output_format is ' +
        'case-sensitive CSV/TSV/Parquet). Do NOT guess a field.') + ' Do NOT write any file.',
    'Return the SECTION_GUIDE for this axis.',
  ].join('\n')
}

function authorPrompt(plan, guides, align, req) {
  const s = plan.snake_name
  return [
    HOWTO, '',
    'TASK (SOP steps 7,9,10): write the THREE artifacts for new step "' + plan.step_name + '" using your Write tool, AND',
    'edit the producer/consumer .step.yaml specs so they accept the new step:',
    '  1. src/cursus/steps/interfaces/' + s + '.step.yaml  (registry / patterns / compute / contract / spec sections)',
    '  2. src/cursus/steps/configs/config_' + s + '_step.py  (the ' + plan.step_name + 'Config three-tier Pydantic class)',
    '  3. src/cursus/steps/scripts/' + s + '.py  (main(input_paths, output_paths, environ_vars, job_args, logger=None);',
    '     for reuse_class=model_dependent, leave the MODEL section as a documented OPEN SECTION skeleton with a',
    '     numbered checklist of what each project must implement — cursus ships the sub-steps, not the model).',
    '  4. THE EDGE EDITS (additive, backward-compatible): apply every item in consumer_edits by EDITING the adjacent',
    '     .step.yaml spec section(s): ' + JSON.stringify((align && align.consumer_edits) || []),
    '',
    'AUTHORING MODEL — exemplar-PLUS-deltas, NOT clone-the-exemplar. Start from the ' + plan.exemplar_step + ' shape,',
    'then APPLY every REQUIRED DIVERGENCE below. These are the exact features that make this a NEW step rather than a',
    'reuse of the exemplar — dropping any one of them reproduces the exemplar and defeats the reason this step exists:',
    '  REQUIRED DIVERGENCES FROM EXEMPLAR: ' + JSON.stringify(plan.divergences_from_exemplar || []),
    '  After writing, VERIFY each divergence is actually present in the artifact (e.g. if a divergence says "ADD a',
    '  `preprocessor` output", confirm both contract.outputs.preprocessor AND spec.outputs.preprocessor exist). A',
    '  divergence the exemplar lacks and you did not add is a BUG, not a simplification.',
    '',
    'SPEC (from AlignEdges — do not drift): the NEW dependency-spec + output-spec types/logical-names/aliases/',
    'compatible_sources are FIXED by the edge alignment: ' + JSON.stringify((align && align.edges) || []),
    'CONTRACT (SOP step 7): ' + CONTAINER_PATHS + ' Layer {job_type}/... beneath the root; contract dict KEYS ==',
    'spec logical_names byte-for-byte. Optional inputs -> presence-test + required=False + listed in env_vars.optional.',
    'SCRIPT (SOP step 9): main() is indexed ONLY by contract logical_names — NEVER hardcode /opt/ml inside main().',
    'Copy the __main__ block + container-path constants VERBATIM from the exemplar (' + plan.exemplar_step + '); read every',
    'behavioral knob via environ_vars.get(NAME, default); read inputs with rglob (not a flat glob) so nested {job_type}/',
    'files are found. EMIT the FULL downstream-consumed artifact set even for artifacts this script does not itself use',
    '(this INCLUDES every output named in the REQUIRED DIVERGENCES — write the file the divergence artifact represents).',
    '',
    'PLAN: ' + JSON.stringify(plan),
    'PER-AXIS GUIDANCE: ' + JSON.stringify(guides),
    'Use the exemplar ' + plan.exemplar_step + ' as the STARTING shape, then apply the required divergences above.',
    'registry.sagemaker_step_type=' + plan.sagemaker_step_type + '.',
    'Do NOT edit any registry file (registry is derived by construction). Report the written file paths, the exact',
    'producer/consumer spec edits you made, AND for each required divergence a one-line confirmation it is present.',
  ].join('\n')
}

// ---------------------------------------------------------------------------
// PIPELINE: each request flows Resolve -> AlignEdges -> (Guide barrier + Author) -> (Validate loop +
// Preflight) independently. A request whose gap ladder resolved to a NON-new-step rung (or whose
// edges cannot be made to resolve) short-circuits to Synthesize with that verdict — the workflow does
// NOT author a step it should not. Gates are STAGES, so a step can't reach Synthesize green without
// passing the same checks CI runs.
// ---------------------------------------------------------------------------
const report = await pipeline(REQUESTS,

  // Stage 1 — Resolve (locate node + gap ladder) then Challenge the rung (verdict-override).
  // On a TOOL-FORCING host (Claude Code) Resolve is ONE tool-forced PLAN_SCHEMA turn (resolvePrompt) —
  // the original behavior, unchanged. On a NON-tool-forcing host (the Kiro runtime) it is THREE small
  // sequential turns (locate -> triage -> identity) merged into the SAME PLAN_SCHEMA-shaped `plan`,
  // because a big all-at-once schema made even Opus 4.8 emit a per-node ARRAY on the frozen kiro-cli
  // 2.5.0 snapshot that the harness could not coerce. Either branch yields the identical `plan` shape,
  // so everything downstream (Challenge, AlignEdges, ...) is unchanged.
  async (req) => {
    const tag = req.name || req.intent.slice(0, 20)
    let plan, identity = null
    if (HOST_TOOL_FORCES_SCHEMA) {
      // Claude Code: one tool-forced turn returns the whole PLAN_SCHEMA object reliably.
      plan = await agent(resolvePrompt(req), { label: 'resolve:' + tag, phase: 'Resolve', schema: PLAN_SCHEMA, effort: 'high' })
      if (!plan) return null
    } else {
      // Kiro: split into three single-object turns, then merge into a PLAN_SCHEMA-shaped plan.
      // Turn 1 — locate + decompose the two adjacent nodes.
      const locate = await agent(locatePrompt(req), { label: 'locate:' + tag, phase: 'Resolve', schema: LOCATE_SCHEMA, effort: 'high' })
      if (!locate) return null
      // Turn 2 — the gap-triage ladder.
      const triage = await agent(triagePrompt(req, locate), { label: 'triage:' + tag, phase: 'Resolve', schema: TRIAGE_SCHEMA, effort: 'high' })
      if (!triage) return null
      // Turn 3 — new-step identity (name/type/handler/exemplar/axes/divergences) — only when new_step.
      if (triage.is_new_step) {
        identity = await agent(identityPrompt(req, locate, triage), { label: 'identity:' + tag, phase: 'Resolve', schema: IDENTITY_SCHEMA, effort: 'high' })
        if (!identity) return null
      }
      // Merge into one PLAN_SCHEMA-shaped plan. For a non-new-step rung, identity is skipped and the
      // identity fields are empty placeholders (downstream never reads them for a short-circuited item).
      plan = {
        step_name: identity ? identity.step_name : (req.name || ''),
        snake_name: identity ? identity.snake_name : '',
        sagemaker_step_type: identity ? identity.sagemaker_step_type : '',
        step_assembly: identity ? identity.step_assembly : '',
        node_type: locate.node_type,
        framework: identity ? identity.framework : '',
        reuse_class: identity ? identity.reuse_class : 'shared',
        bound_handler: identity ? identity.bound_handler : '',
        exemplar_step: identity ? identity.exemplar_step : '',
        needed_axes: identity ? (identity.needed_axes || []) : [],
        gap_rung: triage.gap_rung,
        gap_rung_reason: triage.gap_rung_reason || '',
        is_new_step: triage.is_new_step,
        divergences_from_exemplar: identity ? (identity.divergences_from_exemplar || []) : [],
        producer: locate.producer,
        consumer: locate.consumer,
      }
      // stash locate for the post-challenge identity backfill below (Kiro path only)
      plan.__locate = locate
    }
    // Verdict-override: one skeptic tries to REVERSE the gap rung. A wrong CHEAPER rung
    // (reuse/extend/delete) is the dangerous case — it short-circuits with the need unmet — so pressure
    // it hardest. Reuses the exact gap_rung enum so the challenge authoritatively resets the rung.
    const ch = await agent(
      'TRY TO BREAK this gap-ladder routing decision for "' + (req.name || plan.step_name) + '" (edge ' + edgeStr(req) + '). ' +
      'The Resolve agent chose gap_rung="' + plan.gap_rung + '" (reason: ' + (plan.gap_rung_reason || 'n/a') + '). ' +
      relay(
        'Run the ACTUAL checks to refute it: catalog.step_info / catalog.resolve_step / steps.io on the closest existing step types.',
        'Run the ACTUAL checks to refute it: for each closest existing step type X, use your shell/execute tool to run\n  ' +
          cliJson('catalog show X') + '\nand\n  ' + cliJson('steps io X') + '\nand read the printed JSON (what the step does + its I/O).') +
      ' Is there ' +
      'REALLY no existing step_type that does this config-only (reuse_config_only)? no backward-compatible OPTIONAL-dependency ' +
      'extension of an existing step (extend_optional_dep)? is the artifact already produced upstream (delete_node_artifact_exists)? ' +
      'Pressure the CHEAPER rungs HARDEST — a wrong cheaper rung silently short-circuits with the need unmet; only escalate to ' +
      'new_step if the cheaper rungs genuinely do not fit. Return {holds, challenges, final_rung (same enum), correction}.',
      { label: 'challenge:' + (req.name || plan.step_name), phase: 'Challenge', schema: CHALLENGE_SCHEMA, effort: 'high' })
    if (ch && ch.holds === false && ch.final_rung) {
      plan.gap_rung = ch.final_rung
      plan.is_new_step = (ch.final_rung === 'new_step')
      if (ch.correction) plan.gap_rung_reason = 'CHALLENGE-REVERSED: ' + ch.correction
      // KIRO PATH ONLY: if the challenge ESCALATED a cheaper rung UP to new_step, the identity turn was
      // never run (turn 3 only runs when triage said new_step), so backfill it now — AlignEdges/Author
      // need step_name/exemplar/divergences. The CC single-turn plan already carries identity fields for
      // every rung, so no backfill there (guarded by plan.__locate, set only on the Kiro path).
      if (plan.is_new_step && !identity && plan.__locate) {
        identity = await agent(identityPrompt(req, plan.__locate, { gap_rung: plan.gap_rung }), { label: 'identity:' + tag + ':post-challenge', phase: 'Resolve', schema: IDENTITY_SCHEMA, effort: 'high' })
        if (identity) {
          plan.step_name = identity.step_name
          plan.snake_name = identity.snake_name
          plan.sagemaker_step_type = identity.sagemaker_step_type
          plan.step_assembly = identity.step_assembly
          plan.framework = identity.framework
          plan.reuse_class = identity.reuse_class
          plan.bound_handler = identity.bound_handler
          plan.exemplar_step = identity.exemplar_step
          plan.needed_axes = identity.needed_axes || []
          plan.divergences_from_exemplar = identity.divergences_from_exemplar || []
        }
      }
    }
    if (plan.__locate) delete plan.__locate // internal-only; keep the plan PLAN_SCHEMA-clean
    return { req, plan, challenge: ch }
  },

  // Stage 2 — AlignEdges (the spec-alignment gate) — only for genuine new steps.
  // On a TOOL-FORCING host (Claude Code): one turn returns ALIGN_SCHEMA. On the non-tool-forcing Kiro
  // host: score ONE edge per turn (single object) + a tiny arity turn, then ASSEMBLE the ALIGN_SCHEMA
  // shape in plain code — avoiding the edges[] array-of-objects that triggered the AlignEdges failure
  // in SAIS Runs 5-6. Either branch yields the identical `align` shape, so Guide/Author are unchanged.
  async (ctx) => {
    if (!ctx || !ctx.plan) return null
    if (!ctx.plan.is_new_step) return { ...ctx, align: null, short_circuit: 'gap_rung=' + ctx.plan.gap_rung + ': ' + (ctx.plan.gap_rung_reason || 'no new step needed') }
    let align
    if (HOST_TOOL_FORCES_SCHEMA) {
      align = await agent(alignPrompt(ctx.plan, ctx.req), { label: 'align:' + ctx.plan.step_name, phase: 'AlignEdges', schema: ALIGN_SCHEMA, effort: 'high' })
    } else {
      // Kiro: one turn per real edge + one arity turn, assembled into an ALIGN_SCHEMA-shaped object.
      const expectedEdges = realEdges(ctx.plan, ctx.req).length
      const edges = []
      for (const { edge } of realEdges(ctx.plan, ctx.req)) {
        const e = await agent(edgeAlignPrompt(ctx.plan, ctx.req, edge), { label: 'align:' + ctx.plan.step_name + ':' + edge, phase: 'AlignEdges', schema: EDGE_ALIGN_SCHEMA, effort: 'high' })
        if (e) { e.edge = edge; edges.push(e) } // inject the edge label the runtime dictated (model needn't echo it)
      }
      const ar = await agent(arityPrompt(ctx.plan, ctx.req), { label: 'align:' + ctx.plan.step_name + ':arity', phase: 'AlignEdges', schema: ARITY_SCHEMA, effort: 'high' })
      // CRITICAL — match the single-turn contract's control flow. When the single-turn AlignEdges agent
      // returned null (couldn't produce a scorable result), `align` was null and the pipeline PROCEEDED
      // to Author (the real edge gate is the post-write re-resolve at Stage 4c). The per-edge path must
      // do the same: if NOT ALL expected edges were scored, alignment is INCONCLUSIVE, not "failed" — set
      // align=null and proceed, rather than fabricating ready:false and short-circuiting (the Run-11
      // regression, where edge turns returned non-JSON, decomposition built ready:false, and the workflow
      // authored 0 artifacts even though the post-write oracles would have caught the real issues).
      if (edges.length < expectedEdges) {
        align = null // inconclusive: fewer edges scored than expected — proceed, defer to Stage 4c re-resolve
      } else {
        const consumer_edits = [...new Set(edges.flatMap(e => e.consumer_edits || []))]
        align = {
          arity_ok: ar ? ar.arity_ok : false,
          arity_note: ar ? (ar.arity_note || '') : 'arity turn failed',
          edges: edges.map(e => ({ edge: e.edge, dependency_type: e.dependency_type, output_type: e.output_type, type_ok: e.type_ok, data_type_ok: e.data_type_ok, projected_score: e.projected_score, resolves: e.resolves, fragile: e.fragile })),
          consumer_edits,
          // Every expected edge scored → ready iff they all resolve (a node with zero real edges is
          // vacuously ready, matching the single-turn path).
          ready: edges.every(e => e.resolves === true),
        }
      }
    }
    // Short-circuit ONLY on a DEFINITIVE non-resolution (align present AND ready:false). A null align
    // (single-turn agent died, or per-edge inconclusive) proceeds to Author — the authoritative edge
    // gate is the post-write re-resolve (Stage 4c) + the executable oracles, exactly as before the split.
    return { ...ctx, align, short_circuit: (align && align.ready === false) ? 'edges do not resolve (>=0.5) even after the planned consumer edits — revisit the DAG or the step design' : null }
  },

  // Stage 3 — Guide (per-axis barrier) then Author (the agent's own Write) — skipped if short-circuited
  async (ctx) => {
    if (!ctx || !ctx.plan) return null
    if (ctx.short_circuit) return ctx
    const guides = (await parallel((ctx.plan.needed_axes || []).map(ax => () =>
      agent(guidePrompt(ctx.plan, ax, ctx.align), { label: 'guide:' + ctx.plan.step_name + ':' + ax, phase: 'Guide', schema: GUIDE_SCHEMA })
        .then(g => { if (g && !g.axis) g.axis = ax; return g }) // inject the axis the runtime dictated (model needn't echo it)
    ))).filter(Boolean)
    await agent(authorPrompt(ctx.plan, guides, ctx.align, ctx.req), { label: 'author:' + ctx.plan.step_name, phase: 'Author', effort: 'high' })
    return { ...ctx, guides }
  },

  // Stage 4 — Validate (bounded fix loops) + re-resolve both edges, then Preflight — skipped if short-circuited
  async (ctx) => {
    if (!ctx || !ctx.plan) return null
    if (ctx.short_circuit) return { plan: ctx.plan, req: ctx.req, short_circuit: ctx.short_circuit }
    const plan = ctx.plan

    // 4a. interface gate (contract<->spec + container-path Pydantic validators)
    let v, tries = 0
    do {
      v = await agent(
        relay(
          'Call validate.step_interface(step_name="' + plan.step_name + '") (CLI fallback: `cursus validate step-interface ' + plan.step_name + ' --format json`). Return its {ok, errors, warnings}.',
          'Use your shell/execute tool to run this EXACT command and read its stdout:\n  ' + cliJson('validate step-interface ' + plan.step_name) +
          '\nIt prints {validated, errors (a COUNT), warnings (a COUNT), results:[{step, ok, errors:[...], warnings:[...]}]}. ' +
          'Return EXACTLY {ok: results[0].ok, errors: results[0].errors, warnings: results[0].warnings} — map from results[0]; ' +
          'the TOP-LEVEL errors/warnings are integer counts, do NOT use them (VALIDATE_SCHEMA wants string arrays).'),
        { label: 'validate:' + plan.step_name, phase: 'Validate', schema: VALIDATE_SCHEMA })
      if (v && !v.ok && tries < 2) {
        await agent(
          'validate.step_interface failed for ' + plan.step_name + ' with errors: ' + JSON.stringify(v.errors) +
          '. Common: an input/output path outside the container roots (' + CONTAINER_PATHS + '), or a contract logical name that ' +
          'does not match its spec logical name. Read the named .step.yaml section and edit ONLY the offending field(s). FINALLY ' +
          'run `python3 -c "import yaml; yaml.safe_load(open(\'src/cursus/steps/interfaces/' + plan.snake_name + '.step.yaml\'))"` ' +
          'and return {file_edited, change_summary, py_compile_ok (true if the yaml.safe_load exits 0)}.',
          { label: 'fix:' + plan.step_name + ':' + tries, phase: 'Validate', schema: FIX_SCHEMA, effort: 'high' })
      }
      tries++
    } while (v && !v.ok && tries <= 2)

    // 4b. script<->contract alignment (both directions) incl. --job_type argparse + the {job_type} subdir layout.
    let cs, ctries = 0
    do {
      cs = await agent(
        relay(
          'Call author.check_script(step_name="' + plan.step_name + '"). Return {status, passed, issues}. ' +
          'status:"skipped" means a script-less / SDK-delegation step (treat as pass).',
          'author.check_script has NO cursus CLI on this host (there is no `author` CLI group), so it cannot run here. ' +
          'Return {status:"skipped", passed:true, issues:[]} — this gate is relaxed to skip on the frozen host (the ' +
          'publish path permits skip; the py_compile/yaml PARSE oracle below backstops syntax, and check_script re-runs at CR).') +
        ' Beyond that, ALSO reason ' +
        'about SOP step 8 (the physical contract below the spec): does the producer emit its {job_type} subdir where the ' +
        'consumer scans, or does the script rglob recursively? A flat glob("*.parquet") that misses nested {job_type}/ files ' +
        'is a real bug even when the tool passes — record it as an issue if present.',
        { label: 'check_script:' + plan.step_name, phase: 'Validate', schema: CHECK_SCRIPT_SCHEMA })
      if (cs && cs.status !== 'skipped' && !cs.passed && ctries < 2) {
        const fx = await agent(
          'author.check_script failed for ' + plan.step_name + ' with issues: ' + JSON.stringify(cs.issues || []) +
          '. Fix the SCRIPT/contract: for unparsed_declared_arg add parser.add_argument("--<flag>") in the script ' +
          '__main__ block OR remove the flag from the .step.yaml contract.job_arguments; for unread_required_env_var ' +
          'read it via environ_vars.get("<VAR>") in main() OR demote it to env_vars.optional; for invalid_main_signature ' +
          'fix main(input_paths, output_paths, environ_vars, job_args, logger=None); for a subdir/glob issue switch to ' +
          'rglob or align the {job_type} dir names. Edit ONLY the offending file(s). FINALLY run ' +
          '`python3 -m py_compile src/cursus/steps/scripts/' + plan.snake_name + '.py` and return ' +
          '{file_edited, change_summary, py_compile_ok (from the exit code)}.',
          { label: 'fix_script:' + plan.step_name + ':' + ctries, phase: 'Validate', schema: FIX_SCHEMA, effort: 'high' })
        if (fx && fx.py_compile_ok === false) log('WARNING: fix_script:' + plan.step_name + ' left the script not compiling — escalate (gate/contract bug, not content)')
      }
      ctries++
    } while (cs && cs.status !== 'skipped' && !cs.passed && ctries <= 2)

    // 4b'. Executable PARSE oracle: py_compile the script + config .py and yaml.safe_load the .step.yaml.
    // Every other green-gate boolean is agent-relayed tool JSON; this is the one check that fails hard on
    // a syntax/parse error, which is the hardest thing to fake plausibly. (author.check_script status
    // 'skipped' gives ZERO script coverage for SDK/script-less steps — but a script-less step has no
    // script.py to compile; the agent reports py_compile_ok=true for the absent-script case.)
    const parse = await agent(
      'Confirm the authored artifacts for "' + plan.step_name + '" actually PARSE (run the commands, do not assume):\n' +
      '  python3 -m py_compile src/cursus/steps/configs/config_' + plan.snake_name + '_step.py' +
      ' src/cursus/steps/scripts/' + plan.snake_name + '.py   (omit the script path if this is a script-less/SDK step)\n' +
      '  python3 -c "import yaml; yaml.safe_load(open(\'src/cursus/steps/interfaces/' + plan.snake_name + '.step.yaml\'))"\n' +
      'Return {py_compile_ok, yaml_loads_ok, detail} from the EXIT CODES. If py_compile fails, fix the syntax error ' +
      '(edit ONLY the broken file) and re-run until it exits 0.',
      { label: 'parse:' + plan.step_name, phase: 'Validate', schema: PARSE_SCHEMA, effort: 'high' })

    // 4b''. DIVERGENCE-FIDELITY gate: every REQUIRED delta the Resolve phase said makes this a new step
    // must actually be in the authored artifact. This catches the copy-the-exemplar-and-drop-the-
    // distinguishing-feature bug (the empirical TSATabularPreprocessing run authored the step but dropped
    // the `preprocessor` output that was its whole reason to exist). Bounded fix loop, same shape as 4a/4b.
    const divs = ctx.plan.divergences_from_exemplar || []
    let dv = null
    if (divs.length) {
      let dtries = 0
      do {
        dv = await agent(
          'The new step "' + plan.step_name + '" is a NEW step (not a reuse of exemplar ' + plan.exemplar_step + ') ' +
          'BECAUSE of these REQUIRED divergences: ' + JSON.stringify(divs) + '. Read the authored artifacts ' +
          '(src/cursus/steps/interfaces/' + plan.snake_name + '.step.yaml, config_' + plan.snake_name + '_step.py, ' +
          plan.snake_name + '.py) and confirm EACH divergence is actually present (e.g. a "ADD a preprocessor output" ' +
          'divergence needs BOTH contract.outputs.preprocessor AND spec.outputs.preprocessor, and the script writing it). ' +
          'Return {all_present, checks:[{divergence, present, evidence}]}. A divergence that is absent means the artifact ' +
          'silently reverted to the exemplar shape.',
          { label: 'divergence_check:' + plan.step_name, phase: 'Validate', schema: DIVERGENCE_SCHEMA, effort: 'high' })
        if (dv && !dv.all_present && dtries < 2) {
          const missing = (dv.checks || []).filter(c => !c.present).map(c => c.divergence)
          await agent(
            'The authored "' + plan.step_name + '" is MISSING these required divergences (it copied the exemplar and ' +
            'dropped them): ' + JSON.stringify(missing) + '. ADD each one to the correct artifact + section (an output ' +
            'delta goes in BOTH contract.outputs and spec.outputs of the .step.yaml AND is written by the script; an ' +
            'env_var delta goes in contract.env_vars). Edit ONLY what is needed to add the missing deltas, then stop.',
            { label: 'fix_divergence:' + plan.step_name + ':' + dtries, phase: 'Validate', effort: 'high' })
        }
        dtries++
      } while (dv && !dv.all_present && dtries <= 2)
    }

    // 4c. re-resolve BOTH edges with the REAL resolver now that all three specs exist (SOP step 6, for real).
    // Tool-forcing host: one turn returns RESOLVE_SCHEMA. Non-tool-forcing host: one turn per real edge
    // (single object) assembled into RESOLVE_SCHEMA shape — avoids the edges[] array bias (Validate/
    // resolve failure in SAIS Runs 5-6). Identical `rr` shape either way, so Synthesize is unchanged.
    const nodes = [ctx.req.producer_node, ctx.plan.step_name, ctx.req.consumer_node].filter(Boolean)
    let rr
    if (HOST_TOOL_FORCES_SCHEMA) {
      rr = await agent(
        'The new step .step.yaml now exists, so the real resolver can score its edges. Call ' +
        'validate.deps_resolve(step_names=' + JSON.stringify(nodes) + ') and report, for each of the edges ' +
        edgeStr(ctx.req) + ', the resolution score and whether it resolves (>=0.5). A source/sink side that has no ' +
        'producer/consumer is not an edge. Return {both_edges_resolve, edges:[{edge,score,resolves,note}]}.',
        { label: 'resolve:' + plan.step_name, phase: 'Validate', schema: RESOLVE_SCHEMA, effort: 'high' })
    } else {
      const redges = []
      for (const { edge } of realEdges(plan, ctx.req)) {
        const e = await agent(edgeResolvePrompt(plan, ctx.req, edge, nodes), { label: 'resolve:' + plan.step_name + ':' + edge, phase: 'Validate', schema: EDGE_RESOLVE_SCHEMA, effort: 'high' })
        if (e) { e.edge = edge; redges.push(e) } // inject the edge label the runtime dictated (model needn't echo it)
      }
      const expected = realEdges(plan, ctx.req).length
      // both_edges_resolve when every expected real edge was scored AND resolves (vacuously true for a
      // node with no real edges — matches the single-turn RESOLVE_SCHEMA contract).
      rr = { both_edges_resolve: redges.length === expected && redges.every(e => e.resolves === true), edges: redges }
    }

    // 4d. constructibility proof (CI merge gate) + whole-DAG compile preview.
    const pf = await agent(
      relay(
        'Call author.preflight_step(step_name="' + plan.step_name + '") under offline (PYTHONNOUSERSITE=1) semantics; then ' +
        'sanity-check the whole DAG the new step lives in via compile.preview / compile.validate (nodes ' +
        JSON.stringify(nodes) + ') so the new node resolves to its config + builder with high confidence. ' +
        'Return {constructible, gates:[{name,passed,detail}]}. SDK-delegation rows must be skip-not-error.',
        'author.preflight_step and compile.preview have NO cursus CLI on this host (no `author` group; the compile ' +
        'command has --validate-only but no preview). Assess constructibility by REASONING: the .step.yaml parsed ' +
        '(the PARSE oracle above), its registry.sagemaker_step_type is a known type, and its dependency/output specs ' +
        'match the AlignEdges decision — from these set constructible + a gate {name:"preflight", passed:<bool>, ' +
        'detail:"reasoned; author.preflight_step CLI absent on frozen host — re-verified at CR"}. Return ' +
        '{constructible, gates:[{name,passed,detail}]}. This gate is relaxed (reasoned, not preflight-verified) on the frozen host.'),
      { label: 'preflight:' + plan.step_name, phase: 'Preflight', schema: PREFLIGHT_SCHEMA, effort: 'high' })

    // Gaps: completeness critic over the FULL DAG (not the 3-node triple). The additive
    // compatible_sources edit on the shared consumer dependency can regress ANOTHER producer's
    // pre-existing edge (empty compatible_sources +0.05 -> 0.0 when a wrong entry is added); and the
    // habit-4 cross-branch column trace has no executable backing. Only when we have a dag_path to scan.
    let gaps = null
    if (ctx.req.dag_path) {
      gaps = await agent(
        'COMPLETENESS CRITIC for new step "' + plan.step_name + '" (edge ' + edgeStr(ctx.req) + '). Read the FULL DAG at ' +
        ctx.req.dag_path + ' — the WHOLE node set, not just the ' + JSON.stringify(nodes) + ' triple. Hunt for what the ' +
        'per-edge gates MISS:\n' +
        '(1) EDIT COLLATERAL: the consumer_edits added "' + plan.step_name + '" to an EXISTING consumer dependency\'s ' +
        'compatible_sources (' + JSON.stringify((ctx.align && ctx.align.consumer_edits) || []) + '). Run validate.deps_resolve ' +
        'over the full node set and check whether any OTHER producer feeding that same dependency dropped below 0.5 (an empty ' +
        'compatible_sources scored +0.05 for everyone; adding one entry can zero the others). Report before/after per affected step.\n' +
        '(2) CROSS-BRANCH COLUMN SURVIVAL (habit 4, no tool backing — reason from the contracts/scripts): trace the producer ' +
        'output column/filename names vs the columns the consumer script reads, on BOTH the train AND the calibration/eval ' +
        'branch. Renames do not propagate across parallel branches. Report mismatches per branch.\n' +
        '(3) uncovered_consumers / sequencing_risks if any. Return {full_dag_scanned (true only if you read the whole dag_path ' +
        'node set), edit_collateral, uncovered_consumers, cross_branch_column_mismatches, sequencing_risks}. Be concrete; do not pad.',
        { label: 'gaps:' + plan.step_name, phase: 'Gaps', schema: GAPS_SCHEMA, effort: 'high' })
    }
    return { plan, req: ctx.req, align: ctx.align, validate: v, check_script: cs, parse: parse, divergence: dv, resolve: rr, preflight: pf, gaps: gaps }
  },
)

// Stage 5 — Synthesize (plain code; only this returns to the conversation)
const done = report.filter(Boolean)
const shorted = done.filter(r => r.short_circuit)
const authored = done.filter(r => !r.short_circuit && r.plan && r.plan.is_new_step)
// A gate that returned null (dead/skipped agent) is NOT a pass. check_script status:'skipped' IS a
// legitimate pass (script-less/SDK step); the parse oracle backstops that skip since a script-less step
// still has a config.py + .step.yaml to parse.
const scriptOk = (r) => !!r.check_script && (r.check_script.status === 'skipped' || r.check_script.passed === true)
const parseOk = (r) => !!r.parse && r.parse.py_compile_ok === true && r.parse.yaml_loads_ok === true
// Divergence fidelity: if Resolve named required deltas over the exemplar, they must all be present.
// No divergences declared -> vacuously ok. A missing divergence = a silent exemplar-clone bug, NOT green.
const divergenceOk = (r) => {
  const declared = (r.plan && r.plan.divergences_from_exemplar) || []
  if (!declared.length) return true
  return !!r.divergence && r.divergence.all_present === true
}
// Open gaps found by the completeness critic: edit collateral or a cross-branch column mismatch.
const hasOpenGaps = (r) => !!r.gaps && (((r.gaps.edit_collateral || []).length > 0) || ((r.gaps.cross_branch_column_mismatches || []).length > 0))
const passedCore = (r) =>
  !!(r.validate && r.validate.ok) && scriptOk(r) && parseOk(r) && divergenceOk(r) &&
  !!(r.resolve && r.resolve.both_edges_resolve) &&
  !!(r.preflight && r.preflight.constructible)
const green = authored.filter(r => passedCore(r) && !hasOpenGaps(r))
const greenWithGaps = authored.filter(r => passedCore(r) && hasOpenGaps(r))
const statusOf = (r) => passedCore(r) ? (hasOpenGaps(r) ? 'green_with_open_gaps' : 'green') : 'blocked'
log('processed ' + done.length + ' request(s); ' + authored.length + ' new step(s) authored, ' + green.length + ' green + ' + greenWithGaps.length + ' green_with_open_gaps; ' + shorted.length + ' resolved without a new step')
return {
  authored: authored.map(r => ({
    step: r.plan.step_name,
    edge: edgeStr(r.req),
    status: statusOf(r),
    sagemaker_step_type: r.plan.sagemaker_step_type,
    gap_rung: r.plan.gap_rung,
    rung_challenged: r.challenge ? (r.challenge.holds === false ? 'REVERSED to ' + (r.challenge.final_rung || '?') : 'held') : null,
    files: [
      'src/cursus/steps/interfaces/' + r.plan.snake_name + '.step.yaml',
      'src/cursus/steps/configs/config_' + r.plan.snake_name + '_step.py',
      'src/cursus/steps/scripts/' + r.plan.snake_name + '.py',
    ],
    producer_consumer_edits: r.align ? (r.align.consumer_edits || []) : [],
    edge_scores: r.resolve ? (r.resolve.edges || []) : [],
    validate_ok: r.validate ? r.validate.ok : null,
    script_ok: r.check_script ? (r.check_script.status === 'skipped' ? 'skipped' : r.check_script.passed) : null,
    parse_ok: r.parse ? (r.parse.py_compile_ok === true && r.parse.yaml_loads_ok === true) : null,
    divergences_required: (r.plan && r.plan.divergences_from_exemplar) || [],
    divergences_present: r.divergence ? r.divergence.all_present : (((r.plan && r.plan.divergences_from_exemplar) || []).length === 0 ? true : null),
    divergences_missing: r.divergence ? (r.divergence.checks || []).filter(c => !c.present).map(c => c.divergence) : [],
    edges_resolve: r.resolve ? r.resolve.both_edges_resolve : null,
    constructible: r.preflight ? r.preflight.constructible : null,
    failed_gates: r.preflight ? (r.preflight.gates || []).filter(g => !g.passed).map(g => g.name) : [],
    open_gaps: r.gaps ? {
      edit_collateral: r.gaps.edit_collateral || [],
      cross_branch_column_mismatches: r.gaps.cross_branch_column_mismatches || [],
      uncovered_consumers: r.gaps.uncovered_consumers || [],
    } : null,
  })),
  resolved_without_new_step: shorted.map(r => ({ intent: r.req && r.req.intent, edge: r.req ? edgeStr(r.req) : null, verdict: r.short_circuit, gap_rung: r.plan ? r.plan.gap_rung : null })),
  publish_path: 'A step is "green" only if validate.step_interface ok + author.check_script pass/skip + the py_compile/yaml PARSE oracle + every REQUIRED divergence-from-exemplar present (no silent exemplar clone) + both edges resolve + preflight constructible, AND the completeness critic found no open gaps (else green_with_open_gaps — flagged, not silently shipped). These are NECESSARY, not SUFFICIENT: they cover the single new step interface + its two edges + parse-ability. Before publish, ALSO run `cursus validate step-interface --all` (catches cross-step/registry-parity breakage the per-step gate cannot), and note that registry-parity / RegistryBindingValidator B3 / routability / brazil-build still run at CR. Habit 4: a green edge is a WIRE proof, not a runtime proof — for a data-carrying step run validate.run_scripts (or a sampled dry-run) and confirm the cross-branch column trace. Then: CR + brazil pb build → cursus-peru-dev → cursus-peru-shared → CDK → CodeArtifact.',
}
