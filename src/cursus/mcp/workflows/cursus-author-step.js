export const meta = {
  name: 'cursus-author-step',
  description: 'Drive DAG-driven agent-tool-driven creation of a cursus 2.0 (Design B) step. Input is a user PipelineDAG in which exactly ONE node is a NEW, unregistered step sitting BETWEEN a producer and a consumer. The workflow: locate the undefined node + climb the reuse->extend->new gap ladder; ALIGN both edges (the new dependency-spec to the producer output-spec, the new output-spec to the consumer dependency-spec) with a computed >=0.5 resolution score, editing the consumer/producer specs for room; derive container-path-constrained contract paths; author the .step.yaml + config + script (fixed main signature); then validate, preflight, and (for data-carrying steps) verify execution. Gates are non-skippable pipeline stages. Implements FZ 31e1d3f5a + the FZ 29 mods-migration SOP.',
  phases: [
    { title: 'Resolve', detail: 'DAG-in: locate the undefined node between producer+consumer; decompose node->step_type+_jobtype; climb the reuse->extend->new gap ladder; bind handler + exemplar' },
    { title: 'Challenge', detail: 'verdict-override on the gap rung: a skeptic runs catalog.step_info/steps.io to try to REVERSE the rung (esp. a wrong cheaper reuse/extend/delete that would silently short-circuit); reverses on the same enum' },
    { title: 'AlignEdges', detail: 'the spec-alignment gate: arity (1 PropertyReference per DependencySpec); GATE-1 enum (dependency_type/output_type exact-or-in-matrix, else the edge hard-zeros); edit CONSUMER compatible_sources + OUTPUT aliases for editing room; GATE-2 compute the 6-component score for BOTH edges, refuse <0.5' },
    { title: 'Guide', detail: 'per-axis field guidance + author.config_constraints (allowed_values + case + required-no-default) + the container-path roots per step type' },
    { title: 'Author', detail: 'agent Writes the .step.yaml + config class + script AND edits the producer/consumer .step.yaml specs so they accept the new step' },
    { title: 'Validate', detail: 'validate.step_interface (contract<->spec + container paths) + author.check_script (both directions incl. --job_type + the {job_type} subdir layout) + py_compile/yaml parse oracle + re-resolve both edges, bounded fix loops' },
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
  required: ['step_name', 'snake_name', 'sagemaker_step_type', 'step_assembly', 'node_type', 'framework', 'reuse_class', 'bound_handler', 'exemplar_step', 'needed_axes', 'gap_rung', 'is_new_step', 'producer', 'consumer'],
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
    // The adjacent nodes and their decomposition (SOP step 1). Empty producer => source; empty consumer => sink.
    producer: { type: 'object', additionalProperties: false, required: ['node', 'base_step_type'], properties: { node: { type: 'string' }, base_step_type: { type: 'string' }, job_type: { type: 'string' } } },
    consumer: { type: 'object', additionalProperties: false, required: ['node', 'base_step_type'], properties: { node: { type: 'string' }, base_step_type: { type: 'string' }, job_type: { type: 'string' } } },
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

const GUIDE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['axis', 'recommended', 'restrictions', 'exemplar_snippet'],
  properties: {
    axis: { type: 'string' },
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
    'If new_step: call author.checklist(sagemaker_step_type[, step_assembly]) for the ordered SOP + the bound handler +',
    'the exemplar; strategies.for_step_type to confirm the handler/knobs; author.rules("naming") + author.rules("reuse_class").',
    'needed_axes = only the behavior axes this step actually uses. Return the PLAN.',
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

function guidePrompt(plan, axis, align) {
  return [
    HOWTO, '',
    'TASK (SOP steps 7,10): gather field guidance for the "' + axis + '" axis of new ' + plan.sagemaker_step_type + ' step "' + plan.step_name + '".',
    'Bound handler: ' + plan.bound_handler + '. Exemplar to copy shape from: ' + plan.exemplar_step + '.',
    'The two edges are already aligned; honor that: ' + JSON.stringify((align && align.edges) || []) + '.',
    'Call author.rules for the relevant topic (naming/packaging/sdk_carveout/closure as applicable),',
    'strategies.knobs(axis, strategy) for legal knob values + defaults, and',
    'steps.io / steps.patterns / catalog.config_fields on the exemplar for the wired shape.',
    'For inputs/outputs: the dependency/output spec logical names + types + aliases + compatible_sources MUST match the',
    'AlignEdges decision above; the contract path for each is under the container roots — ' + CONTAINER_PATHS,
    'For the config-class field VALUES (esp. env_vars / compute), ALSO call author.config_constraints(' + plan.exemplar_step + ')',
    'to get each field allowed_values + case_sensitive + the required_no_default list — so you write a LEGAL enum value',
    '(e.g. output_format CSV/TSV/Parquet) and supply every required field, not a guessed one. Do NOT write any file.',
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
    'SPEC (from AlignEdges — do not drift): the NEW dependency-spec + output-spec types/logical-names/aliases/',
    'compatible_sources are FIXED by the edge alignment: ' + JSON.stringify((align && align.edges) || []),
    'CONTRACT (SOP step 7): ' + CONTAINER_PATHS + ' Layer {job_type}/... beneath the root; contract dict KEYS ==',
    'spec logical_names byte-for-byte. Optional inputs -> presence-test + required=False + listed in env_vars.optional.',
    'SCRIPT (SOP step 9): main() is indexed ONLY by contract logical_names — NEVER hardcode /opt/ml inside main().',
    'Copy the __main__ block + container-path constants VERBATIM from the exemplar (' + plan.exemplar_step + '); read every',
    'behavioral knob via environ_vars.get(NAME, default); read inputs with rglob (not a flat glob) so nested {job_type}/',
    'files are found. EMIT the FULL downstream-consumed artifact set even for artifacts this script does not itself use.',
    '',
    'PLAN: ' + JSON.stringify(plan),
    'PER-AXIS GUIDANCE: ' + JSON.stringify(guides),
    'Copy section shapes from exemplar ' + plan.exemplar_step + '. registry.sagemaker_step_type=' + plan.sagemaker_step_type + '.',
    'Do NOT edit any registry file (registry is derived by construction). Report the written file paths AND the exact',
    'producer/consumer spec edits you made.',
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

  // Stage 1 — Resolve (locate node + gap ladder) then Challenge the rung (verdict-override)
  async (req) => {
    const plan = await agent(resolvePrompt(req), { label: 'resolve:' + (req.name || req.intent.slice(0, 20)), phase: 'Resolve', schema: PLAN_SCHEMA, effort: 'high' })
    if (!plan) return null
    // Verdict-override: one skeptic tries to REVERSE the gap rung. A wrong CHEAPER rung
    // (reuse/extend/delete) is the dangerous case — it short-circuits with the need unmet — so pressure
    // it hardest. Reuses the exact gap_rung enum so the challenge authoritatively resets the rung.
    const ch = await agent(
      'TRY TO BREAK this gap-ladder routing decision for "' + (req.name || plan.step_name) + '" (edge ' + edgeStr(req) + '). ' +
      'The Resolve agent chose gap_rung="' + plan.gap_rung + '" (reason: ' + (plan.gap_rung_reason || 'n/a') + '). Run the ACTUAL ' +
      'checks to refute it: catalog.step_info / catalog.resolve_step / steps.io on the closest existing step types. Is there ' +
      'REALLY no existing step_type that does this config-only (reuse_config_only)? no backward-compatible OPTIONAL-dependency ' +
      'extension of an existing step (extend_optional_dep)? is the artifact already produced upstream (delete_node_artifact_exists)? ' +
      'Pressure the CHEAPER rungs HARDEST — a wrong cheaper rung silently short-circuits with the need unmet; only escalate to ' +
      'new_step if the cheaper rungs genuinely do not fit. Return {holds, challenges, final_rung (same enum), correction}.',
      { label: 'challenge:' + (req.name || plan.step_name), phase: 'Challenge', schema: CHALLENGE_SCHEMA, effort: 'high' })
    if (ch && ch.holds === false && ch.final_rung) {
      plan.gap_rung = ch.final_rung
      plan.is_new_step = (ch.final_rung === 'new_step')
      if (ch.correction) plan.gap_rung_reason = 'CHALLENGE-REVERSED: ' + ch.correction
    }
    return { req, plan, challenge: ch }
  },

  // Stage 2 — AlignEdges (the spec-alignment gate) — only for genuine new steps
  async (ctx) => {
    if (!ctx || !ctx.plan) return null
    if (!ctx.plan.is_new_step) return { ...ctx, align: null, short_circuit: 'gap_rung=' + ctx.plan.gap_rung + ': ' + (ctx.plan.gap_rung_reason || 'no new step needed') }
    const align = await agent(alignPrompt(ctx.plan, ctx.req), { label: 'align:' + ctx.plan.step_name, phase: 'AlignEdges', schema: ALIGN_SCHEMA, effort: 'high' })
    return { ...ctx, align, short_circuit: (align && !align.ready) ? 'edges do not resolve (>=0.5) even after the planned consumer edits — revisit the DAG or the step design' : null }
  },

  // Stage 3 — Guide (per-axis barrier) then Author (the agent's own Write) — skipped if short-circuited
  async (ctx) => {
    if (!ctx || !ctx.plan) return null
    if (ctx.short_circuit) return ctx
    const guides = (await parallel((ctx.plan.needed_axes || []).map(ax => () =>
      agent(guidePrompt(ctx.plan, ax, ctx.align), { label: 'guide:' + ctx.plan.step_name + ':' + ax, phase: 'Guide', schema: GUIDE_SCHEMA })
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
        'Call validate.step_interface(step_name="' + plan.step_name + '") (CLI fallback: `cursus validate step-interface ' + plan.step_name + ' --format json`). Return its {ok, errors, warnings}.',
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
        'Call author.check_script(step_name="' + plan.step_name + '"). Return {status, passed, issues}. ' +
        'status:"skipped" means a script-less / SDK-delegation step (treat as pass). Beyond the tool result, ALSO reason ' +
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

    // 4c. re-resolve BOTH edges with the REAL resolver now that all three specs exist (SOP step 6, for real).
    const nodes = [ctx.req.producer_node, ctx.plan.step_name, ctx.req.consumer_node].filter(Boolean)
    const rr = await agent(
      'The new step .step.yaml now exists, so the real resolver can score its edges. Call ' +
      'validate.deps_resolve(step_names=' + JSON.stringify(nodes) + ') and report, for each of the edges ' +
      edgeStr(ctx.req) + ', the resolution score and whether it resolves (>=0.5). A source/sink side that has no ' +
      'producer/consumer is not an edge. Return {both_edges_resolve, edges:[{edge,score,resolves,note}]}.',
      { label: 'resolve:' + plan.step_name, phase: 'Validate', schema: RESOLVE_SCHEMA, effort: 'high' })

    // 4d. constructibility proof (CI merge gate) + whole-DAG compile preview.
    const pf = await agent(
      'Call author.preflight_step(step_name="' + plan.step_name + '") under offline (PYTHONNOUSERSITE=1) semantics; then ' +
      'sanity-check the whole DAG the new step lives in via compile.preview / compile.validate (nodes ' +
      JSON.stringify(nodes) + ') so the new node resolves to its config + builder with high confidence. ' +
      'Return {constructible, gates:[{name,passed,detail}]}. SDK-delegation rows must be skip-not-error.',
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
    return { plan, req: ctx.req, align: ctx.align, validate: v, check_script: cs, parse: parse, resolve: rr, preflight: pf, gaps: gaps }
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
// Open gaps found by the completeness critic: edit collateral or a cross-branch column mismatch.
const hasOpenGaps = (r) => !!r.gaps && (((r.gaps.edit_collateral || []).length > 0) || ((r.gaps.cross_branch_column_mismatches || []).length > 0))
const passedCore = (r) =>
  !!(r.validate && r.validate.ok) && scriptOk(r) && parseOk(r) &&
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
  publish_path: 'A step is "green" only if validate.step_interface ok + author.check_script pass/skip + the py_compile/yaml PARSE oracle + both edges resolve + preflight constructible, AND the completeness critic found no open gaps (else green_with_open_gaps — flagged, not silently shipped). These are NECESSARY, not SUFFICIENT: they cover the single new step interface + its two edges + parse-ability. Before publish, ALSO run `cursus validate step-interface --all` (catches cross-step/registry-parity breakage the per-step gate cannot), and note that registry-parity / RegistryBindingValidator B3 / routability / brazil-build still run at CR. Habit 4: a green edge is a WIRE proof, not a runtime proof — for a data-carrying step run validate.run_scripts (or a sampled dry-run) and confirm the cross-branch column trace. Then: CR + brazil pb build → cursus-peru-dev → cursus-peru-shared → CDK → CodeArtifact.',
}
