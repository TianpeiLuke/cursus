export const meta = {
  name: 'cursus-configure-pipeline',
  description: 'Drive agent-tool-driven generation of a pipeline config.json for a cursus DAG — author/repair a generate_config.py (the DAGConfigFactory pattern: set_base_config -> set_step_config per node -> generate_all_configs -> merge_and_save_configs), gated per-step by author.config_constraints (legal values) + author.preflight_config (model_validate the values). Closes the common case (config VALUES for EXISTING step types) where the Munged Address migration found 19 of its 21 bugs. Companion to /cursus-author-step (which authors a NEW step type).',
  phases: [
    { title: 'Map', detail: 'load/build the DAG, map each node -> config class, read base + per-step field requirements' },
    { title: 'Constrain', detail: 'per-step: author.config_constraints — legal enum values + case + declared type + required-no-default' },
    { title: 'Author', detail: 'agent writes/repairs generate_config.py (base + set_step_config per node)' },
    { title: 'Validate', detail: 'author.preflight_config per step (model_validate the values) + a bounded fix loop' },
    { title: 'Generate', detail: 'run generate_config.py: get_pending_steps gate -> generate_all_configs -> merge_and_save_configs' },
    { title: 'DagCheck', detail: 'CROSS-NODE gate: node/edge integrity (every add_edge endpoint was add_node\'d — no phantom nodes from typos) + compile.validate (missing configs) + compile.preview (no low-confidence/ambiguous resolution = no wrong-config-class collision) + validate.deps_resolve (edges wire)' },
    { title: 'Synthesize', detail: 'report the config.json + per-step validity + the cross-node verdict + the SAIS run path' },
  ],
}

// ---------------------------------------------------------------------------
// INPUT (via `args`): { dag_nodes: ["StepName" | "StepName_jobtype", ...], project?: str,
//                       region?: str, project_root?: str, config_path?: str, generate_script?: str }
// dag_nodes is the list of DAG node names whose config must be authored (each maps to a step type
// via canonical-name extraction). With no args, a one-node illustrative request is used.
//
// PROJECT FOLDER LAYOUT (the BAMT/cursus convention — see munged_address_pytorch / neat_spam_pytorch):
// a project is a self-contained folder holding, AT ITS ROOT, generate_config.py (+ <project>_na.py,
// run_pipeline.py) alongside dockers/ (scripts/ + hyperparams/ + training/inference entry points) and
// pipeline_config/ (config_<REGION>.json + dag_*.json + exe_doc). The folder lives under either:
//   - cursus dev repo:     projects/<project>/
//   - BuyerAbuseModsTemplate (deployed): src/buyer_abuse_mods_template/<project>/
// Pass project_root explicitly to target either; it defaults to the cursus dev-repo location.
// generate_config.py writes pipeline_config/config_<REGION>.json (singular 'pipeline_config').
// ---------------------------------------------------------------------------
const INPUT = (args && typeof args === 'object' && !Array.isArray(args)) ? args
  : (args && Array.isArray(args) ? { dag_nodes: args } : {})
const DAG_NODES = (INPUT.dag_nodes && INPUT.dag_nodes.length) ? INPUT.dag_nodes : ['TabularPreprocessing']
const PROJECT = INPUT.project || '<project>'
const REGION = INPUT.region || 'NA'
// project_root is the project folder (the dir that holds generate_config.py + dockers/ + pipeline_config/).
const PROJECT_ROOT = INPUT.project_root || ('projects/' + PROJECT)
const GEN_SCRIPT = INPUT.generate_script || (PROJECT_ROOT + '/generate_config.py')
const CONFIG_PATH = INPUT.config_path || (PROJECT_ROOT + '/pipeline_config/config_' + REGION + '.json')

// Each DAG node name carries an optional _<job_type> / -<job_type> suffix; the step TYPE is the
// canonical prefix. The suffix may be lower- or capital-cased (CradleDataLoading_tagging,
// RiskTableMapping_Training, TabularPreprocessing-Calibration), so strip one trailing
// `[_-]<segment>` of either case. A name with no separator is already the bare step type.
function stepTypeOf(node) {
  const m = node.replace(/[_-][A-Za-z][A-Za-z0-9]*$/, '')
  return m || node
}

const HOWTO = [
  'cursus 2.0 (Design B) config.json generation. A pipeline config.json is produced by a',
  'generate_config.py that uses DAGConfigFactory: build/import the DAG -> DAGConfigFactory(dag) ->',
  'set_base_config(shared: bucket/role/region/author/service_name/pipeline_version/model_class/',
  'project_root_folder/source_dir/use_secure_pypi) -> set_base_processing_config(processing_source_dir',
  '+ instance types) -> set_step_config(node, **per-step values) for EACH node -> get_pending_steps()',
  'must be empty -> generate_all_configs() -> merge_and_save_configs(configs, config_path).',
  'set_step_config VALIDATES each config immediately (Pydantic), so wrong/invalid VALUES fail there.',
  'Tools (call as MCP; never hand-edit config.json blindly): author.config_constraints (legal values +',
  'declared TYPE + required-no-default per field), author.preflight_config (model_validate a value set),',
  'catalog.config_fields, config.requirements (factory field requirements), and for the cross-node gate',
  'compile.validate / compile.preview / validate.deps_resolve. The agent WRITES generate_config.py with Write.',
  '',
  'THE 3-STEP HABIT (this caught 19 of 21 real migration bugs — DO IT for every value you set):',
  '(1) READ THE VALIDATOR, NOT THE DOCSTRING: copy each enum/Literal value VERBATIM from the @field_validator',
  '    code (author.config_constraints) — docstrings + design notes + example JSON LIE about case-insensitivity',
  '    (output_format is case-sensitive CSV/TSV/Parquet; cluster_type is STANDARD/SMALL/MEDIUM/LARGE, NOT XLARGE;',
  '    allowed sets are PER config class — Cradle PARQUET != TabularPreprocessing Parquet). Match the declared',
  '    Python TYPE (a Dict field like bedrock_validation_schema takes a dict, NOT a JSON string).',
  '(2) TRACE config-value -> builder -> env var -> script-read -> downstream-column: a value is wrong if it is',
  '    the wrong runtime LAYER (config Dict vs env string vs CLI arg), or if a schema field name + prefix does not',
  '    equal the downstream reader\'s expected column (strangeness_rating + llm_ = llm_strangeness_rating).',
  '(3) PLACEHOLDER-IS-A-BUG: replace every sample/demo/example value (placeholder field lists, demo dates,',
  '    example ARNs, notebook-typo entry_points) with the real value from a named source (hyperparameters.json,',
  '    the source notebook SQL, EDX/manifest, model_metadata.json). "Simplified script doesn\'t use it" does NOT',
  '    make a builder-required field (id_name, label_name) optional — set every Field(...)-with-no-default.',
].join('\n')

const CONSTRAINTS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['step_name', 'config_class', 'required_no_default', 'constrained_fields'],
  properties: {
    step_name: { type: 'string' },
    config_class: { type: 'string' },
    required_no_default: { type: 'array', items: { type: 'string' } },
    // typed_fields: declared Python TYPE per field (catches Dict-vs-JSON-string bugs).
    typed_fields: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['name', 'type'], properties: { name: { type: 'string' }, type: { type: 'string' } } } },
    constrained_fields: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['name', 'allowed_values'], properties: { name: { type: 'string' }, allowed_values: { type: 'array', items: {} }, case_sensitive: { type: 'boolean' } } } },
  },
}

const PRECONFIG_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['valid', 'errors'],
  properties: {
    valid: { type: 'boolean' },
    errors: { type: 'array', items: { type: 'object', additionalProperties: true, required: ['field', 'message'], properties: { field: { type: 'string' }, message: { type: 'string' }, type: { type: 'string' } } } },
  },
}

const GENERATE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['generated', 'config_path', 'config_json_parses'],
  properties: {
    generated: { type: 'boolean' },
    config_path: { type: 'string' },
    // config_json_parses: an EXECUTED check, not a self-report — the agent must run
    // `python3 -c "import json; json.load(open(CONFIG_PATH))"` and set this from its exit code.
    // Turns "plausibly generated" into "the file is on disk and parses as JSON".
    config_json_parses: { type: 'boolean' },
    config_count: { type: 'integer' },
    pending: { type: 'array', items: { type: 'string' } },
    error: { type: 'string' },
  },
}

// The Author stage now returns a py_compile result (executable parse oracle) instead of nothing:
// a generate_config.py that does not even compile can no longer pass silently to Validate/Generate.
const AUTHOR_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['script_path', 'py_compile_ok'],
  properties: {
    script_path: { type: 'string' },
    py_compile_ok: { type: 'boolean', description: 'set from the exit code of `python3 -m py_compile <GEN_SCRIPT>` — the agent MUST run it' },
    notes: { type: 'string' },
  },
}

// Fix agents now report which file they touched + whether it still compiles, so a fixer that injects
// a syntax error escalates (gate/contract bug) instead of silently burning the next retry.
const FIX_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['file_edited', 'py_compile_ok'],
  properties: {
    file_edited: { type: 'string' },
    change_summary: { type: 'string' },
    py_compile_ok: { type: 'boolean', description: 'run `python3 -m py_compile <file_edited>` after editing; set from its exit code' },
  },
}

// The cross-node gate result: did the whole DAG resolve against the generated config, with no
// missing config, no low-confidence/ambiguous (wrong-config-class) resolution, and edges that wire?
// undeclared_edge_nodes: add_edge endpoints NOT declared via add_node — add_edge silently auto-creates
// them, so a single typo'd edge name spawns a phantom unconfigured node and orphans the real one.
const DAGCHECK_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['valid', 'issues'],
  properties: {
    valid: { type: 'boolean' },
    undeclared_edge_nodes: { type: 'array', items: { type: 'string' } },
    missing_configs: { type: 'array', items: { type: 'string' } },
    low_confidence_nodes: { type: 'array', items: { type: 'string' } },
    unresolved_edges: { type: 'array', items: { type: 'string' } },
    issues: { type: 'array', items: { type: 'string' } },
  },
}

function constraintsPrompt(node, stepType) {
  return [
    HOWTO, '',
    'TASK: read the config CONSTRAINTS for DAG node "' + node + '" (step type ' + stepType + '). Do NOT write anything.',
    'Call author.config_constraints(step_name="' + stepType + '") and return its config_class,',
    'required_no_default (the Tier-1 fields a VALUE must be supplied for), constrained_fields',
    '(each field\'s allowed_values + case_sensitive — copy VERBATIM from the validator), and typed_fields',
    '(each field\'s declared Python type from "type" — note any Dict/List field so you pass a dict/list, NOT a',
    'JSON string). This is what the per-step config block must satisfy.',
  ].join('\n')
}

function authorPrompt(plan) {
  return [
    HOWTO, '',
    'TASK: author (or repair) the config-generation script ' + GEN_SCRIPT + ' with your Write/Edit tool.',
    'PROJECT LAYOUT (relative to the project root ' + PROJECT_ROOT + ' — the SAME structure in every project,',
    'whether under the dev-repo projects/ or BuyerAbuseModsTemplate src/buyer_abuse_mods_template/):',
    '  ' + PROJECT_ROOT + '/generate_config.py             <- the script you write (project root)',
    '  ' + PROJECT_ROOT + '/<project>_<region>.py          <- the DAG def (lowercase region, e.g. *_na.py);',
    '                                                         create_<project>_dag() -> ONE worldwide (WW) DAG',
    '  ' + PROJECT_ROOT + '/dockers/scripts/               <- application-agnostic Processing scripts (processing_source_dir)',
    '  ' + PROJECT_ROOT + '/dockers/<framework>_training.py, _model_inference.py, _model_eval.py, _inference_handler.py',
    '                                                       <- model-dependent scripts live in the dockers ROOT (source_dir)',
    '  ' + PROJECT_ROOT + '/dockers/hyperparams/hyperparameters_' + REGION + '.json   <- per-region training hyperparameters',
    '  ' + PROJECT_ROOT + '/pipeline_config/config_' + REGION + '.json   <- the output (SINGULAR pipeline_config, per-region)',
    'KEY: the DAG TOPOLOGY is worldwide (one create_<project>_dag); only config_<REGION>.json +',
    'hyperparameters_<REGION>.json + exe_doc are per-region (regional SQL / marketplace IDs / ARNs).',
    'The script must, in order: build/import the WW DAG (nodes: ' + JSON.stringify(DAG_NODES) + '); DAGConfigFactory(dag);',
    'set_base_config(... project_root_folder="<project>", source_dir="dockers", region=REGION, use_secure_pypi=True, ...);',
    'set_base_processing_config(processing_source_dir="dockers/scripts", instance types);',
    'then ONE factory.set_step_config(node, **values) per node below, supplying every required_no_default',
    'field and only LEGAL enum values (per the constraints gathered):',
    JSON.stringify(plan, null, 1),
    '',
    'CRADLE / SQL rules (the Cradle nodes caused the most run-time failures):',
    '- transform_sql is a SINGLE SELECT/CTE — never CREATE TEMPORARY VIEW or any DDL (Cradle wraps it in a subquery).',
    '- use EXACT lowercase Andes column names + native int types (Spark caseSensitive); no "0"/"1" string literals',
    '  against int columns; strip the Redshift d_ prefix + schema qualifier and use the real Andes provider+table.',
    '- map notebook MDSDataLoader params to the config (they are NOT flat fields): auto_tag->ANDES LEFT JOIN,',
    '  tag_file->EDX INNER JOIN, filter_conditions->WHERE, customized_sql->transform_sql, no_split/days_per_split',
    '  ->JobSplitOptions, merge_sql "SELECT DISTINCT <schema> FROM INPUT" when split_job=true.',
    '- DATE RANGE / SIZING: derive start/end from model_metadata.json offset_days (never a hardcoded multi-year',
    '  window); ceil((end-start)/days_per_split) MUST be <=30 (Cradle hard limit; >5 is a yellow flag) and must',
    '  overlap the EDX/tag manifest vintage and yield enough positives for a stratified split. org_id by region',
    '  (NA=1/EU=2/FE=9), service_name verbatim (e.g. FORTRESS_RETAIL). Use the NESTED Cradle config objects',
    '  (cradle_job_spec / data_sources_spec / output_spec / transform_spec), not flat kwargs. Choose EdxUploadingConfig',
    '  (not DataUploadingConfig) when the notebook uploads to EDX.',
    '',
    'CROSS-NODE contract checks — answer per producer->consumer edge BEFORE writing the values:',
    '- job_type chain: the upstream output subdir == the downstream input lookup (both {job_type}/...).',
    '- column contract: a producer schema-field-name + output prefix == the consumer\'s expected column',
    '  (e.g. strangeness_rating + llm_ = llm_strangeness_rating that STRANGENESS_COLUMN reads), and the declared',
    '  schema dtype is numeric where a downstream "> threshold" consumes it.',
    '- id_name names the entity THIS model scores (order/address/customer) and exists in the data; label_name is',
    '  a documented placeholder if the (calibration) data is unlabeled. Payload FIELD_DEFAULTS overrides when the',
    '  API field (saddr) != the model-internal text_name (shippingAddress). Registration output-var name ==',
    '  the inference handler\'s output field, byte-for-byte.',
    '- DAG-node name == the metadata.config_types key EXACTLY (a rename without re-running set_step_config makes',
    '  the resolver fuzzy-match the WRONG config class -> a duplicate-step ValueError at execution).',
    '',
    'End with: assert not factory.get_pending_steps(); configs = factory.generate_all_configs();',
    'merge_and_save_configs(configs, "' + CONFIG_PATH + '"). Copy the base-config + per-step shape from an',
    'existing project generate_config.py (e.g. projects/munged_address_pytorch/generate_config.py or',
    'neat_spam_pytorch) — but treat every copied value as a placeholder to verify (Step-3 habit), since that is',
    'exactly how XLARGE / lowercase-parquet / notebook-typo entry_points propagate.',
    '',
    'FINALLY (executable self-check): after writing, RUN `python3 -m py_compile ' + GEN_SCRIPT + '` and iterate until it ' +
    'exits 0. Return {script_path, py_compile_ok (from that exit code — do NOT claim true without running it), notes}.',
  ].join('\n')
}

// ---------------------------------------------------------------------------
// PIPELINE: per node, gather constraints (Constrain) -> the values are authored together in ONE
// generate_config.py (Author, after the Constrain barrier) -> per-step preflight_config (Validate)
// -> run the script (Generate). The factory's own set_step_config + get_pending_steps are the
// final Pydantic gate; preflight_config is the per-step early gate.
// ---------------------------------------------------------------------------

// Constrain: one agent per node (barrier — Author needs all constraints to write one script).
phase('Constrain')
const constraints = (await parallel(DAG_NODES.map(node => () =>
  agent(constraintsPrompt(node, stepTypeOf(node)), { label: 'constrain:' + node, phase: 'Constrain', schema: CONSTRAINTS_SCHEMA })
    .then(c => (c ? { node, step_type: stepTypeOf(node), ...c } : null))
))).filter(Boolean)
log('gathered constraints for ' + constraints.length + '/' + DAG_NODES.length + ' nodes')

// Author: the agent writes the whole generate_config.py and self-checks it compiles.
phase('Author')
const authored = await agent(authorPrompt(constraints), { label: 'author:generate_config', phase: 'Author', schema: AUTHOR_SCHEMA, effort: 'high' })
if (authored && authored.py_compile_ok === false) {
  log('WARNING: generate_config.py does not py_compile after Author — Validate/Generate will surface the error')
}

// Validate: per-step value gate (preflight_config) with a bounded fix loop on the script.
phase('Validate')
const validations = await pipeline(constraints,
  async (c) => {
    let pc, tries = 0
    do {
      pc = await agent(
        'Read the per-step config VALUES your generate_config.py passes to factory.set_step_config for node "' + c.node + '" ' +
        '(step type ' + c.step_type + '), then call author.preflight_config(step_name="' + c.step_type + '", values=<those values incl. inherited base fields>). ' +
        'Return {valid, errors:[{field,message,type}]}.',
        { label: 'preflight_config:' + c.node, phase: 'Validate', schema: PRECONFIG_SCHEMA })
      if (pc && !pc.valid && tries < 2) {
        const fx = await agent(
          'author.preflight_config failed for node "' + c.node + '" with errors: ' + JSON.stringify(pc.errors || []) +
          '. Fix the set_step_config(...) values for this node in ' + GEN_SCRIPT + ': for a wrong-enum/case error use the ' +
          'allowed_values from author.config_constraints("' + c.step_type + '"); for a missing-field error supply the required ' +
          'field. Edit ONLY this node\'s block. FINALLY run `python3 -m py_compile ' + GEN_SCRIPT + '` and return ' +
          '{file_edited, change_summary, py_compile_ok (from the exit code)}.',
          { label: 'fix_config:' + c.node + ':' + tries, phase: 'Validate', schema: FIX_SCHEMA, effort: 'high' })
        // A fixer that edited but left the script un-compilable is a gate/contract problem, not a value
        // problem — flag it rather than burning the next silent retry on a broken file.
        if (fx && fx.py_compile_ok === false) log('WARNING: fix_config:' + c.node + ' left ' + GEN_SCRIPT + ' not compiling — escalate (gate/contract bug)')
      }
      tries++
    } while (pc && !pc.valid && tries <= 2)
    return { node: c.node, valid: pc ? pc.valid : null, errors: pc ? pc.errors : [] }
  },
)

// Generate: run the script end-to-end (the factory's get_pending_steps + generate_all_configs is the
// final whole-config Pydantic gate).
phase('Generate')
const gen = await agent(
  'Run the config-generation script: `python ' + GEN_SCRIPT + '`' + (INPUT.region ? ' (region ' + REGION + ')' : '') + '. ' +
  'It must reach get_pending_steps() empty, generate_all_configs(), and merge_and_save_configs to "' + CONFIG_PATH + '". ' +
  'THEN confirm the output is really on disk and well-formed: run ' +
  '`python3 -c "import json; d=json.load(open(\'' + CONFIG_PATH + '\')); print(len(d))"`; it MUST exit 0 (a soft check: the ' +
  'printed length should be >= the ' + DAG_NODES.length + ' DAG nodes — some steps emit multiple configs, so do not hard-fail on the exact count). ' +
  'Return {generated, config_json_parses (from that json.load exit code — set false if it does not parse or the file is absent), ' +
  'config_path, config_count, pending, error}. If get_pending_steps is non-empty or any config fails Pydantic validation, ' +
  'generated=false and report the pending nodes / error.',
  { label: 'generate:config', phase: 'Generate', schema: GENERATE_SCHEMA, effort: 'high' })

// DagCheck: the CROSS-NODE gate the per-node Validate phase cannot reach. Per the migration
// learnings (FZ 29), most SAIS-run failures are cross-node: a node renamed without regenerating its
// config silently fuzzy-resolves to the WRONG config class (duplicate-step ValueError); a missing
// config; or an edge that does not wire. compile.validate + compile.preview + validate.deps_resolve
// catch these OFFLINE, before run_pipeline. (Only meaningful if the config generated.)
let dagcheck = null
if (gen && gen.generated && gen.config_json_parses !== false) {
  phase('DagCheck')
  dagcheck = await agent(
    'The pipeline config.json is now at "' + CONFIG_PATH + '". Run the CROSS-NODE validation gate against the WW DAG ' +
    '(declared nodes: ' + JSON.stringify(DAG_NODES) + ', defined in ' + PROJECT_ROOT + '/<project>_<region>.py):\n' +
    '0. NODE/EDGE INTEGRITY (do this FIRST): PipelineDAG.add_edge SILENTLY auto-creates any endpoint that was not ' +
    'already add_node\'d (see api/dag/base_dag.py), so a single typo in an edge name ' +
    '(add_edge("A","TabularPreprocessing_traning")) spawns a PHANTOM unconfigured node AND orphans the real one — ' +
    'construction never raises and the serializer\'s dangling-edge check is fooled (the phantom is already in nodes). ' +
    'Call dag.validate_integrity(nodes=the add_node\'d names, edges=the add_edge pairs) and read its ' +
    'issues.undeclared_edge_nodes (the framework now reports exactly this via PipelineDAG.validate_node_declarations). ' +
    'CRITICAL: pass `nodes` = ONLY the names explicitly add_node\'d / in the nodes=[...] arg of the DAG source — do NOT ' +
    'union in edge endpoints, or you mask the typo. As a backstop, also read the DAG source and diff the set of ' +
    'add_node names against every add_edge endpoint. Report undeclared_edge_nodes; it MUST be empty — any member is a ' +
    'typo or a forgotten add_node and is a BUG.\n' +
    '1. compile.validate(dag, config_path) -> is_valid + missing_configs (every node resolves to a config + builder).\n' +
    '2. compile.preview(dag, config_path) -> per-node config_type + confidence; FLAG any node with LOW confidence or an ' +
    'ambiguous resolution — that is the duplicate-step / wrong-config-class collision (a node whose name does not EXACTLY ' +
    'match its metadata.config_types key fuzzy-resolves to the wrong class).\n' +
    '3. validate.deps_resolve(step_names=the DAG nodes) -> every required dependency matches an upstream output ' +
    '(compatible_sources + aliases wire); FLAG unresolved edges.\n' +
    'Return {valid (all four clean), undeclared_edge_nodes, missing_configs, low_confidence_nodes, unresolved_edges, issues}. ' +
    'valid=false if undeclared_edge_nodes is non-empty OR anything is missing/low-confidence/unresolved — those WILL ' +
    'fail in SAIS (or silently run the wrong graph) even though the config wrote fine.',
    { label: 'dagcheck', phase: 'DagCheck', schema: DAGCHECK_SCHEMA, effort: 'high' })
}

// Synthesize.
// A skipped/dead skeptic must NOT read as green. Distinguish "the cross-node gate RAN" from "it PASSED":
// a null dagcheck (agent died, or generation failed so it never ran) is neither. And a per-step preflight
// that returned null (v.valid===null) is not a pass either — use `!== true`, not `=== false`.
const invalid = validations.filter(v => v && v.valid !== true)
const crossNodeRan = !!dagcheck
const crossNodeOk = !!dagcheck && dagcheck.valid === true
const configParses = !gen || gen.config_json_parses !== false
const shippable = !!(gen && gen.generated && configParses && crossNodeRan && crossNodeOk && invalid.length === 0)
log('config gen: ' + (gen && gen.generated ? 'OK (' + (gen.config_count || '?') + ' configs)' : 'INCOMPLETE') +
  '; ' + invalid.length + ' node(s) with value errors' +
  (crossNodeRan ? '; cross-node ' + (crossNodeOk ? 'OK' : 'FAILED') : (gen && gen.generated ? '; cross-node DID NOT RUN' : '')))
return {
  config_path: CONFIG_PATH,
  generate_script: GEN_SCRIPT,
  nodes: DAG_NODES.length,
  constrained: constraints.length,
  per_step: validations.map(v => ({ node: v.node, values_valid: v.valid, errors: (v.errors || []).map(e => e.field) })),
  generated: gen ? gen.generated : null,
  config_json_parses: gen ? gen.config_json_parses : null,
  config_count: gen ? gen.config_count : null,
  pending: gen ? gen.pending : null,
  author_py_compile_ok: authored ? authored.py_compile_ok : null,
  value_errors: invalid.length,
  cross_node_ran: crossNodeRan,
  cross_node: dagcheck ? {
    valid: dagcheck.valid,
    undeclared_edge_nodes: dagcheck.undeclared_edge_nodes || [],
    missing_configs: dagcheck.missing_configs || [],
    low_confidence_nodes: dagcheck.low_confidence_nodes || [],
    unresolved_edges: dagcheck.unresolved_edges || [],
    issues: dagcheck.issues || [],
  } : null,
  shippable: shippable,
  next: shippable
    ? ('config.json ready at ' + CONFIG_PATH + ' — run in SAIS via run_pipeline.py: ' +
       '`python ' + PROJECT_ROOT + '/run_pipeline.py --preview` (offline DAG-resolution + validate_dag_compatibility), ' +
       'then `--region ' + REGION + '` (setup_session -> PipelineDAGCompiler.compile_with_report -> ' +
       'ExecutionDocumentGenerator.fill_execution_document -> SagemakerPipelineHelper.start_pipeline_execution).')
    : (gen && gen.generated && configParses && crossNodeRan && !crossNodeOk
      ? 'config wrote but the CROSS-NODE gate FAILED — fix undeclared_edge_nodes (typo / missing add_node) / ' +
        'missing_configs / low_confidence_nodes / unresolved_edges above (these pass per-node validation but WILL ' +
        'fail in SAIS or silently run the wrong graph) and re-run before launching run_pipeline.py.'
      : (gen && gen.generated && configParses && !crossNodeRan
        ? 'config wrote but the CROSS-NODE gate DID NOT RUN (DagCheck returned null) — a skipped skeptic is not a pass; ' +
          're-run DagCheck before launching run_pipeline.py.'
        : (invalid.length > 0
          ? 'config generation reported OK but ' + invalid.length + ' node(s) failed / never cleared per-step value ' +
            'validation — resolve those before trusting the config.'
          : 'config incomplete — resolve the pending nodes / value errors above and re-run.'))),
}
