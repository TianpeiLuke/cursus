export const meta = {
  name: 'cursus-init-project',
  description: 'Scaffold a NEW Cursus-based ML project (phase-0): the fixed region-agnostic run_pipeline.py, the @MODSTemplate class <name>_<framework>_pipeline.py (loads pipeline_config/dag.json, compiles via PipelineDAGCompiler), a shared generate_config.py skeleton with project_root_folder filled + a TODO per-node value-init block, an empty dag.json stub, the folder tree + per-folder READMEs, and a root README ACTION-ITEM LEDGER handing every context-dependent piece (author the DAG, copy scripts/handlers, fill MODS metadata, fill config via /cursus-configure-pipeline) to its owning downstream workflow. Generates only what is knowable at t=0; grounds the fixed templates against a real reference project (skip-rather-than-invent). Companion/precursor to /cursus-configure-pipeline; headless twin of the Harmony new-project scaffold wizard.',
  phases: [
    { title: 'Consult', detail: 'resolve name+framework+target -> PLAN (full name, PascalCase class, import prefix, handler filenames, reference project)' },
    { title: 'Scaffold', detail: 'source-grounded generation of the fixed templates from the reference project; skip-rather-than-invent; project_root_folder MANDATORY' },
    { title: 'Ledger', detail: 'persisted completeness critic — the root README action-item ledger, naming the downstream workflow that owns each remaining step' },
    { title: 'Verify', detail: 'executable gate: py_compile the 3 .py + json.load the dag.json stub + confirm the tree + all READMEs' },
  ],
}

// ---------------------------------------------------------------------------
// INPUT (via `args`): { name: str, framework: str, target_dir?: str }
//   name       — project base name (snake_case), e.g. "secure_delivery"
//   framework  — one of xgboost | pytorch | lightgbmmt | bedrock
//   target_dir — where the <name>_<framework>/ folder is created. Defaults to the
//                AmazonCursus dev location "projects"; for a BAMT deploy pass
//                "src/buyer_abuse_mods_template".
// The full project name is `${name}_${framework}` (framework is the suffix).
//
// The two Python entry files split into a fixed skeleton + a per-project value layer:
//   run_pipeline.py                 — ~100% fixed (load dag.json -> SAIS session ->
//                                     PipelineDAGCompiler.compile_with_report ->
//                                     ExecutionDocumentGenerator -> start_pipeline_execution)
//   <project>_pipeline.py           — the @MODSTemplate class; fixed skeleton, only the
//                                     import prefix + AUTHOR/PIPELINE_* consts + class name vary;
//                                     LOADS dag.json (never inline)
//   generate_config.py              — fixed DAGConfigFactory skeleton; only project_root_folder
//                                     (filled here) + the per-node set_step_config value-init
//                                     (left TODO for /cursus-configure-pipeline) vary
// Everything context-dependent (the DAG, the scripts, the handlers, the config values) is NOT
// generated — it is recorded as a numbered, owner-assigned checklist in the root README ledger.
// ---------------------------------------------------------------------------

const INPUT = (args && typeof args === 'object' && !Array.isArray(args)) ? args : {}
const NAME = INPUT.name || 'demo_project'
const FRAMEWORK = (INPUT.framework || 'xgboost').toLowerCase()
const TARGET_DIR = INPUT.target_dir || 'projects'
const PROJECT = `${NAME}_${FRAMEWORK}`
const TARGET_PATH = `${TARGET_DIR}/${PROJECT}`

// Per-framework matrix (config-as-code — embedded, not via args): handler filenames + the
// hyperparameter class name + a KNOWN reference project to ground the fixed templates against.
const FRAMEWORKS = {
  xgboost:    { training: 'xgboost_training.py',   inference: 'xgboost_inference_handler.py',   hp_class: 'XGBoostModelHyperparameters',   reference: 'projects/atoz_xgboost' },
  pytorch:    { training: 'pytorch_training.py',   inference: 'pytorch_inference_handler.py',   hp_class: 'ModelHyperparameters',          reference: 'projects/munged_address_pytorch' },
  lightgbmmt: { training: 'lightgbm_training.py',  inference: 'lightgbm_inference_handler.py',  hp_class: 'LightGBMModelHyperparameters',  reference: 'projects/cap_mtgbm' },
  bedrock:    { training: 'bedrock_train.py',      inference: 'bedrock_inference_handler.py',   hp_class: 'BedrockModelHyperparameters',   reference: 'projects/rnr_pytorch_bedrock' },
}
const FW = FRAMEWORKS[FRAMEWORK] || FRAMEWORKS.xgboost

const PLAN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['project', 'pascal_class', 'import_prefix', 'training_entry', 'inference_handler', 'reference_project', 'target_path'],
  properties: {
    project: { type: 'string' },
    pascal_class: { type: 'string', description: 'PascalCase(name) + "Pipeline"' },
    import_prefix: { type: 'string', description: '"cursus" for an AmazonCursus dev target, "buyer_abuse_mods_template.cursus" for a BAMT target' },
    training_entry: { type: 'string' },
    inference_handler: { type: 'string' },
    reference_project: { type: 'string', description: 'a real project dir to ground the fixed templates against' },
    target_path: { type: 'string' },
  },
}

const SCAFFOLD_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['files_written', 'skipped', 'notes'],
  properties: {
    files_written: { type: 'array', items: { type: 'string' } },
    skipped: { type: 'array', items: { type: 'string' }, description: 'anything that could not be grounded in the reference project + why' },
    notes: { type: 'string' },
  },
}

const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['py_compile_ok', 'json_ok', 'tree_ok', 'files'],
  properties: {
    py_compile_ok: { type: 'boolean' },
    json_ok: { type: 'boolean' },
    tree_ok: { type: 'boolean' },
    files: { type: 'array', items: { type: 'string' } },
  },
}

// The verbatim contract injected into the Scaffold agent (source-grounded generation).
const SCAFFOLD_CONTRACT = [
  '# CURSUS PROJECT SCAFFOLD CONTRACT (phase-0 — generate only what is knowable now)',
  '',
  'GROUND every fixed template against the REAL reference project <REFERENCE>/ (read its',
  'run_pipeline.py if present + its *_pipeline.py / *_<region>.py @MODSTemplate file). Adapt the',
  'import prefix + class name + constants to the PLAN. Do NOT invent an API that is not in the',
  'reference; if a construct cannot be grounded, SKIP it and note why (never emit a broken stub).',
  '',
  'Write, with the Write tool, into <TARGET>/ :',
  '1. run_pipeline.py — the FIXED region-agnostic-DAG template: load pipeline_config/dag.json via',
  '   `<PREFIX>.api.dag import_dag_from_json` -> setup_session (SaisSession / SecurityConfig /',
  '   create_secure_session_config / PipelineSession) -> PipelineDAGCompiler.compile_with_report(dag)',
  '   -> ExecutionDocumentGenerator.fill_execution_document -> SagemakerPipelineHelper.',
  '   start_pipeline_execution. A --preview branch does offline preview_resolution +',
  '   validate_dag_compatibility (no session). argparse: --config --dag --preview --save-exe-doc.',
  '2. <PROJECT>_pipeline.py — the @MODSTemplate(author=AUTHOR, description=PIPELINE_DESCRIPTION,',
  '   version=PIPELINE_VERSION) class <CLASS>: a create_pipeline_dag() that LOADS',
  '   pipeline_config/dag.json (NEVER an inline DAG), an __init__(sagemaker_session, execution_role,',
  '   regional_alias, pipeline_parameters) that builds config_<region>.json + PipelineDAGCompiler,',
  '   generate_pipeline() -> compile_with_report, and validate_dag_compatibility()/preview_resolution()',
  '   helpers. AUTHOR / PIPELINE_DESCRIPTION / DEFAULT_SERVICE_NAME as TODO literals.',
  '3. generate_config.py — the SHARED DAGConfigFactory skeleton: a SAIS-base fallback ->',
  '   import_dag_from_json -> DAGConfigFactory(dag) -> set_base_config(project_root_folder="<PROJECT>",',
  '   source_dir="dockers", framework hint from <FRAMEWORK>, ...) -> set_base_processing_config ->',
  '   a TODO per-node set_step_config block guarded by get_pending_steps() -> generate_all_configs',
  '   -> merge_and_save_configs. project_root_folder="<PROJECT>" is MANDATORY and load-bearing — it',
  '   MUST be the literal project name, not a placeholder.',
  '4. pipeline_config/dag.json — the empty stub exactly: {"dag": {"nodes": [], "edges": []}}',
  '5. __init__.py (empty) at the project root AND at dockers/.',
  '6. A README.md in EACH of pipeline_config/, dockers/, dockers/scripts/, dockers/processing/,',
  '   dockers/hyperparams/ stating that folder\'s PURPOSE + what to put in it: pipeline_config holds',
  '   config_<region>.json + dag.json + exe_doc; dockers is the source_dir root (framework',
  '   training/inference handlers); dockers/scripts is the processing_source_dir (application-agnostic',
  '   Processing scripts, one per DAG node); dockers/processing is domain code you write; dockers/',
  '   hyperparams holds hyperparameters_<region>.json (from the feature lists + framework defaults).',
].join('\n')

// --- Consult -------------------------------------------------------------
phase('Consult')
const plan = await agent(
  [
    'Resolve the parameters for a new Cursus project. Do NOT write any file yet.',
    `name="${NAME}", framework="${FRAMEWORK}", target_dir="${TARGET_DIR}".`,
    `Compute: project="${PROJECT}"; pascal_class = PascalCase(name) + "Pipeline";`,
    'import_prefix = "buyer_abuse_mods_template.cursus" if target_dir contains "buyer_abuse_mods_template", else "cursus";',
    `training_entry="${FW.training}"; inference_handler="${FW.inference}";`,
    `reference_project="${FW.reference}" (a real project dir to ground the fixed templates against);`,
    `target_path="${TARGET_PATH}". Return the PLAN.`,
  ].join('\n'),
  { label: 'consult', phase: 'Consult', schema: PLAN_SCHEMA, effort: 'high' }
)
if (!plan) {
  log('Consult failed — cannot resolve the project plan')
  return { project: PROJECT, target_path: TARGET_PATH, ready: false, stopped_at: 'Consult' }
}

// --- Scaffold (source-grounded generation) -------------------------------
phase('Scaffold')
const scaffold = await agent(
  `Scaffold the phase-0 project ${plan.project} at ${plan.target_path}.\n\n` +
  SCAFFOLD_CONTRACT
    .replace(/<TARGET>/g, plan.target_path)
    .replace(/<PROJECT>/g, plan.project)
    .replace(/<CLASS>/g, plan.pascal_class)
    .replace(/<PREFIX>/g, plan.import_prefix)
    .replace(/<FRAMEWORK>/g, FRAMEWORK)
    .replace(/<REFERENCE>/g, plan.reference_project) +
  `\n\nReference project to ground against: ${plan.reference_project}/. Return {files_written, skipped, notes}.`,
  { label: 'scaffold', phase: 'Scaffold', schema: SCAFFOLD_SCHEMA, effort: 'high' }
)

// --- Ledger (persisted completeness critic) ------------------------------
phase('Ledger')
await agent(
  [
    `Write ${plan.target_path}/README.md as an ACTION-ITEM LEDGER for ${plan.project}.`,
    'A numbered, ORDERED checklist of the REMAINING steps, each naming the downstream workflow that owns it:',
    '  1. Author the DAG -> write pipeline_config/dag.json (nodes + edges). For a NEW step type not in the registry, run /cursus-author-step.',
    '  2. Copy per-node Processing scripts into dockers/scripts/ — one per DAG node, from cursus/steps/scripts/.',
    `  3. Copy the framework handlers into dockers/ — ${plan.training_entry} + ${plan.inference_handler} — from a reference project, then adapt.`,
    '  4. Fill hyperparameters -> dockers/hyperparams/hyperparameters_<region>.json (from the full/cat/tab field lists + label + id + framework defaults).',
    `  5. Fill the @MODSTemplate metadata — AUTHOR / PIPELINE_DESCRIPTION / DEFAULT_SERVICE_NAME in ${plan.project}_pipeline.py.`,
    '  6. Fill the generate_config.py TODO value-init (+ the base service_name/framework_version). Run /cursus-configure-pipeline to author + gate it.',
    '  7. Generate config -> `python generate_config.py --region <R>` -> pipeline_config/config_<R>.json.',
    '  8. Preview then run -> `python run_pipeline.py --preview`, then `--region <R>`.',
    'Then a "What is already done" section listing the scaffolded files (run_pipeline.py, the @MODSTemplate class, generate_config.py skeleton with project_root_folder filled, the empty dag.json, per-folder READMEs).',
    scaffold && scaffold.skipped && scaffold.skipped.length
      ? 'Also add a "Scaffold skips (could not be grounded — review)" section listing: ' + JSON.stringify(scaffold.skipped)
      : '',
    'Preserve exact Markdown; do not touch other files.',
  ].filter(Boolean).join('\n'),
  { label: 'ledger', phase: 'Ledger', effort: 'high' }
)

// --- Verify (executable gate) --------------------------------------------
phase('Verify')
const verify = await agent(
  [
    `Verify the scaffold at ${plan.target_path}. RUN these and report the TRUTHFUL booleans:`,
    `  cd ${plan.target_path} && python3 -m py_compile run_pipeline.py ${plan.project}_pipeline.py generate_config.py`,
    `  python3 -c "import json; json.load(open('pipeline_config/dag.json'))"`,
    'Confirm the folder tree exists (pipeline_config/, dockers/, dockers/scripts/, dockers/processing/, dockers/hyperparams/),',
    'all 5 folder READMEs + the root README.md exist, and pipeline_config/dag.json is present.',
    'Return {py_compile_ok, json_ok, tree_ok, files[]} — py_compile_ok=true ONLY if all three .py compile.',
  ].join('\n'),
  { label: 'verify', phase: 'Verify', schema: VERIFY_SCHEMA, effort: 'high' }
)

const ready = !!(verify && verify.py_compile_ok && verify.json_ok && verify.tree_ok)
log(`scaffolded ${plan.project}: py_compile=${verify ? verify.py_compile_ok : null} json=${verify ? verify.json_ok : null} tree=${verify ? verify.tree_ok : null} — ${ready ? 'READY' : 'INCOMPLETE'}`)

return {
  project: plan.project,
  target_path: plan.target_path,
  ready,
  files_written: scaffold ? (scaffold.files_written || []) : [],
  skipped: scaffold ? (scaffold.skipped || []) : [],
  next: ready
    ? 'author pipeline_config/dag.json -> /cursus-configure-pipeline -> python generate_config.py --region <R> -> python run_pipeline.py --preview'
    : 'scaffold did not reach ready; read files_written / skipped and the Verify booleans',
}
