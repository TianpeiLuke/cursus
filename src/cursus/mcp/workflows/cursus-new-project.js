export const meta = {
  name: 'cursus-new-project',
  description: 'Bring up a new Cursus project END-TO-END by composing the shipped workflows: Scaffold (cursus-init-project — the phase-0 skeleton + ledger) -> SeedDAG (catalog: pipeline_catalog.recommend + load_dag -> pipeline_config/dag.json; manual: STOP with the ledger for a human to author the DAG) -> GateDAG (dag.validate_integrity + non-empty — refuse to proceed on an empty/invalid DAG) -> Configure (cursus-configure-pipeline — fill generate_config.py + write config_<region>.json). Auto-chains fully when the DAG comes from the shared catalog; gates on a resumable human handoff otherwise. A THIN composing parent (workflow() nests one level only) — the chaining lives here, not inside cursus-init-project, so init stays phase-0-minimal.',
  phases: [
    { title: 'Scaffold', detail: 'workflow(cursus-init-project) -> phase-0 skeleton + root README ledger' },
    { title: 'SeedDAG', detail: 'catalog: pipeline_catalog.recommend + load_dag -> dag.json; manual: STOP with the ledger + resume instruction' },
    { title: 'GateDAG', detail: 'dag.validate_integrity on a non-empty dag.json — an empty/invalid DAG is an explicit STOP, never a silent pass' },
    { title: 'Configure', detail: 'workflow(cursus-configure-pipeline, {dag_nodes, project, region, project_root})' },
  ],
}

// ---------------------------------------------------------------------------
// INPUT (via `args`): { name: str, framework: str, target_dir?: str, region?: str,
//                       dag_source?: "catalog" | "manual", pattern?: str }
//   dag_source "catalog" (default) — recommend + load a shared catalog DAG into dag.json,
//       then chain straight through to configure (fully automatic).
//   dag_source "manual" — run Scaffold, then STOP: the human authors pipeline_config/dag.json
//       and re-invokes (dag_source not needed on the second run — a non-empty dag.json is
//       detected and the SeedDAG catalog step is skipped).
// The chain gates on "a non-empty, integrity-valid dag.json now exists" before Configure,
// because cursus-configure-pipeline authors one set_step_config per DAG node — with zero nodes
// there is nothing to configure.
// ---------------------------------------------------------------------------

const INPUT = (args && typeof args === 'object' && !Array.isArray(args)) ? args : {}
const NAME = INPUT.name || 'demo_project'
const FRAMEWORK = (INPUT.framework || 'xgboost').toLowerCase()
const TARGET_DIR = INPUT.target_dir || 'projects'
const REGION = INPUT.region || 'NA'
const DAG_SOURCE = INPUT.dag_source || 'catalog'
const PROJECT = `${NAME}_${FRAMEWORK}`
const PROJECT_ROOT = `${TARGET_DIR}/${PROJECT}`
const DAG_PATH = `${PROJECT_ROOT}/pipeline_config/dag.json`

const DAG_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['nodes', 'is_valid'],
  properties: {
    nodes: { type: 'array', items: { type: 'string' } },
    is_valid: { type: 'boolean' },
    notes: { type: 'string' },
  },
}

// --- Scaffold: run the phase-0 init inline -------------------------------
phase('Scaffold')
const scaffold = await workflow('cursus-init-project', { name: NAME, framework: FRAMEWORK, target_dir: TARGET_DIR })
if (!scaffold || !scaffold.ready) {
  return {
    stopped_at: 'Scaffold',
    reason: 'cursus-init-project did not reach ready; see its files_written / skipped',
    project: PROJECT, target_path: PROJECT_ROOT, scaffold,
  }
}

// --- SeedDAG: only the catalog branch is automatable end-to-end ----------
phase('SeedDAG')
if (DAG_SOURCE === 'manual') {
  // The init ledger already tells the human to author dag.json + how to resume.
  return {
    stopped_at: 'SeedDAG',
    reason: 'dag_source="manual" — author ' + DAG_PATH + ' (the ledger lists the steps), then re-invoke cursus-new-project to run the DAG->Configure tail.',
    project: PROJECT, target_path: PROJECT_ROOT,
    next: 'author pipeline_config/dag.json, then re-invoke cursus-new-project',
  }
}
const seed = await agent(
  [
    `Seed ${DAG_PATH} for a ${FRAMEWORK} project from the shared pipeline catalog.`,
    `Call pipeline_catalog.recommend({framework:"${FRAMEWORK}"${INPUT.pattern ? `, pattern:"${INPUT.pattern}"` : ''}}),`,
    'pick the top-recommended dag_id, load it via pipeline_catalog.load_dag(dag_id), and WRITE the resulting',
    `DAG (nodes + edges) to ${DAG_PATH} in the serializer form {"dag": {"nodes": [...], "edges": [...]}}.`,
    'Return {nodes, is_valid: true, notes}. If no catalog DAG fits, return {nodes: [], is_valid: false, notes: "..."}.',
  ].join('\n'),
  { label: 'seed-dag', phase: 'SeedDAG', schema: DAG_SCHEMA, effort: 'high' }
)
if (!seed || !seed.nodes || seed.nodes.length === 0) {
  return {
    stopped_at: 'SeedDAG',
    reason: 'no shared catalog DAG could be seeded — author pipeline_config/dag.json manually, then re-invoke.',
    project: PROJECT, target_path: PROJECT_ROOT, seed,
  }
}

// --- GateDAG: the DAG-exists precondition. Never configure an empty/invalid DAG.
phase('GateDAG')
const gate = await agent(
  [
    `Call dag.validate_integrity on the DAG at ${DAG_PATH} and report {nodes, is_valid}.`,
    'Read the file, pass its {nodes, edges} to dag.validate_integrity, and set is_valid to the tool\'s verdict.',
    'Also confirm nodes.length > 0. Return {nodes, is_valid, notes}.',
  ].join('\n'),
  { label: 'gate-dag', phase: 'GateDAG', schema: DAG_SCHEMA, effort: 'high' }
)
if (!gate || !gate.is_valid || !gate.nodes || gate.nodes.length === 0) {
  return {
    stopped_at: 'GateDAG',
    reason: 'dag.json is empty or fails integrity — configure would have nothing valid to do. Fix the DAG and re-run.',
    project: PROJECT, target_path: PROJECT_ROOT, gate,
  }
}

// --- Configure: chain into the config workflow with the now-real DAG nodes.
phase('Configure')
const config = await workflow('cursus-configure-pipeline', {
  dag_nodes: gate.nodes,
  project: PROJECT,
  region: REGION,
  project_root: PROJECT_ROOT,
})

const shippable = !!(config && config.shippable)
return {
  project: PROJECT,
  target_path: PROJECT_ROOT,
  scaffolded: scaffold.ready,
  dag_nodes: gate.nodes.length,
  config_shippable: shippable,
  next: shippable
    ? 'python ' + PROJECT_ROOT + '/run_pipeline.py --preview  ->  --region ' + REGION
    : 'cursus-configure-pipeline did not reach shippable — read its per_step / cross_node verdict and the ledger',
}
