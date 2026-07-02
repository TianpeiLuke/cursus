#!/usr/bin/env node
// test-author-step-e2e.js — end-to-end proof that the SPLIT Resolve phase of the REAL
// cursus-author-step.js runs to a green authored step on the Kiro runtime, and that the split
// eliminated the SAIS failure mode (a big all-at-once PLAN_SCHEMA turn that even Opus 4.8 answered
// with a multi-element array `[{...},{...}]` the harness cannot coerce).
//
// A mock kiro-cli answers every phase prompt of the real workflow. It routes on each prompt's unique
// TASK marker. Crucially, if it EVER sees the old combined 15-field PLAN_SCHEMA in a single turn
// (detected by that schema's exact `required` fragment) it writes a SENTINEL file and returns the
// multi-element array — the exact SAIS failure. The workflow no longer emits that schema, so the run
// succeeds AND the sentinel never appears. Both facts are asserted.
//
// Run: `node test-author-step-e2e.js`  (offline, deterministic; no live kiro-cli).

'use strict';

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');

let pass = 0;
let fail = 0;
function ok(name, cond) {
  if (cond) pass++;
  else {
    fail++;
    console.error(`FAIL ${name}`);
  }
}

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiro-author-e2e-'));
const sentinel = path.join(dir, 'BIG_SCHEMA_SEEN'); // written iff the mock sees the old combined schema
const alignSingleSentinel = path.join(dir, 'ALIGN_SINGLE_SEEN'); // written iff the single-turn AlignEdges fired (must NOT on Kiro)
const alignEdgeLog = path.join(dir, 'ALIGN_EDGES.log'); // one line per decomposed per-edge align turn

// The mock kiro-cli. It gets the prompt as its last positional arg (as run-workflow.js passes it),
// routes on the prompt's unique TASK marker, and prints a schema-valid JSON reply with the same
// ANSI + "> " decoration real headless kiro-cli emits.
const mock = path.join(dir, 'mock-kiro.js');
fs.writeFileSync(
  mock,
  `#!/usr/bin/env node
const fs = require('node:fs');
const argv = process.argv.slice(2);
const p = argv[argv.length - 1] || "";
process.stderr.write("mock kiro-cli\\n");   // stderr noise the runtime must ignore
function say(obj){ const s = typeof obj === "string" ? obj : JSON.stringify(obj);
  process.stdout.write("\\x1b[38;5;141m> \\x1b[0m" + s + "\\x1b[0m\\n"); process.exit(0); }

// SAIS failure trap: the OLD single-turn PLAN_SCHEMA had this exact required fragment. If the workflow
// still asked for it in one turn, reproduce the observed failure: a MULTI-element array (one plan per
// DAG node) that the runtime cannot coerce to the single object wanted.
if (p.includes('"gap_rung","is_new_step","divergences_from_exemplar","producer","consumer"')) {
  fs.writeFileSync(${JSON.stringify(sentinel)}, "seen");
  return say('[{"step_name":"A"},{"step_name":"B"},{"step_name":"C"}]');
}

// --- Resolve: three small single-object turns ---
if (p.includes("LOCATE ONLY"))
  return say({ node_type:"internal",
    producer:{ node:"XGBoostModelEval_calibration", base_step_type:"XGBoostModelEval", job_type:"calibration" },
    consumer:{ node:"ModelMetricsComputation", base_step_type:"ModelMetricsComputation" } });
if (p.includes("GAP TRIAGE ONLY"))
  return say({ gap_rung:"new_step", gap_rung_reason:"no existing step calibrates probabilities", is_new_step:true });
if (p.includes("NEW-STEP IDENTITY"))
  return say({ step_name:"BetaCalibration", snake_name:"beta_calibration", sagemaker_step_type:"Processing",
    step_assembly:"code", framework:"sklearn", reuse_class:"shared", bound_handler:"ProcessingHandler",
    exemplar_step:"TabularPreprocessing", needed_axes:["inputs","outputs"],
    divergences_from_exemplar:["outputs: ADD a calibrator output (fitted beta .pkl) — exemplar emits only processed_data"] });

// --- Challenge (rung holds) ---
if (p.includes("TRY TO BREAK"))
  return say({ holds:true, final_rung:"new_step", challenges:["reuse_config_only checked: none fits"], correction:"" });

// --- AlignEdges (single-turn, tool-forcing host) — must NEVER fire on the Kiro path; leave a sentinel ---
if (p.includes("ALIGN the two DAG edges")) {
  fs.writeFileSync(${JSON.stringify(alignSingleSentinel)}, "seen");
  return say({ arity_ok:true, arity_note:"one data producer",
    edges:[{ edge:"producer->NEW", dependency_type:"PROCESSING_OUTPUT", output_type:"PROCESSING_OUTPUT", type_ok:true, data_type_ok:true, projected_score:0.85, resolves:true, fragile:false },
           { edge:"NEW->consumer", dependency_type:"PROCESSING_OUTPUT", output_type:"PROCESSING_OUTPUT", type_ok:true, data_type_ok:true, projected_score:0.8, resolves:true, fragile:false }],
    consumer_edits:["add BetaCalibration to ModelMetricsComputation dependency compatible_sources"], ready:true });
}
// --- AlignEdges DECOMPOSED (Kiro path): one object per edge + one arity object; count the per-edge turns ---
if (p.includes("ALIGN EXACTLY ONE EDGE")) {
  const m = p.match(/ALIGN EXACTLY ONE EDGE: "([^"]+)"/);
  const edge = m ? m[1] : "producer->NEW";
  const isProd = edge === "producer->NEW";
  try { fs.appendFileSync(${JSON.stringify(alignEdgeLog)}, edge + "\\n"); } catch (e) {}
  return say({ edge, dependency_type:"PROCESSING_OUTPUT", output_type:"PROCESSING_OUTPUT", type_ok:true, data_type_ok:true,
    projected_score: isProd ? 0.85 : 0.8, resolves:true, fragile:false,
    consumer_edits: isProd ? [] : ["add BetaCalibration to ModelMetricsComputation dependency compatible_sources"] });
}
if (p.includes("ARITY CHECK ONLY"))
  return say({ arity_ok:true, arity_note:"one data producer" });

// --- Guide (one per needed axis) ---
{ const m = p.match(/guidance for the "(\\w+)" axis/);
  if (m) return say({ axis:m[1], recommended:"value for "+m[1], restrictions:[], exemplar_snippet:m[1]+": ..." }); }

// --- Author (no schema — prose reply) ---
if (p.includes("write the THREE artifacts"))
  return say("Wrote src/cursus/steps/interfaces/beta_calibration.step.yaml, config, script; edited consumer spec. Divergence present: calibrator output.");

// --- Validate loop / oracles ---
if (p.includes("validate.step_interface(step_name="))
  return say({ ok:true, errors:[], warnings:[] });
if (p.includes("author.check_script(step_name="))
  return say({ status:"checked", passed:true, issues:[] });
if (p.includes("actually PARSE"))
  return say({ py_compile_ok:true, yaml_loads_ok:true, detail:"exit 0" });
if (p.includes("BECAUSE of these REQUIRED divergences:"))
  return say({ all_present:true, checks:[{ divergence:"outputs: ADD a calibrator output", present:true, evidence:"spec.outputs.calibrator" }] });
// --- re-resolve DECOMPOSED (Kiro path): one object per edge (check BEFORE the single-turn branch) ---
if (p.includes("Report ONLY the single edge")) {
  const m = p.match(/Report ONLY the single edge "([^"]+)"/);
  const edge = m ? m[1] : "producer->NEW";
  return say({ edge, score: edge === "producer->NEW" ? 0.85 : 0.8, resolves:true, note:"" });
}
// --- re-resolve single-turn (tool-forcing host) ---
if (p.includes("validate.deps_resolve(step_names="))
  return say({ both_edges_resolve:true, edges:[{ edge:"producer->NEW", score:0.85, resolves:true, note:"" },{ edge:"NEW->consumer", score:0.8, resolves:true, note:"" }] });
if (p.includes("author.preflight_step(step_name="))
  return say({ constructible:true, gates:[{ name:"offline_construct", passed:true, detail:"ok" }] });
if (p.includes("COMPLETENESS CRITIC"))
  return say({ full_dag_scanned:true, edit_collateral:[], uncovered_consumers:[], cross_branch_column_mismatches:[], sequencing_risks:[] });

// Fallback — any unrouted prompt is a test gap; make it loud (invalid for every schema).
return say("UNROUTED PROMPT — mock has no branch for this turn");
`
);
fs.chmodSync(mock, 0o755);

const runner = path.join(__dirname, 'run-workflow.js');
const workflow = path.join(__dirname, '..', 'cursus-author-step.js');
const reqArgs = JSON.stringify({
  intent: 'a post-training probability calibration step',
  name: 'BetaCalibration',
  producer_node: 'XGBoostModelEval_calibration',
  consumer_node: 'ModelMetricsComputation',
}); // no dag_path -> the Gaps critic is skipped

let out = '';
let threw = false;
try {
  out = execFileSync(
    'node',
    [runner, workflow, '--kiro-bin', mock, '--args', reqArgs, '--concurrency', '4', '--timeout-ms', '20000'],
    { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }
  );
} catch (e) {
  threw = true;
  out = e.stdout || '';
  console.error('runner exited non-zero: ' + (e.message || '').slice(0, 200));
}

let result = null;
try {
  result = JSON.parse(out);
} catch (e) {
  console.error('result not JSON: ' + out.slice(0, 300));
}

ok('runner completed without throwing', !threw);
ok('result parsed as JSON', !!result);
ok('the old combined PLAN_SCHEMA was never emitted (no SAIS array trap)', !fs.existsSync(sentinel));
if (result) {
  ok('exactly one step authored', Array.isArray(result.authored) && result.authored.length === 1);
  const a = result.authored && result.authored[0];
  ok('authored step is BetaCalibration', a && a.step === 'BetaCalibration');
  ok('authored step is green (all gates passed, no open gaps)', a && a.status === 'green');
  ok('merged plan carried the sagemaker_step_type from the identity turn', a && a.sagemaker_step_type === 'Processing');
  ok('gap_rung merged from the triage turn', a && a.gap_rung === 'new_step');
  ok('required divergence flowed through the split into the plan', a && Array.isArray(a.divergences_required) && a.divergences_required.length === 1);
  ok('divergence-fidelity gate saw it present', a && a.divergences_present === true);
  ok('both edges resolve after specs written', a && a.edges_resolve === true);
  ok('nothing resolved-without-a-new-step', Array.isArray(result.resolved_without_new_step) && result.resolved_without_new_step.length === 0);
}
// AlignEdges was DECOMPOSED on the Kiro path: the single-turn ALIGN_SCHEMA prompt must NEVER fire, and
// there must be one per-edge align turn per real edge (2 here: producer->NEW + NEW->consumer).
ok('AlignEdges single-turn (array-of-objects) prompt was NEVER emitted on the Kiro path', !fs.existsSync(alignSingleSentinel));
{
  const edgeTurns = fs.existsSync(alignEdgeLog) ? fs.readFileSync(alignEdgeLog, 'utf8').trim().split('\n').filter(Boolean) : [];
  ok('AlignEdges ran one per-edge turn per real edge (2)', edgeTurns.length === 2 && edgeTurns.includes('producer->NEW') && edgeTurns.includes('NEW->consumer'));
}

fs.rmSync(dir, { recursive: true, force: true });
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
