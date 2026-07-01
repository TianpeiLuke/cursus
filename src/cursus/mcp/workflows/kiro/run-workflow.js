#!/usr/bin/env node
// run-workflow.js — execute an UNMODIFIED Claude Code dynamic-workflow .js on Kiro (kiro-cli).
//
// USAGE
//   node run-workflow.js <workflow.js> [--args '<json>'] [options]
//   node run-workflow.js ../cursus-author-step.js --args '{"intent":"...","name":"...", ...}'
//
// OPTIONS (all optional)
//   --args '<json>'      value bound to the workflow's `args` global (object or array). Also accepted
//                        from the KIRO_WF_ARGS env var, or a @file.json path.
//   --agent <name>       kiro-cli agent (context profile) for every turn
//   --model <id>         kiro-cli model for every turn
//   --effort <level>     low|medium|high|xhigh|max — default per turn (opts.effort still overrides)
//   --cwd <dir>          working dir kiro-cli runs in (default: the workflow file's dir, so relative
//                        dag_path / project_root in args resolve the same as under Claude Code)
//   --concurrency <n>    max concurrent turns (default: min(16, cores-2))
//   --timeout-ms <n>     per-turn timeout (default 900000 = 15 min)
//   --trust-tools <csv>  pass --trust-tools=<csv> instead of --trust-all-tools
//   --kiro-bin <path>    kiro-cli binary (default: kiro-cli on PATH)
//   --budget <n>         token-estimate ceiling for budget.total (default: none / Infinity)
//   --transport <t>      'headless' (default; one `kiro-cli chat --no-interactive` per turn) or
//                        'acp' (one long-lived `kiro-cli acp` process, one ACP session per turn —
//                        persistent JSON-RPC connection, streaming, per-tool permission handling)
//   --print-result       write the workflow's return value as JSON to STDOUT (default on)
//
//   Version-skew controls — SAIS runs a FROZEN kiro-cli 2.5.0 snapshot; the runtime was captured on
//   2.10.0. On the 2.5.0 build, pass these so newer flags/entry points don't hard-fail:
//   --legacy-kiro        emit only the 2.5.0-safe flag set (drops `--effort` and granular
//                        `--trust-tools`; keeps `--no-interactive`/`--trust-all-tools`/`--agent`/`--model`)
//   --acp-entry <e>      'subcommand' (default, `kiro-cli acp`) or 'chat-binary' (2.5.0 ships a
//                        separate `kiro-cli-chat` binary that IS the ACP server)
//   --kiro-chat-bin <p>  the ACP-server binary for --acp-entry chat-binary (default: kiro-cli-chat)
//   --acp-protocol <n>   ACP protocolVersion to request (default 1; re-negotiated to server's echo)
//
// CONTRACT
//   The workflow's return value is printed to STDOUT as pretty JSON; ALL progress/log lines go to
//   STDERR. So `node run-workflow.js ... 1>result.json 2>progress.log` captures the result cleanly.
//
// This is the Kiro counterpart to the Claude Code `Workflow` tool. It binds the same primitives
// (agent/parallel/pipeline/phase/log) + globals (args/budget) that the CC host injects, so the same
// script runs in both places. See README.md in this directory.

'use strict';

const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { KiroWorkflowRuntime } = require('./kiro-workflow-runtime');

function parseArgv(argv) {
  const out = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const key = a.slice(2);
      const flags = new Set(['print-result', 'no-print-result', 'legacy-kiro']);
      if (flags.has(key)) out[key] = true;
      else out[key] = argv[++i];
    } else out._.push(a);
  }
  return out;
}

function loadArgsValue(raw) {
  if (raw == null) return undefined;
  let text = raw;
  if (typeof raw === 'string' && raw.startsWith('@')) {
    text = fs.readFileSync(raw.slice(1), 'utf8');
  }
  try {
    return JSON.parse(text);
  } catch (e) {
    throw new Error(`--args is not valid JSON: ${e.message}`);
  }
}

// Strip `export ` from `export const meta = ...` / `export default ...` so the CC ESM script body
// evaluates as a plain script. The CC workflow body is otherwise ordinary JS with top-level await,
// which we support by wrapping it in an async function.
function normalizeSource(src) {
  return src
    .replace(/^\s*export\s+default\s+/m, 'return ')
    .replace(/^\s*export\s+(const|let|var|function|class)\s+/gm, '$1 ');
}

async function main() {
  const opts = parseArgv(process.argv.slice(2));
  const workflowPath = opts._[0];
  if (!workflowPath) {
    process.stderr.write('usage: node run-workflow.js <workflow.js> [--args <json>] [options]\n');
    process.exit(2);
  }
  const absWorkflow = path.resolve(workflowPath);
  if (!fs.existsSync(absWorkflow)) {
    process.stderr.write(`workflow not found: ${absWorkflow}\n`);
    process.exit(2);
  }

  const argsValue = loadArgsValue(opts.args != null ? opts.args : process.env.KIRO_WF_ARGS);
  const cwd = opts.cwd ? path.resolve(opts.cwd) : path.dirname(absWorkflow);

  const runtime = new KiroWorkflowRuntime({
    kiroBin: opts['kiro-bin'],
    cwd,
    agent: opts.agent,
    model: opts.model,
    effort: opts.effort,
    trustTools: opts['trust-tools'],
    concurrency: opts.concurrency ? Number(opts.concurrency) : undefined,
    timeoutMs: opts['timeout-ms'] ? Number(opts['timeout-ms']) : undefined,
    budgetTotal: opts.budget ? Number(opts.budget) : null,
    transport: opts.transport, // 'headless' (default) | 'acp'
    // Version-skew controls (SAIS is a frozen kiro-cli 2.5.0 snapshot):
    allowNewFlags: opts['legacy-kiro'] ? false : undefined, // --legacy-kiro => 2.5.0-safe flags only
    acpEntry: opts['acp-entry'], // 'subcommand' (default) | 'chat-binary'
    kiroChatBin: opts['kiro-chat-bin'], // the 2.5.0 ACP-server binary (default kiro-cli-chat)
    acpProtocolVersion: opts['acp-protocol'] ? Number(opts['acp-protocol']) : undefined,
  });

  const rawSource = fs.readFileSync(absWorkflow, 'utf8');
  const body = normalizeSource(rawSource);

  // Bind the primitives to the runtime instance and expose them + globals to the script.
  const sandbox = {
    agent: (prompt, o) => runtime.agent(prompt, o),
    parallel: (thunks) => runtime.parallel(thunks),
    pipeline: (items, ...stages) => runtime.pipeline(items, ...stages),
    phase: (title) => runtime.phase(title),
    log: (msg) => runtime.log(msg),
    args: argsValue,
    budget: runtime.budget,
    // give scripts the usual ambient globals
    console: { log: (...a) => runtime.log(a.join(' ')), error: (...a) => runtime.log(a.join(' ')) },
    JSON,
    Math,
    Array,
    Object,
    String,
    Number,
    Boolean,
    Promise,
    Set,
    Map,
    Date, // NOTE: unlike the CC host, Date is available here (kiro runs are not journaled/resumed)
    RegExp,
    parseInt,
    parseFloat,
    isNaN,
    isFinite,
  };

  const wrapped =
    '(async function __kiro_workflow__() {\n' +
    body +
    '\n})()';

  const context = vm.createContext(sandbox);
  let metaCaptured = null;
  try {
    const script = new vm.Script(wrapped, { filename: absWorkflow });
    const resultPromise = script.runInContext(context);
    // `meta` is a top-level const inside the async wrapper, so it is not visible on the sandbox;
    // grab it from the source's meta.name for the banner only (best-effort, non-fatal).
    const nameMatch = rawSource.match(/name:\s*['"]([^'"]+)['"]/);
    metaCaptured = nameMatch ? nameMatch[1] : path.basename(absWorkflow);
    runtime._emit(`\n▶ running workflow "${metaCaptured}" on kiro-cli (cwd: ${cwd})`);

    const result = await resultPromise;
    runtime._emit(
      `\n■ workflow complete — ${runtime.totalAgents} agent turn(s), ` +
        `~${runtime._spent} est. tokens`
    );
    if (opts['no-print-result'] !== true) {
      process.stdout.write(JSON.stringify(result ?? null, null, 2) + '\n');
    }
    await runtime.close();
    process.exit(0);
  } catch (e) {
    runtime._emit(`\n✗ workflow error: ${e && e.stack ? e.stack : e}`);
    await runtime.close().catch(() => {});
    process.exit(1);
  }
}

main().catch((e) => {
  process.stderr.write(`fatal: ${e && e.stack ? e.stack : e}\n`);
  process.exit(1);
});
