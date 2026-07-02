#!/usr/bin/env node
// run-toolforce-probe.js — a ONE-SHOT probe answering the single open question behind the tool-forcing
// fix: does a HEADLESS `kiro-cli chat --no-interactive` turn on THIS build (esp. the frozen SAIS 2.5.0)
//   (1) read a static ~/.kiro-style MCP config,
//   (2) run its agentic tool loop, and
//   (3) actually emit a `submit_result` tool call with structured arguments (not just prose)?
// If yes, submit-result-server.js can become the primary schema-forcing path (the Claude Code
// StructuredOutput analog). If no, the four-layer emulation (shapeDirective/tolerantParse/coerce/
// re-prompt) stays the mechanism on this build. This is the 2.5.0-SAFE route by construction: it uses
// ONLY a static config file written BEFORE spawn + headless — never dynamic ACP session/new (which
// crashes kiro-cli 2.5.0, SAIS Run 8).
//
// USAGE
//   node run-toolforce-probe.js [--kiro-bin kiro-cli] [--model <id>] [--agent <name>] [--legacy-kiro]
//                               [--mcp-config <path>] [--timeout-ms 120000]
//
//   --mcp-config <path>  where to write the MCP config the run points kiro-cli at. DEFAULT: a temp
//                        file, passed via the KIRO_MCP_CONFIG env var. NOTE: kiro-cli reads MCP servers
//                        from ~/.kiro/settings/mcp.json (or the --agent spec). If this build ignores an
//                        env/flag override, pass --mcp-config ~/.kiro/settings/mcp.json to write there
//                        directly (the script will refuse to clobber an existing one unless --force).
//   --force              allow overwriting an existing --mcp-config file (it is restored afterward).
//
// OUTPUT: a clear GREEN/RED verdict on STDERR + a JSON summary on STDOUT.

'use strict';

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

function parseArgv(argv) {
  const out = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const k = a.slice(2);
      if (k === 'legacy-kiro' || k === 'force') out[k] = true;
      else out[k] = argv[++i];
    } else out._.push(a);
  }
  return out;
}

function log(s) {
  process.stderr.write(s + '\n');
}

function main() {
  const opts = parseArgv(process.argv.slice(2));
  const kiroBin = opts['kiro-bin'] || 'kiro-cli';
  const timeoutMs = opts['timeout-ms'] ? Number(opts['timeout-ms']) : 120000;

  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiro-toolforce-probe-'));
  const schemaFile = path.join(workDir, 'schema.json');
  const resultFile = path.join(workDir, 'result.json');
  const serverPath = path.join(__dirname, 'submit-result-server.js');

  // A trivial one-field object schema — the model must return {"answer": "..."} via the tool.
  const schema = { type: 'object', required: ['answer'], properties: { answer: { type: 'string' } } };
  fs.writeFileSync(schemaFile, JSON.stringify(schema));

  const mcpJson = {
    mcpServers: {
      'cursus-submit': {
        command: process.execPath,
        args: [serverPath],
        env: { SUBMIT_SCHEMA_FILE: schemaFile, SUBMIT_RESULT_FILE: resultFile, SUBMIT_TOOL_NAME: 'submit_result' },
      },
    },
  };

  // Decide where to write the config. Default temp + env override; if the build ignores that, the user
  // points --mcp-config at ~/.kiro/settings/mcp.json (the confirmed static location).
  const cfgPath = opts['mcp-config'] ? path.resolve(opts['mcp-config']) : path.join(workDir, 'mcp.json');
  let restore = null;
  if (opts['mcp-config'] && fs.existsSync(cfgPath)) {
    if (!opts.force) {
      log(`✗ ${cfgPath} already exists — pass --force to overwrite (it will be restored afterward).`);
      process.exit(2);
    }
    restore = fs.readFileSync(cfgPath, 'utf8');
  }
  fs.mkdirSync(path.dirname(cfgPath), { recursive: true });
  fs.writeFileSync(cfgPath, JSON.stringify(mcpJson, null, 2));
  log(`· wrote MCP config: ${cfgPath}`);
  log(`· submit-result server: ${serverPath}`);

  const args = ['chat', '--no-interactive'];
  args.push('--trust-all-tools');
  if (opts.agent) args.push('--agent', opts.agent);
  if (opts.model) args.push('--model', opts.model);
  if (opts.effort && !opts['legacy-kiro']) args.push('--effort', opts.effort);
  const prompt =
    'Call the submit_result tool EXACTLY ONCE, passing answer set to the string "HELLO". ' +
    'Do not print anything else. The tool validates your arguments against its inputSchema.';
  args.push(prompt);

  log(`· spawning: ${kiroBin} ${args.slice(0, -1).join(' ')} "<prompt>"`);
  const env = { ...process.env, KIRO_MCP_CONFIG: cfgPath };
  const res = spawnSync(kiroBin, args, { env, encoding: 'utf8', timeout: timeoutMs });

  // Restore a clobbered user config.
  if (restore != null) fs.writeFileSync(cfgPath, restore);

  const captured = fs.existsSync(resultFile) ? JSON.parse(fs.readFileSync(resultFile, 'utf8')) : null;
  const exit = res.status;
  const crashedEarly = res.status !== 0 && !captured;
  const summary = {
    green: !!(captured && captured.answer === 'HELLO' && exit === 0),
    exit_code: exit,
    signal: res.signal || null,
    tool_was_invoked: !!captured,
    captured_arguments: captured,
    stderr_tail: (res.stderr || '').split('\n').slice(-8).join('\n'),
    stdout_tail: (res.stdout || '').split('\n').slice(-8).join('\n'),
  };

  log('');
  if (summary.green) {
    log('✅ GREEN — kiro-cli headless read the static MCP config, ran the tool loop, and dispatched a');
    log('   real submit_result call with structured arguments. Tool-forcing IS viable on this build:');
    log('   wire submit_result in as the primary schema-forcing path (keep the 4-layer emulation as fallback).');
  } else if (summary.tool_was_invoked) {
    log('🟡 PARTIAL — the tool was invoked but the payload/exit was unexpected (see captured_arguments/exit_code).');
  } else if (crashedEarly) {
    log('🔴 RED — kiro-cli exited non-zero WITHOUT invoking the tool (possible early-exit like Run 8, or it');
    log('   ignored the MCP config). Do NOT wire tool-forcing on this build; keep the 4-layer emulation.');
  } else {
    log('🔴 RED — kiro-cli returned but never called submit_result (it likely answered in prose). This build');
    log('   does not reliably tool-force via a static MCP config; keep the 4-layer emulation as the mechanism.');
    log('   If you used the temp config, retry with --mcp-config ~/.kiro/settings/mcp.json (the confirmed location).');
  }
  log('');

  fs.rmSync(workDir, { recursive: true, force: true });
  process.stdout.write(JSON.stringify(summary, null, 2) + '\n');
  process.exit(summary.green ? 0 : 1);
}

main();
