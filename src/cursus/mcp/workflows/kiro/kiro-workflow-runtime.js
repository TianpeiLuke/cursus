// kiro-workflow-runtime.js — run Claude Code dynamic-workflow .js scripts on Kiro.
//
// WHY THIS EXISTS
// ---------------
// The workflow scripts in ../ (cursus-author-step.js, cursus-configure-pipeline.js) are written
// for the Claude Code `Workflow` runtime: an `export const meta = {...}` header plus the host
// primitives `agent()` / `parallel()` / `pipeline()` / `phase()` / `log()` and the `args` / `budget`
// globals. Those primitives are INJECTED by the Claude Code host — they are not defined in the file
// and there is no `node <workflow>.js`, so the scripts are inert under `kiro-cli` (FZ 31e1d3f5a5;
// see abuse_slipbox faq_kiro_run_claude_code_workflow.md). Kiro's own agent cannot execute that
// runtime.
//
// This module re-implements those primitives on top of `kiro-cli` running HEADLESS
// (`kiro-cli chat --no-interactive`). Each `agent()` call becomes one non-interactive kiro-cli turn
// — a fresh sub-agent per call, which is exactly the Claude Code semantics (the sub-agent's final
// message IS the return value). Because the primitive API is kept identical, the *unmodified*
// workflow scripts run here. What differs from the real CC host: schema-forced output is emulated
// by a JSON-schema prompt suffix + parse/validate + bounded re-prompt (CC forces a StructuredOutput
// tool call), and there is no true token budget (kiro reports credits, not tokens) — see `budget`.
//
// The real work is still done by the SAME MCP tools: kiro-cli invokes author.* / validate.* /
// steps.io / compile.* itself when a phase prompt tells it to. This is the "re-enact via the same
// tools" conclusion, automated and ordered.
//
// TRANSPORT CHOICE — headless vs ACP: this runtime uses headless (`chat --no-interactive`) because
// each agent() call is an independent one-shot turn (no shared session needed), it is dependency-free
// (no npm package, no ACP JSON-RPC handshake), and it matches the verified internal headless pattern
// (SAGE 2054738). `kiro-cli acp` (JSON-RPC 2.0 over stdio, driven from JS via @agentclientprotocol/sdk
// — A2AServer4KiroCLI / KiroOverlayApp / AGIArsenalConsole) is the streaming alternative; see README.

'use strict';

const { spawn } = require('node:child_process');
const os = require('node:os');
const { KiroAcpClient } = require('./kiro-acp-client');

// ---------------------------------------------------------------------------------------------
// Output cleaning — kiro-cli headless writes the model reply to STDOUT (with ANSI colour codes, a
// leading "> " prompt marker, and sometimes a ```json fence); warnings/spinner/footer go to STDERR.
// ---------------------------------------------------------------------------------------------

// eslint-disable-next-line no-control-regex
const ANSI_RE = /\x1b\[[0-9;?]*[ -/]*[@-~]/g;

function stripAnsi(s) {
  return String(s).replace(ANSI_RE, '');
}

// Turn raw headless stdout into the model's plain reply text.
function cleanReply(stdout) {
  let t = stripAnsi(stdout);
  // Drop the leading "> " turn marker kiro prints before the answer.
  t = t.replace(/^\s*>\s?/, '');
  // Strip a single wrapping ```...``` / ```json fence if the whole reply is fenced.
  const fence = t.match(/```(?:json|JSON)?\s*([\s\S]*?)\s*```/);
  if (fence && fence[1] && t.trim().startsWith('```')) t = fence[1];
  return t.trim();
}

// Extract the first balanced top-level JSON value ({...} or [...]) from arbitrary text. Handles
// strings and escapes so braces inside string literals do not miscount. Returns the substring or null.
function extractJsonText(text) {
  const s = String(text);
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c !== '{' && c !== '[') continue;
    const open = c;
    const close = c === '{' ? '}' : ']';
    let depth = 0;
    let inStr = false;
    let esc = false;
    for (let j = i; j < s.length; j++) {
      const ch = s[j];
      if (inStr) {
        if (esc) esc = false;
        else if (ch === '\\') esc = true;
        else if (ch === '"') inStr = false;
        continue;
      }
      if (ch === '"') inStr = true;
      else if (ch === open) depth++;
      else if (ch === close) {
        depth--;
        if (depth === 0) return s.slice(i, j + 1);
      }
    }
  }
  return null;
}

// Tolerant JSON parse. Tries strict JSON.parse FIRST — so already-valid JSON is returned untouched and
// is NEVER mutated by a repair. Only if strict parse throws does it apply a sequence of SAFE, syntax-
// level repairs (each provably unable to corrupt valid JSON, since valid JSON already returned) and
// re-parse. This is RC#3's parse-failure half: on the frozen kiro-cli 2.5.0 a model commonly emits
// JSON5-ish output (trailing commas, // or /* */ comments, single-quoted strings, smart quotes, Python
// True/False/None, unquoted keys) that the bare JSON.parse rejects with an opaque error the model then
// repeats across re-prompts. Returns { value, repaired (bool), repairs: string[], error?: string }.
// repairs names each defect fixed so the re-prompt can be specific. On total failure value is undefined
// and error carries the strict-parse message.
function tolerantParse(text) {
  const raw = String(text);
  // 1) Strict first. Valid JSON exits here — no repair ever runs on it.
  try {
    return { value: JSON.parse(raw), repaired: false, repairs: [] };
  } catch (strictErr) {
    const repairs = [];
    // Walk the text tracking whether we are inside a JSON string, so every repair is applied ONLY in
    // the appropriate context (never inside string values, where the characters are legitimate data).
    // A single left-to-right scan handles quotes/comments/commas together to avoid cross-interference.
    let out = '';
    const s = raw;
    let inStr = false; // inside a double-quoted JSON string
    let esc = false;
    for (let i = 0; i < s.length; i++) {
      const ch = s[i];
      const next = s[i + 1];
      if (inStr) {
        // A curly close-quote terminates a string we opened from a curly OPEN quote — normalize it to
        // ASCII so the string closes properly (otherwise “v” would become "v” and never terminate).
        if (ch === '”' || ch === '“') {
          if (!repairs.includes('normalized smart quotes')) repairs.push('normalized smart quotes');
          out += '"';
          inStr = false;
          continue;
        }
        out += ch;
        if (esc) esc = false;
        else if (ch === '\\') esc = true;
        else if (ch === '"') inStr = false;
        continue;
      }
      // --- OUTSIDE a string ---
      // Line comment // ... -> drop to end of line
      if (ch === '/' && next === '/') {
        if (!repairs.includes('stripped // line comments')) repairs.push('stripped // line comments');
        while (i < s.length && s[i] !== '\n') i++;
        continue;
      }
      // Block comment /* ... */ -> drop
      if (ch === '/' && next === '*') {
        if (!repairs.includes('stripped /* */ block comments')) repairs.push('stripped /* */ block comments');
        i += 2;
        while (i < s.length && !(s[i] === '*' && s[i + 1] === '/')) i++;
        i++; // skip the closing '/'
        continue;
      }
      // Smart / curly double quotes -> ASCII " (opening a JSON string)
      if (ch === '“' || ch === '”') {
        if (!repairs.includes('normalized smart quotes')) repairs.push('normalized smart quotes');
        out += '"';
        inStr = true;
        continue;
      }
      // A single-quoted string 'like this' -> re-emit as a double-quoted JSON string (escaping any
      // embedded " and preserving \' as '). Only OUTSIDE a real string, so apostrophes in JSON string
      // *values* are never touched.
      if (ch === "'") {
        if (!repairs.includes('converted single-quoted strings')) repairs.push('converted single-quoted strings');
        let body = '';
        i++;
        while (i < s.length && s[i] !== "'") {
          if (s[i] === '\\' && (s[i + 1] === "'" || s[i + 1] === '\\')) { body += s[i + 1]; i += 2; continue; }
          if (s[i] === '"') { body += '\\"'; i++; continue; }
          body += s[i];
          i++;
        }
        out += '"' + body + '"';
        continue;
      }
      out += ch;
      if (ch === '"') inStr = true;
    }
    // Trailing commas: `,` immediately before a } or ] (allowing whitespace) — only meaningful outside
    // strings, and the scan above preserved strings verbatim, so a regex over `out` is safe here.
    const beforeTrailing = out;
    out = out.replace(/,(\s*[}\]])/g, '$1');
    if (out !== beforeTrailing) repairs.push('removed trailing commas');
    // Python / JS literals as bare words (outside strings). Word-boundary anchored so they don't hit
    // substrings of keys; strings were preserved verbatim above so real string values are untouched.
    const litMap = [
      [/\bTrue\b/g, 'true', 'True->true'],
      [/\bFalse\b/g, 'false', 'False->false'],
      [/\bNone\b/g, 'null', 'None->null'],
      [/\bNaN\b/g, 'null', 'NaN->null'],
      [/\b-?Infinity\b/g, 'null', 'Infinity->null'],
    ];
    for (const [re, to, name] of litMap) {
      if (re.test(out)) { out = out.replace(re, to); repairs.push('mapped Python literal ' + name); }
    }
    // Unquoted object keys: `{ key:` or `, key:` -> quote the key. Conservative: only bare identifiers.
    const beforeKeys = out;
    out = out.replace(/([{,]\s*)([A-Za-z_$][A-Za-z0-9_$]*)(\s*:)/g, '$1"$2"$3');
    if (out !== beforeKeys) repairs.push('quoted unquoted keys');

    if (!repairs.length) {
      // Nothing we recognize — surface the original strict error, unrepaired.
      return { value: undefined, repaired: false, repairs: [], error: strictErr.message };
    }
    try {
      return { value: JSON.parse(out), repaired: true, repairs };
    } catch (repairErr) {
      return { value: undefined, repaired: false, repairs, error: repairErr.message };
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Minimal JSON-Schema validation — enough to emulate CC's schema-forced output for the shapes the
// cursus workflows use (object with `required`, `properties`, `type`, `enum`, `items`). It is a
// gate, not a full validator: it checks presence, primitive type, and enum membership recursively.
// ---------------------------------------------------------------------------------------------

function typeOf(v) {
  if (v === null) return 'null';
  if (Array.isArray(v)) return 'array';
  if (Number.isInteger(v)) return 'integer';
  return typeof v; // 'object' | 'string' | 'number' | 'boolean'
}

function typeMatches(v, t) {
  if (!t) return true;
  const types = Array.isArray(t) ? t : [t];
  const actual = typeOf(v);
  return types.some((want) => {
    if (want === 'number') return actual === 'number' || actual === 'integer';
    if (want === 'integer') return actual === 'integer';
    return actual === want;
  });
}

// Build an explicit natural-language shape directive from a schema's top-level type. Kiro can't
// tool-force output, and some models (notably Opus 4.8 on the frozen kiro-cli 2.5.0) have a strong
// bias to wrap even a SMALL object answer in an array — the raw JSON Schema alone does not override
// it (observed: the little `locate` sub-schema still came back as `[{...},{...}]`). So we spell the
// container out in words AND show a skeleton of the exact top-level shape; the literal `{ }` braces
// anchor the model to an object. This is RC#1(a) — "return ONE object, not an array of alternatives" —
// and because it lives in the runtime it covers EVERY schema-gated turn in EVERY workflow at once
// (author-step's locate/triage/identity AND configure-pipeline's DagCheck + per-node Validate), with
// no per-workflow edits and no model change. The Claude Code host tool-forces the shape, so this is
// Kiro-runtime-specific.
function shapeDirective(schema) {
  if (!schema || typeof schema !== 'object') return 'Return a single JSON value and NOTHING else.';
  const t = Array.isArray(schema.type) ? schema.type[0] : schema.type;
  if (t === 'object') {
    const keys = Object.keys(schema.properties || {});
    const skeleton = keys.length
      ? '{ ' + keys.map((k) => JSON.stringify(k) + ': ...').join(', ') + ' }'
      : '{ ... }';
    return (
      'Return EXACTLY ONE JSON object and NOTHING else. Your entire reply MUST start with `{` and end ' +
      'with `}`. Do NOT return a JSON array; do NOT wrap the object in `[ ]`; do NOT return a list of ' +
      'candidates or alternatives — if you weighed several options, pick the single best and return ONLY ' +
      'that one object. The top-level shape is exactly:\n' + skeleton
    );
  }
  if (t === 'array') return 'Return a JSON array `[ ... ]` and NOTHING else (no prose, no object wrapper).';
  return 'Return a single JSON value and NOTHING else.';
}

// Coerce common LLM shape mismatches toward the schema's top-level type BEFORE validating. Models
// (even strong ones like Opus 4.8) sometimes wrap a single object in a one-element array (`[{...}]`)
// when the schema wants an object, or return a bare object when the schema wants an array — and they
// can stay stuck on that shape across re-prompts. This normalizes those cases only; it never changes
// field values. Returns { value, coerced (bool), note }. The Claude Code host tool-forces the exact
// shape, so this path is Kiro-runtime-specific.
function coerceToSchema(value, schema) {
  if (!schema || typeof schema !== 'object' || !schema.type) return { value, coerced: false };
  const want = Array.isArray(schema.type) ? schema.type : [schema.type];
  const actual = typeOf(value);
  // object wanted, got an array of object(s)
  if (want.includes('object') && !want.includes('array') && actual === 'array') {
    const objs = value.filter((v) => typeOf(v) === 'object');
    // single-element array holding an object -> unwrap
    if (value.length === 1 && objs.length === 1) {
      return { value: value[0], coerced: true, note: 'unwrapped single-element array -> object' };
    }
    // multi-element array of IDENTICAL objects (the model emitted the same answer N times) -> unwrap
    // to the first; no data is lost because the elements are deep-equal. Genuinely DIFFERENT elements
    // are ambiguous (which one is "the" answer?), so leave those for the re-prompt rather than guess.
    if (value.length > 1 && objs.length === value.length) {
      const first = JSON.stringify(value[0]);
      if (value.every((v) => JSON.stringify(v) === first)) {
        return { value: value[0], coerced: true, note: `unwrapped ${value.length}-element array of identical objects -> object` };
      }
    }
  }
  // array wanted, got a bare object -> wrap
  if (want.includes('array') && !want.includes('object') && actual === 'object') {
    return { value: [value], coerced: true, note: 'wrapped bare object -> single-element array' };
  }
  return { value, coerced: false };
}

function validateAgainstSchema(value, schema, path = '$') {
  const errors = [];
  if (!schema || typeof schema !== 'object') return errors;

  if (schema.type && !typeMatches(value, schema.type)) {
    errors.push(`${path}: expected type ${JSON.stringify(schema.type)}, got ${typeOf(value)}`);
    return errors; // type is wrong — deeper checks are meaningless
  }
  if (schema.enum && !schema.enum.includes(value)) {
    errors.push(`${path}: value ${JSON.stringify(value)} not in enum ${JSON.stringify(schema.enum)}`);
  }
  if (typeOf(value) === 'object' && (schema.properties || schema.required)) {
    for (const key of schema.required || []) {
      if (!(key in value)) errors.push(`${path}.${key}: required property missing`);
    }
    for (const [key, sub] of Object.entries(schema.properties || {})) {
      if (key in value) errors.push(...validateAgainstSchema(value[key], sub, `${path}.${key}`));
    }
  }
  if (typeOf(value) === 'array' && schema.items) {
    value.forEach((el, i) => errors.push(...validateAgainstSchema(el, schema.items, `${path}[${i}]`)));
  }
  return errors;
}

// ---------------------------------------------------------------------------------------------
// Concurrency pool — run tasks (thunks returning promises) with a bounded number in flight.
// ---------------------------------------------------------------------------------------------

async function runPool(thunks, limit) {
  const results = new Array(thunks.length);
  let next = 0;
  const workers = new Array(Math.min(limit, thunks.length)).fill(0).map(async () => {
    while (true) {
      const i = next++;
      if (i >= thunks.length) return;
      try {
        results[i] = await thunks[i]();
      } catch (err) {
        results[i] = { __error: err };
      }
    }
  });
  await Promise.all(workers);
  return results;
}

// ---------------------------------------------------------------------------------------------
// The runtime.
// ---------------------------------------------------------------------------------------------

class KiroWorkflowRuntime {
  constructor(config = {}) {
    this.config = config;
    this.kiroBin = config.kiroBin || 'kiro-cli';
    this.cwd = config.cwd || process.cwd();
    this.defaultAgent = config.agent || null;
    this.defaultModel = config.model || null;
    this.defaultEffort = config.effort || null;
    this.trustAllTools = config.trustAllTools !== false; // default true — unattended run
    this.trustTools = config.trustTools || null; // string; overrides trustAllTools when set
    // Version skew: SAIS runs a frozen kiro-cli 2.5.0 snapshot. `--effort` and granular
    // `--trust-tools` are 2.10.0-era flags; set allowNewFlags=false to emit only the 2.5.0-safe set
    // (`--no-interactive`, `--trust-all-tools`, `--agent`, `--model`). Applies to BOTH transports.
    this.allowNewFlags = config.allowNewFlags !== false;
    // ACP entry differs by build (2.10.0 `kiro-cli acp` subcommand vs 2.5.0 `kiro-cli-chat` binary).
    this.acpEntry = config.acpEntry || 'subcommand';
    this.kiroChatBin = config.kiroChatBin || 'kiro-cli-chat';
    this.acpProtocolVersion = config.acpProtocolVersion || 1;
    this.timeoutMs = config.timeoutMs || 15 * 60 * 1000;
    this.maxRetries = config.maxRetries ?? 1; // transport-level retries (spawn/timeout failures)
    this.maxSchemaRetries = config.maxSchemaRetries ?? 3; // re-prompts on schema mismatch (raise via --schema-retries)
    const cores = (os.cpus() || []).length || 4;
    this.concurrency = config.concurrency || Math.max(1, Math.min(16, cores - 2));

    // MCP servers to make available to kiro-cli's own agent (RC#3/RC#5). The post-Resolve phases
    // (Challenge, AlignEdges, validate, check_script, resolve, preflight, DagCheck) are RELAY-TOOL-
    // RESULT prompts: they tell kiro-cli to CALL author.*/validate.*/steps.io/catalog.* and return the
    // tool's JSON. If no cursus MCP server is registered, kiro-cli has no such tools and FABRICATES the
    // JSON — which is array-prone and parse-fragile (the Run-5 failing phases). Registering the server
    // lets those turns return real tool output. Each entry: { name, command, args?, env?, cwd? }.
    // `config.mcpServers` is an explicit list; `config.mcpCursus` (or the derived default) auto-adds the
    // cursus stdio server (`cursus mcp serve`, else `python -m cursus.mcp.server`). Passed to ACP via
    // session/new; for headless, kiro-cli reads servers from its --agent config, so a warning fires if
    // headless is used with servers set but no --agent (see _mcpServers()).
    this.mcpServers = Array.isArray(config.mcpServers) ? config.mcpServers.slice() : [];
    if (config.mcpCursus) {
      const cmd = typeof config.mcpCursus === 'string' ? config.mcpCursus : 'cursus';
      this.mcpServers.push(
        cmd === 'cursus'
          ? { name: 'cursus', command: 'cursus', args: ['mcp', 'serve'] }
          : { name: 'cursus', command: cmd, args: ['-m', 'cursus.mcp.server'] }
      );
    }

    // Transport: 'headless' (default) spawns one `kiro-cli chat --no-interactive` per turn; 'acp'
    // holds one long-lived `kiro-cli acp` process and runs one ACP session per turn (persistent
    // connection, streaming, per-tool permission handling). ACP amortizes process/agent-init cost.
    this.transport = config.transport === 'acp' ? 'acp' : 'headless';
    this._acp = null; // lazily created KiroAcpClient when transport === 'acp'

    this.agentSeq = 0;
    this.totalAgents = 0;
    this.maxTotalAgents = config.maxTotalAgents || 1000;
    this.currentPhase = null;
    this.abort = config.signal || null;

    // Budget: kiro reports credits, not tokens, so this is a best-effort char/4 estimate. total is
    // null unless the caller sets one, so `while (budget.total && ...)` loops never run away.
    this._spent = 0;
    const total = config.budgetTotal ?? null;
    const self = this;
    this.budget = {
      total,
      spent: () => self._spent,
      remaining: () => (total == null ? Infinity : Math.max(0, total - self._spent)),
    };
  }

  // --- progress output goes to STDERR so a captured STDOUT holds only the final workflow result ---
  _emit(line) {
    process.stderr.write(line + '\n');
  }

  phase(title) {
    this.currentPhase = title;
    this._emit(`\n━━ phase: ${title} ━━`);
  }

  log(message) {
    this._emit(`  · ${message}`);
  }

  _estimateTokens(text) {
    return Math.ceil(String(text).length / 4);
  }

  // Dispatch one turn to the active transport. Returns { text, code } or throws on transport error.
  _runTurn(prompt, opts, attempt) {
    if (this.transport === 'acp') return this._runTurnAcp(prompt, opts, attempt);
    return this._runTurnHeadless(prompt, opts, attempt);
  }

  // The MCP servers to expose to kiro-cli, normalized. ACP registers them via session/new (2.10.0
  // only — 2.5.0's ACP CRASHES on that payload, see the ACP client + SAIS Run 8). Headless has no
  // per-call registration at all: kiro-cli reads MCP servers from its --agent config there. So warn
  // once when servers are configured but this run can't actually deliver them, pointing at the ONE
  // route that works on the frozen 2.5.0 build: a static ~/.kiro/settings/mcp.json + headless --agent.
  _mcpServers() {
    if (!this.mcpServers.length) return [];
    // Headless never delivers MCP via the runtime; it must come from --agent config. On the 2.5.0-safe
    // path ACP can't deliver it either. In both cases, emit the mcp.json the user should install.
    const cantDeliver =
      (this.transport === 'headless' && !this.defaultAgent) ||
      (this.transport === 'acp' && !this.allowNewFlags);
    if (cantDeliver && !this._warnedMcpUndeliverable) {
      this._warnedMcpUndeliverable = true;
      const reason =
        this.transport === 'headless'
          ? 'transport=headless has no per-call MCP registration (kiro-cli reads it from --agent config)'
          : 'transport=acp + --legacy-kiro: kiro-cli 2.5.0 ACP crashes on session/new mcpServers (SAIS Run 8)';
      this._emit(
        `  ⚠ ${this.mcpServers.length} MCP server(s) configured but ${reason}. On the frozen 2.5.0 build ` +
          `the WORKING route is a static config + headless --agent. Add to ~/.kiro/settings/mcp.json:\n` +
          `    ${JSON.stringify({ mcpServers: this._mcpJsonShape() })}\n` +
          `  then run: --transport headless --agent <name-bound-to-that-config> (drop --transport acp / --mcp-cursus).`
      );
    }
    return this.mcpServers;
  }

  // The ~/.kiro/settings/mcp.json "mcpServers" object shape for the configured servers (name -> spec).
  _mcpJsonShape() {
    const obj = {};
    for (const s of this.mcpServers) {
      obj[s.name] = { command: s.command, args: Array.isArray(s.args) ? s.args : [] };
      if (s.env && typeof s.env === 'object') obj[s.name].env = s.env;
    }
    return obj;
  }

  // Build the 2.5.0-SAFE static config to tool-force this turn's output via the submit-result MCP
  // server (the analog of the Claude Code StructuredOutput mechanism). Writes the phase schema to a
  // file and returns an mcp.json-shaped object registering submit-result-server.js with that schema +
  // a result-file sink. The caller writes the returned `mcpJson` to ~/.kiro/settings/mcp.json (or an
  // --agent spec) BEFORE spawning kiro-cli headless, and after the turn reads `resultFile` for the
  // validated structured payload. This NEVER uses dynamic ACP session/new (which crashes 2.5.0, Run 8).
  //
  // NOTE: this is the OFFLINE-SAFE builder only. It is NOT yet wired into agent() as the primary
  // schema-forcing path — that waits on a SAIS probe confirming kiro-cli 2.5.0 headless actually reads
  // the static config, runs the tool loop, and emits the submit_result call (see README "Tool-forcing").
  buildSubmitResultConfig(schema, paths, opts = {}) {
    const toolName = opts.toolName || 'submit_result';
    const serverName = opts.serverName || 'cursus-submit';
    const serverPath = opts.serverPath || require('node:path').join(__dirname, 'submit-result-server.js');
    const nodeBin = opts.nodeBin || process.execPath; // absolute node path — robust under a bare PATH
    const env = {
      SUBMIT_SCHEMA_FILE: paths.schemaFile,
      SUBMIT_RESULT_FILE: paths.resultFile,
      SUBMIT_TOOL_NAME: toolName,
    };
    const mcpJson = { mcpServers: { [serverName]: { command: nodeBin, args: [serverPath], env } } };
    return {
      mcpJson,
      env,
      serverName,
      toolName,
      // The line to append to the phase prompt so the model calls the tool instead of emitting prose.
      promptSuffix:
        '\n\n---\nDo NOT print your answer as text. Instead CALL the `' + toolName + '` tool EXACTLY ONCE, ' +
        'passing your answer as its arguments; the arguments MUST conform to that tool\'s inputSchema. ' +
        'The tool validates your arguments and records the result.',
      // Convenience: write the schema file now (the server reads it via SUBMIT_SCHEMA_FILE).
      writeSchema: () => require('node:fs').writeFileSync(paths.schemaFile, JSON.stringify(schema)),
    };
  }

  // Lazily create + start the shared ACP client (one long-lived `kiro-cli acp` process).
  async _ensureAcp() {
    if (!this._acp) {
      this._acp = new KiroAcpClient({
        kiroBin: this.kiroBin,
        kiroChatBin: this.kiroChatBin,
        acpEntry: this.acpEntry,
        protocolVersion: this.acpProtocolVersion,
        allowNewFlags: this.allowNewFlags,
        cwd: this.cwd,
        agent: this.defaultAgent,
        model: this.defaultModel,
        effort: this.defaultEffort,
        trustAllTools: this.trustAllTools,
        trustTools: this.trustTools,
        mcpServers: this._mcpServers(), // RC#3/RC#5: register cursus MCP tools for relay-tool-result turns
        promptTimeoutMs: this.timeoutMs,
        onLog: (m) => this._emit(`  · ${m}`),
      });
      await this._acp.start();
    }
    return this._acp;
  }

  // Run one turn over the persistent ACP session. Note: --agent/--model/--effort are session-level
  // (set once at ACP start), so per-call opts.model/opts.effort are honored only in headless mode.
  async _runTurnAcp(prompt, opts, attempt) {
    if (this.abort && this.abort.aborted) throw new Error('aborted');
    const client = await this._ensureAcp();
    const { text, stopReason } = await client.prompt(prompt);
    // Map ACP stop reasons to a headless-like { text, code } shape (0 = ok).
    const ok = stopReason === 'end_turn' || stopReason === 'max_tokens' || stopReason === 'unknown';
    return { text, rawOut: text, stderr: '', code: ok ? 0 : 1, stopReason };
  }

  // Run one kiro-cli headless turn. Returns { text, stderr, code } or throws on spawn/timeout error.
  _runTurnHeadless(prompt, opts, attempt) {
    return new Promise((resolve, reject) => {
      const args = ['chat', '--no-interactive'];
      // `--trust-all-tools` and `--agent` are 2.5.0-safe; granular `--trust-tools` + `--effort` are
      // 2.10.0-era, so emit them only when allowNewFlags (avoids "unexpected argument" on the frozen build).
      if (this.trustTools != null && this.allowNewFlags) args.push(`--trust-tools=${this.trustTools}`);
      else if (this.trustAllTools) args.push('--trust-all-tools');
      const agentName = opts.agentType || opts.agent || this.defaultAgent;
      if (agentName) args.push('--agent', agentName);
      const model = opts.model || this.defaultModel;
      if (model) args.push('--model', model);
      const effort = opts.effort || this.defaultEffort;
      if (effort && this.allowNewFlags) args.push('--effort', effort);
      else if (effort && !this.allowNewFlags && !this._warnedEffortDropped) {
        // A step explicitly asked for a reasoning effort (e.g. Resolve requests 'high'), but the
        // 2.5.0-safe flag set drops --effort, so this turn runs at the build's default/Auto model —
        // the strong-model steps are silently under-powered. Warn ONCE; this is the likeliest cause
        // of a schema turn that returns prose instead of JSON on the frozen build.
        this._warnedEffortDropped = true;
        this._emit(
          `  ⚠ a step requested effort='${effort}' but --legacy-kiro drops --effort (kiro-cli 2.5.0 ` +
            `has no --effort flag) — that step runs at the build default and may fail strict-JSON/` +
            `schema turns. For a fair run, use kiro-cli >= 2.10.0 without --legacy-kiro.`
        );
      }
      args.push(prompt); // INPUT positional — kiro does not read the prompt from stdin

      const child = spawn(this.kiroBin, args, {
        cwd: opts.cwd || this.cwd,
        env: process.env,
        stdio: ['ignore', 'pipe', 'pipe'],
      });

      let out = '';
      let err = '';
      let settled = false;
      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        child.kill('SIGKILL');
        reject(new Error(`kiro-cli turn timed out after ${this.timeoutMs}ms (attempt ${attempt})`));
      }, this.timeoutMs);

      const onAbort = () => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        child.kill('SIGKILL');
        reject(new Error('aborted'));
      };
      if (this.abort) {
        if (this.abort.aborted) return onAbort();
        this.abort.addEventListener('abort', onAbort, { once: true });
      }

      child.stdout.on('data', (d) => (out += d.toString()));
      child.stderr.on('data', (d) => (err += d.toString()));
      child.on('error', (e) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        reject(e);
      });
      child.on('close', (code) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (this.abort) this.abort.removeEventListener('abort', onAbort);
        resolve({ text: cleanReply(out), rawOut: out, stderr: err, code });
      });
    });
  }

  // The `agent()` primitive. Without a schema returns the reply text; with a schema returns the
  // validated object (re-prompting on mismatch). Returns null on terminal failure — matching CC,
  // where a dead agent resolves to null so callers can .filter(Boolean).
  async agent(prompt, opts = {}) {
    if (this.totalAgents >= this.maxTotalAgents) {
      throw new Error(`agent cap reached (${this.maxTotalAgents}) — runaway-loop backstop`);
    }
    if (this.budget.total != null && this.budget.remaining() <= 0) {
      throw new Error(`token budget exhausted (${this.budget.total})`);
    }
    this.totalAgents++;
    const id = ++this.agentSeq;
    const label = opts.label || `agent-${id}`;
    const phaseName = opts.phase || this.currentPhase || '-';
    this._emit(`  ▸ [${phaseName}] ${label} …`);
    this._spent += this._estimateTokens(prompt);

    const schema = opts.schema || null;
    let finalPrompt = prompt;
    if (schema) {
      // Lead with the explicit shape directive (words + skeleton), THEN the schema. The directive is
      // on the initial prompt — not just re-prompts — because the array-wrap bias shows on attempt 1.
      finalPrompt =
        prompt +
        '\n\n---\n' +
        shapeDirective(schema) +
        '\nNo prose, no explanation, no markdown code fence. The JSON must conform to this JSON Schema:\n' +
        JSON.stringify(schema);
    }

    // Outer loop = schema re-prompts; inner loop = transport retries.
    for (let sAttempt = 0; sAttempt <= (schema ? this.maxSchemaRetries : 0); sAttempt++) {
      let turn = null;
      for (let tAttempt = 0; tAttempt <= this.maxRetries; tAttempt++) {
        try {
          turn = await this._runTurn(finalPrompt, opts, tAttempt);
          break;
        } catch (e) {
          if (e.message === 'aborted') return null;
          if (tAttempt >= this.maxRetries) {
            this._emit(`  ✗ ${label}: transport failed — ${e.message}`);
            return null;
          }
          this._emit(`  ↻ ${label}: retry ${tAttempt + 1} (${e.message})`);
        }
      }
      if (!turn) return null;
      this._spent += this._estimateTokens(turn.text);

      if (turn.code !== 0 && !turn.text) {
        this._emit(`  ✗ ${label}: kiro-cli exit ${turn.code}`);
        return null;
      }

      if (!schema) {
        this._emit(`  ✓ ${label}`);
        return turn.text;
      }

      // Build a corrective re-prompt (used by every failure branch so the retry is not identical).
      // Re-state the full shape directive (words + skeleton) each time, not a generic "single value".
      const reprompt = (reasonLines) =>
        prompt +
        '\n\n---\nYour previous reply was rejected: ' +
        reasonLines +
        '\n' +
        shapeDirective(schema) +
        '\nNo prose, no markdown fence. Conform to this JSON Schema:\n' +
        JSON.stringify(schema);

      const jsonText = extractJsonText(turn.text);
      if (jsonText) {
        // Tolerant parse: strict JSON.parse first (valid JSON is returned untouched), then SAFE
        // syntax-level repairs for the JSON5-ish output kiro-cli 2.5.0 emits (trailing commas,
        // comments, single/smart quotes, Python True/False/None, unquoted keys). RC#3 parse-failure half.
        const tp = tolerantParse(jsonText);
        if (tp.value !== undefined) {
          if (tp.repaired) this._emit(`  · ${label}: repaired JSON (${tp.repairs.join('; ')})`);
          // Normalize a common shape mismatch (e.g. single-element array when an object is wanted)
          // before validating — models can get stuck emitting the wrong container across re-prompts.
          const co = coerceToSchema(tp.value, schema);
          const parsed = co.value;
          if (co.coerced) this._emit(`  · ${label}: ${co.note}`);
          const errors = validateAgainstSchema(parsed, schema);
          if (errors.length === 0) {
            this._emit(`  ✓ ${label}`);
            return parsed;
          }
          if (sAttempt < this.maxSchemaRetries) {
            // If the top-level container is wrong, say so LOUDLY and concretely — a generic
            // "expected object, got array" is what the model ignored last time. Name the exact case:
            // an OBJECT schema answered with an N-element array is the model returning a LIST OF
            // ALTERNATIVES; tell it to pick one.
            const wantType = Array.isArray(schema.type) ? schema.type.join('|') : schema.type;
            let shapeHint = '';
            if (wantType && typeOf(parsed) !== wantType && (wantType === 'object' || wantType === 'array')) {
              if (wantType === 'object' && typeOf(parsed) === 'array') {
                shapeHint =
                  `\nCRITICAL: you returned a JSON ARRAY of ${parsed.length} item(s), but the answer must be ` +
                  'EXACTLY ONE JSON object. This is not a list task — do NOT return multiple candidates or ' +
                  'alternatives. Pick the single best answer and return ONLY that one object: your reply must ' +
                  'start with `{` and end with `}`, with no surrounding `[ ]`.';
              } else if (wantType === 'array') {
                shapeHint = '\nCRITICAL: the TOP-LEVEL value must be a JSON array `[...]`. You returned a ' + typeOf(parsed) + '.';
              }
            }
            this._emit(`  ↻ ${label}: schema mismatch, re-prompting (${errors[0]})`);
            finalPrompt = reprompt(
              'it did not satisfy the schema. Errors:\n' +
                errors.slice(0, 8).map((e) => '  - ' + e).join('\n') +
                shapeHint
            );
            continue;
          }
          this._emit(`  ✗ ${label}: schema still invalid after ${this.maxSchemaRetries} re-prompts`);
          return null;
        }
        // Even tolerant parse failed. Name the likely defects (if the repair pass recognized any) so
        // the re-prompt is specific instead of echoing an opaque JS SyntaxError the model just repeats.
        if (sAttempt < this.maxSchemaRetries) {
          const defectHint = tp.repairs.length
            ? ' Likely issue(s) in your JSON: ' + tp.repairs.join('; ') +
              '. Emit STRICT JSON only: double-quoted keys and strings, no trailing commas, no comments, ' +
              'no single or “smart” quotes, and use true/false/null (not True/False/None).'
            : ' Emit STRICT, valid JSON only — double-quoted keys/strings, no trailing commas, no comments.';
          this._emit(`  ↻ ${label}: JSON parse failed, re-prompting${tp.repairs.length ? ' (' + tp.repairs.join('; ') + ')' : ''}`);
          finalPrompt = reprompt('the JSON could not be parsed (' + tp.error + ').' + defectHint);
          continue;
        }
        this._emit(`  ✗ ${label}: unparseable JSON — ${tp.error}`);
        return null;
      }
      if (sAttempt < this.maxSchemaRetries) {
        this._emit(`  ↻ ${label}: no JSON found in reply, re-prompting`);
        finalPrompt = reprompt('no JSON value was found in the reply (it contained prose only).');
        continue;
      }
      this._emit(`  ✗ ${label}: no JSON value in reply`);
      return null;
    }
    return null;
  }

  // parallel(thunks) — BARRIER fan-out; a throwing/dead thunk resolves to null (never rejects).
  async parallel(thunks) {
    const list = Array.from(thunks || []);
    if (list.length > 4096) throw new Error(`parallel(): ${list.length} tasks exceeds 4096 cap`);
    const raw = await runPool(list, this.concurrency);
    return raw.map((r) => (r && r.__error ? null : r));
  }

  // pipeline(items, ...stages) — each item flows through all stages independently (no barrier
  // between stages); items run concurrently, capped. A stage that throws drops that item to null.
  async pipeline(items, ...stages) {
    const list = Array.from(items || []);
    if (list.length > 4096) throw new Error(`pipeline(): ${list.length} items exceeds 4096 cap`);
    const tasks = list.map((item, index) => async () => {
      let acc = item;
      for (const stage of stages) {
        try {
          acc = await stage(acc, item, index);
        } catch (e) {
          this._emit(`  ✗ pipeline item ${index}: stage threw — ${e.message}`);
          return null;
        }
      }
      return acc;
    });
    const raw = await runPool(tasks, this.concurrency);
    return raw.map((r) => (r && r.__error ? null : r));
  }

  // Release transport resources (the long-lived ACP process, if any). Safe to call always.
  async close() {
    if (this._acp) {
      await this._acp.close();
      this._acp = null;
    }
  }
}

module.exports = {
  KiroWorkflowRuntime,
  // exported for reuse / unit tests:
  stripAnsi,
  cleanReply,
  extractJsonText,
  tolerantParse,
  shapeDirective,
  coerceToSchema,
  validateAgainstSchema,
  runPool,
};
