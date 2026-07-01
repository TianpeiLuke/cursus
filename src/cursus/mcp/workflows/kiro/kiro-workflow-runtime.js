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
    this.timeoutMs = config.timeoutMs || 15 * 60 * 1000;
    this.maxRetries = config.maxRetries ?? 1; // transport-level retries (spawn/timeout failures)
    this.maxSchemaRetries = config.maxSchemaRetries ?? 2; // re-prompts on schema mismatch
    const cores = (os.cpus() || []).length || 4;
    this.concurrency = config.concurrency || Math.max(1, Math.min(16, cores - 2));

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

  // Run one kiro-cli headless turn. Returns { text, stderr, code } or throws on spawn/timeout error.
  _runTurn(prompt, opts, attempt) {
    return new Promise((resolve, reject) => {
      const args = ['chat', '--no-interactive'];
      if (this.trustTools != null) args.push(`--trust-tools=${this.trustTools}`);
      else if (this.trustAllTools) args.push('--trust-all-tools');
      const agentName = opts.agentType || opts.agent || this.defaultAgent;
      if (agentName) args.push('--agent', agentName);
      const model = opts.model || this.defaultModel;
      if (model) args.push('--model', model);
      const effort = opts.effort || this.defaultEffort;
      if (effort) args.push('--effort', effort);
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
      finalPrompt =
        prompt +
        '\n\n---\nRESPOND WITH A SINGLE JSON VALUE AND NOTHING ELSE. No prose, no explanation, ' +
        'no markdown code fence. The JSON must conform to this JSON Schema:\n' +
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
      const reprompt = (reasonLines) =>
        prompt +
        '\n\n---\nYour previous reply was rejected: ' +
        reasonLines +
        '\nRespond AGAIN with a SINGLE valid JSON value and NOTHING else — no prose, no markdown ' +
        'fence — conforming to this JSON Schema:\n' +
        JSON.stringify(schema);

      const jsonText = extractJsonText(turn.text);
      if (jsonText) {
        try {
          const parsed = JSON.parse(jsonText);
          const errors = validateAgainstSchema(parsed, schema);
          if (errors.length === 0) {
            this._emit(`  ✓ ${label}`);
            return parsed;
          }
          if (sAttempt < this.maxSchemaRetries) {
            this._emit(`  ↻ ${label}: schema mismatch, re-prompting (${errors[0]})`);
            finalPrompt = reprompt(
              'it did not satisfy the schema. Errors:\n' +
                errors.slice(0, 8).map((e) => '  - ' + e).join('\n')
            );
            continue;
          }
          this._emit(`  ✗ ${label}: schema still invalid after ${this.maxSchemaRetries} re-prompts`);
          return null;
        } catch (e) {
          if (sAttempt < this.maxSchemaRetries) {
            this._emit(`  ↻ ${label}: JSON parse failed, re-prompting`);
            finalPrompt = reprompt('the JSON could not be parsed (' + e.message + ').');
            continue;
          }
          this._emit(`  ✗ ${label}: unparseable JSON — ${e.message}`);
          return null;
        }
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
}

module.exports = {
  KiroWorkflowRuntime,
  // exported for reuse / unit tests:
  stripAnsi,
  cleanReply,
  extractJsonText,
  validateAgainstSchema,
  runPool,
};
