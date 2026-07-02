#!/usr/bin/env node
// submit-result-server.js — a tiny, dependency-free stdio MCP server that exposes ONE tool,
// `submit_result`, whose inputSchema IS a workflow phase's JSON schema. It is the Kiro runtime's
// analog of the Claude Code host's StructuredOutput tool-forcing: instead of asking the model for JSON
// in prose and salvaging the reply (shapeDirective -> tolerantParse -> coerceToSchema -> re-prompt),
// the model is told to CALL submit_result with its answer; kiro-cli marshals that tool call to THIS
// server, which validates the arguments against the schema and writes them to a result file. The
// runtime then reads that file as the structured result — no stdout scraping, no array-bias, because
// the payload arrives as a validated JSON-RPC object, not free text.
//
// WHY A SEPARATE SERVER (not the cursus MCP server): the cursus server exposes the author.*/validate.*
// tool surface; this one exposes a single per-phase result SINK whose schema changes every turn. It is
// registered the 2.5.0-safe way — a STATIC ~/.kiro/settings/mcp.json (or an --agent spec) written
// BEFORE kiro-cli spawns — never via dynamic ACP session/new (which crashes kiro-cli 2.5.0, SAIS Run 8).
//
// PROTOCOL: MCP over newline-delimited JSON-RPC 2.0 on stdio. Handles initialize, notifications/
// initialized, tools/list, tools/call, ping. This is the same wire shape the probe's echo server used
// and that the `mcp` Python SDK speaks (initialize -> initialized -> tools/list -> tools/call).
//
// CONFIG (via env, so a static mcp.json can set it per phase):
//   SUBMIT_SCHEMA_FILE   path to a JSON file holding the phase's JSON schema (the tool inputSchema)
//   SUBMIT_RESULT_FILE   path this server writes the validated arguments to (the runtime reads it)
//   SUBMIT_TOOL_NAME     tool name (default "submit_result")
//   SUBMIT_TOOL_DESC     tool description (optional)
//
// This file touches no live kiro-cli and cannot crash SAIS: it is a passive stdio server, exercised
// here only by the offline mock harness until a SAIS probe confirms 2.5.0 headless invokes it.

'use strict';

const fs = require('node:fs');
const { validateAgainstSchema } = require('./kiro-workflow-runtime');

const TOOL_NAME = process.env.SUBMIT_TOOL_NAME || 'submit_result';
const TOOL_DESC =
  process.env.SUBMIT_TOOL_DESC ||
  'Submit your final answer for this step. Call this tool exactly once with the answer as its ' +
    'arguments; the arguments MUST conform to this tool\'s inputSchema. Do not also print the answer as text.';
const SCHEMA_FILE = process.env.SUBMIT_SCHEMA_FILE || null;
const RESULT_FILE = process.env.SUBMIT_RESULT_FILE || null;
const PROTOCOL_VERSION = process.env.SUBMIT_PROTOCOL || '2024-11-05';

// Load the phase schema (the tool's inputSchema). If none is configured, accept any object so the
// server still functions (the runtime always sets SUBMIT_SCHEMA_FILE in practice).
function loadSchema() {
  if (SCHEMA_FILE && fs.existsSync(SCHEMA_FILE)) {
    try {
      return JSON.parse(fs.readFileSync(SCHEMA_FILE, 'utf8'));
    } catch (e) {
      return { type: 'object' };
    }
  }
  return { type: 'object' };
}

function write(obj) {
  process.stdout.write(JSON.stringify(obj) + '\n');
}

function reply(id, result) {
  write({ jsonrpc: '2.0', id, result });
}

function replyError(id, code, message) {
  write({ jsonrpc: '2.0', id, error: { code, message } });
}

function handle(msg) {
  const { id, method, params } = msg;
  // Notifications (no id) — acknowledge silently.
  if (method === 'notifications/initialized' || method === 'initialized') return;
  if (method === 'notifications/cancelled') return;

  if (method === 'initialize') {
    reply(id, {
      protocolVersion: (params && params.protocolVersion) || PROTOCOL_VERSION,
      capabilities: { tools: {} },
      serverInfo: { name: 'cursus-submit-result', version: '1.0.0' },
    });
    return;
  }

  if (method === 'ping') {
    reply(id, {});
    return;
  }

  if (method === 'tools/list') {
    reply(id, {
      tools: [{ name: TOOL_NAME, description: TOOL_DESC, inputSchema: loadSchema() }],
    });
    return;
  }

  if (method === 'tools/call') {
    const name = params && params.name;
    const args = (params && params.arguments) || {};
    if (name !== TOOL_NAME) {
      replyError(id, -32602, `unknown tool: ${name}`);
      return;
    }
    // Validate against the phase schema BEFORE accepting — this is the tool-forcing guarantee: a
    // schema-violating payload is rejected here (as an isError tool result the model can react to),
    // exactly as the CC host validates StructuredOutput before returning it.
    const errors = validateAgainstSchema(args, loadSchema());
    if (errors.length) {
      reply(id, {
        content: [
          {
            type: 'text',
            text:
              'REJECTED: the arguments did not conform to the inputSchema. Fix and call ' +
              TOOL_NAME +
              ' again. Errors:\n' +
              errors.slice(0, 8).map((e) => '  - ' + e).join('\n'),
          },
        ],
        isError: true,
      });
      return;
    }
    // Valid — persist the structured payload for the runtime to read, then ack.
    if (RESULT_FILE) {
      try {
        fs.writeFileSync(RESULT_FILE, JSON.stringify(args));
      } catch (e) {
        replyError(id, -32000, 'could not write result file: ' + e.message);
        return;
      }
    }
    reply(id, { content: [{ type: 'text', text: 'Result recorded. You are done with this step.' }] });
    return;
  }

  // Unknown method — respond with a JSON-RPC method-not-found rather than crashing.
  if (id != null) replyError(id, -32601, `method not found: ${method}`);
}

function main() {
  let buf = '';
  process.stdin.on('data', (d) => {
    buf += d.toString();
    let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i);
      buf = buf.slice(i + 1);
      if (!line.trim()) continue;
      let msg;
      try {
        msg = JSON.parse(line);
      } catch (e) {
        continue; // ignore non-JSON lines
      }
      try {
        handle(msg);
      } catch (e) {
        if (msg && msg.id != null) replyError(msg.id, -32000, 'internal error: ' + e.message);
      }
    }
  });
  process.stdin.on('end', () => process.exit(0));
}

if (require.main === module) main();

module.exports = { handle, loadSchema, TOOL_NAME };
