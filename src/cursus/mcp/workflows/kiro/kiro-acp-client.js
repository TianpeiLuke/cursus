// kiro-acp-client.js — a dependency-free JS ACP client that drives `kiro-cli acp`.
//
// WHY
// ---
// The sibling runtime (kiro-workflow-runtime.js) drives kiro-cli HEADLESS: one
// `kiro-cli chat --no-interactive` process per agent() call. That is simplest, but every call pays
// full process-start + agent-init cost and there is no live streaming. This module is the ACP
// alternative: it spawns ONE long-lived `kiro-cli acp` process and speaks the Agent Client Protocol
// (JSON-RPC 2.0 over newline-delimited stdio) to it — initialize once, then one ACP session per
// agent() turn. This is the same mechanism verified in internal JS clients (A2AServer4KiroCLI,
// KiroOverlayApp, AGIArsenalConsole); here it is hand-rolled (no @agentclientprotocol/sdk dependency),
// mirroring AGIArsenalConsole's raw-JSON-RPC approach.
//
// VERSION SKEW — IMPORTANT: this was captured on kiro-cli 2.10.0, but SAIS runs a FROZEN 2.5.0
// snapshot. Two things differ across builds and are made configurable here:
//   (1) ACP entry point — 2.10.0 exposes the `kiro-cli acp` SUBCOMMAND; the 2.5.0 snapshot ships a
//       separate `kiro-cli-chat` binary that IS the ACP server (per the vault repo_kiro_cli note).
//       Select with opts.acpEntry: 'subcommand' (default) | 'chat-binary' (+ opts.kiroChatBin).
//   (2) protocolVersion — requested via opts.protocolVersion (default 1) and RE-NEGOTIATED to
//       whatever the server echoes in the initialize result.
//   Also: `--effort` and granular `--trust-tools` are 2.10.0-era flags; set opts.allowNewFlags=false
//   to emit only the 2.5.0-safe set (`--trust-all-tools`, `--agent`, `--model`).
//   If ACP is entirely absent/broken on the frozen build, use the HEADLESS transport instead.
//
// WIRE CONTRACT (captured live from kiro-cli 2.10.0, protocolVersion 1):
//   -> initialize {protocolVersion:1, clientCapabilities:{fs:{readTextFile,writeTextFile}}}
//        <- result {protocolVersion, agentInfo, agentCapabilities, ...}
//   -> session/new {cwd, mcpServers:[]}                    <- result {sessionId, modes}
//   -> session/prompt {sessionId, prompt:[{type:'text',text}]}
//        <- session/update notifications: {sessionId, update:{sessionUpdate:'agent_message_chunk',
//                                            content:{type:'text', text}}}   (accumulate the text)
//        <- result {stopReason:'end_turn'|'max_tokens'|'refusal'|'cancelled'|...}
//   <- session/request_permission {sessionId, options:[{optionId,name,kind}]}
//        -> result {outcome:{outcome:'selected', optionId}}   (auto-approve for unattended runs)
//   Ignore every notification whose method starts with '_kiro.dev/'.
//
// Note: the live agent nested the chunk under params.update in one build and flat in params in
// another; this reader handles both (params.update ?? params).

'use strict';

const { spawn } = require('node:child_process');

const PROTOCOL_VERSION = 1;

class KiroAcpClient {
  constructor(opts = {}) {
    this.kiroBin = opts.kiroBin || 'kiro-cli';
    this.cwd = opts.cwd || process.cwd();
    this.agent = opts.agent || null;
    this.model = opts.model || null;
    this.effort = opts.effort || null;
    this.trustAllTools = opts.trustAllTools !== false;
    this.trustTools = opts.trustTools || null;
    this.promptTimeoutMs = opts.promptTimeoutMs || 15 * 60 * 1000;
    this.startTimeoutMs = opts.startTimeoutMs || 60 * 1000;
    this.onLog = opts.onLog || (() => {});

    // --- Version-skew handling (SAIS runs a frozen kiro-cli 2.5.0 snapshot; laptop is 2.10.0) ---
    // How ACP is entered differs by build. Newer builds expose the `kiro-cli acp` SUBCOMMAND; the
    // 2.5.0 snapshot ships a separate `kiro-cli-chat` binary that IS the ACP server (see the vault
    // repo_kiro_cli note). `acpEntry` selects the shape:
    //   'subcommand' (default): spawn <kiroBin> with args ['acp', ...]      (2.10.0-style)
    //   'chat-binary':          spawn <kiroChatBin> with args [...] (no 'acp' arg)  (2.5.0-style)
    this.acpEntry = opts.acpEntry || 'subcommand';
    this.kiroChatBin = opts.kiroChatBin || 'kiro-cli-chat';
    // Protocol version to REQUEST in initialize. The server echoes the version it will speak; if it
    // returns a different one we adopt it (best-effort negotiation) rather than assuming 1.
    this.protocolVersion = opts.protocolVersion || PROTOCOL_VERSION;
    this.negotiatedProtocol = null;
    // `--effort` and granular `--trust-tools` are 2.10.0-era flags absent in 2.5.0. They are only
    // emitted when explicitly provided AND allowed; `--trust-all-tools` and `--agent` are safe on 2.5.0.
    this.allowNewFlags = opts.allowNewFlags !== false; // set false to guarantee a 2.5.0-safe arg set

    this.child = null;
    this._buf = '';
    this._nextId = 1;
    this._pending = new Map(); // id -> {resolve, reject}
    this._started = false;
    this._closed = false;
    this._agentInfo = null;
    // Serialize prompts: kiro-cli ACP handles one prompt turn at a time per session, and we use one
    // session per turn, so we run turns sequentially through this chain (no interleaved stdin writes).
    this._turnChain = Promise.resolve();
  }

  // Returns { bin, args }. `acpEntry` decides the binary + whether the leading 'acp' arg is present.
  _spawnTarget() {
    const chatBinary = this.acpEntry === 'chat-binary';
    const bin = chatBinary ? this.kiroChatBin : this.kiroBin;
    const args = chatBinary ? [] : ['acp'];
    // Trust: `--trust-all-tools` is present on 2.5.0; granular `--trust-tools` is newer, so only emit
    // it when new flags are allowed.
    if (this.trustTools != null && this.allowNewFlags) args.push(`--trust-tools=${this.trustTools}`);
    else if (this.trustAllTools) args.push('--trust-all-tools');
    if (this.agent) args.push('--agent', this.agent); // `--agent` is safe on 2.5.0
    if (this.model) args.push('--model', this.model);
    // `--effort` is 2.10.0-era; skip it on old builds to avoid an "unexpected argument" hard-fail.
    if (this.effort && this.allowNewFlags) args.push('--effort', this.effort);
    else if (this.effort && !this.allowNewFlags) {
      // The ACP session's effort is dropped on the 2.5.0-safe path, so every turn runs at the build
      // default — strong-model steps (e.g. Resolve, requested at effort='high') may fail schema turns.
      this.onLog(
        `⚠ effort='${this.effort}' dropped (kiro-cli 2.5.0 has no --effort); ACP turns run at the ` +
          `build default and may fail strict-JSON turns — use kiro-cli >= 2.10.0 for a fair run`
      );
    }
    return { bin, args };
  }

  async start() {
    if (this._started) return;
    const { bin, args } = this._spawnTarget();
    this.child = spawn(bin, args, {
      cwd: this.cwd,
      env: process.env,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    this.child.stdout.on('data', (d) => this._onData(d));
    this.child.stderr.on('data', () => {}); // ACP logs go to stderr; ignore
    this.child.on('close', (code) => this._onClose(code));
    this.child.on('error', (e) => this._failAll(e));

    const initResult = await this._request(
      'initialize',
      {
        protocolVersion: this.protocolVersion,
        clientCapabilities: { fs: { readTextFile: true, writeTextFile: true } },
      },
      this.startTimeoutMs
    );
    this._agentInfo = initResult && initResult.agentInfo;
    // Adopt the server's echoed protocol version if it differs (best-effort negotiation across builds).
    if (initResult && typeof initResult.protocolVersion !== 'undefined') {
      this.negotiatedProtocol = initResult.protocolVersion;
      if (initResult.protocolVersion !== this.protocolVersion) {
        this.onLog(
          `ACP protocol: requested ${this.protocolVersion}, server speaks ${initResult.protocolVersion} — adopting`
        );
      }
    }
    this._started = true;
    this.onLog(
      `ACP connected: ${(this._agentInfo && this._agentInfo.name) || 'kiro-cli'} ` +
        `v${(this._agentInfo && this._agentInfo.version) || '?'} (protocol ${
          this.negotiatedProtocol != null ? this.negotiatedProtocol : '?'
        }, entry ${this.acpEntry})`
    );
  }

  _onData(chunk) {
    this._buf += chunk.toString();
    let i;
    while ((i = this._buf.indexOf('\n')) >= 0) {
      const line = this._buf.slice(0, i);
      this._buf = this._buf.slice(i + 1);
      if (!line.trim()) continue;
      let msg;
      try {
        msg = JSON.parse(line);
      } catch {
        continue; // non-JSON line (shouldn't happen on stdout, but be defensive)
      }
      this._dispatch(msg);
    }
  }

  _dispatch(msg) {
    // Response to one of our requests.
    if (msg.id != null && (msg.result !== undefined || msg.error !== undefined)) {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
        if (msg.error) p.reject(new Error(`ACP error ${msg.error.code}: ${msg.error.message}`));
        else p.resolve(msg.result);
      }
      return;
    }
    // Server -> client REQUEST (has method + id): permission / fs.
    if (msg.method && msg.id != null) {
      this._handleServerRequest(msg);
      return;
    }
    // Notification (method, no id): session/update or _kiro.dev/*.
    if (msg.method && msg.id == null) {
      if (msg.method === 'session/update' && this._activeTurn) {
        this._activeTurn.onUpdate(msg.params);
      }
      // _kiro.dev/* and other notifications are intentionally ignored.
    }
  }

  _handleServerRequest(msg) {
    if (msg.method === 'session/request_permission') {
      const options = (msg.params && msg.params.options) || [];
      // Prefer an "allow"/"allow_always"-flavoured option; else the first.
      const pick =
        options.find((o) => /allow/i.test(o.optionId || '') || /allow/i.test(o.kind || '')) ||
        options[0];
      const optionId = pick ? pick.optionId : 'allow';
      this._reply(msg.id, { outcome: { outcome: 'selected', optionId } });
      return;
    }
    if (msg.method && /^fs\/(read|write)/.test(msg.method)) {
      // We advertise fs capability; give benign responses (the workflows use tools, not client fs).
      this._reply(msg.id, msg.method.includes('read') ? { content: '' } : {});
      return;
    }
    // Unknown server request: reply with a JSON-RPC "method not found" so the agent isn't left hanging.
    this._replyError(msg.id, -32601, `method not handled by client: ${msg.method}`);
  }

  _write(obj) {
    if (this._closed || !this.child) throw new Error('ACP client is closed');
    this.child.stdin.write(JSON.stringify(obj) + '\n');
  }

  _reply(id, result) {
    this._write({ jsonrpc: '2.0', id, result });
  }

  _replyError(id, code, message) {
    this._write({ jsonrpc: '2.0', id, error: { code, message } });
  }

  _request(method, params, timeoutMs) {
    const id = this._nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (this._pending.has(id)) {
          this._pending.delete(id);
          reject(new Error(`ACP request '${method}' timed out after ${timeoutMs}ms`));
        }
      }, timeoutMs);
      this._pending.set(id, {
        resolve: (v) => {
          clearTimeout(timer);
          resolve(v);
        },
        reject: (e) => {
          clearTimeout(timer);
          reject(e);
        },
      });
      try {
        this._write({ jsonrpc: '2.0', id, method, params });
      } catch (e) {
        clearTimeout(timer);
        this._pending.delete(id);
        reject(e);
      }
    });
  }

  // Run one prompt turn: new session -> prompt -> accumulate agent_message_chunk text -> stopReason.
  // Serialized so only one turn writes to stdin at a time.
  prompt(text) {
    const run = async () => {
      if (!this._started) await this.start();
      const session = await this._request('session/new', { cwd: this.cwd, mcpServers: [] }, this.startTimeoutMs);
      const sessionId = session && session.sessionId;
      if (!sessionId) throw new Error('session/new returned no sessionId');

      let acc = '';
      this._activeTurn = {
        sessionId,
        onUpdate: (params) => {
          if (!params) return;
          if (params.sessionId && params.sessionId !== sessionId) return;
          const u = params.update || params;
          if (u && u.sessionUpdate === 'agent_message_chunk' && u.content && u.content.type === 'text') {
            acc += u.content.text;
          }
        },
      };
      try {
        const result = await this._request(
          'session/prompt',
          { sessionId, prompt: [{ type: 'text', text }] },
          this.promptTimeoutMs
        );
        const stopReason = (result && result.stopReason) || 'unknown';
        return { text: acc.trim(), stopReason };
      } finally {
        this._activeTurn = null;
      }
    };
    // chain to serialize; isolate failures so one bad turn doesn't poison the chain
    const p = this._turnChain.then(run, run);
    this._turnChain = p.then(
      () => undefined,
      () => undefined
    );
    return p;
  }

  _onClose(code) {
    this._closed = true;
    this._failAll(new Error(`kiro-cli acp exited (code ${code})`));
  }

  _failAll(err) {
    for (const [, p] of this._pending) p.reject(err);
    this._pending.clear();
  }

  async close() {
    if (this._closed || !this.child) return;
    this._closed = true;
    try {
      this.child.stdin.end();
    } catch {
      /* ignore */
    }
    this.child.kill('SIGTERM');
    // give it a moment, then hard-kill
    await new Promise((r) => setTimeout(r, 200));
    try {
      this.child.kill('SIGKILL');
    } catch {
      /* ignore */
    }
  }
}

module.exports = { KiroAcpClient, PROTOCOL_VERSION };
