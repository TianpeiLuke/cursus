#!/usr/bin/env node
// test-runtime.js — offline tests for the Kiro workflow runtime. No live kiro-cli needed: a mock
// binary stands in for it, so this is fast, free, and deterministic. Run: `node test-runtime.js`.

'use strict';

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const {
  cleanReply,
  extractJsonText,
  tolerantParse,
  shapeDirective,
  coerceToSchema,
  validateAgainstSchema,
  runPool,
} = require('./kiro-workflow-runtime');
const { KiroAcpClient } = require('./kiro-acp-client');

let pass = 0;
let fail = 0;
function check(name, got, want) {
  const g = JSON.stringify(got);
  const w = JSON.stringify(want);
  if (g === w) pass++;
  else {
    fail++;
    console.error(`FAIL ${name}\n  got:  ${g}\n  want: ${w}`);
  }
}
function ok(name, cond) {
  check(name, !!cond, true);
}

// ---- pure helpers ----
check('cleanReply strips ANSI + "> " marker', cleanReply('\x1b[38;5;141m> \x1b[0m{"ok":true}\x1b[0m'), '{"ok":true}');
check('cleanReply unwraps ```json fence', cleanReply('```json\n{"a":1}\n```'), '{"a":1}');
check('extract object from prose', extractJsonText('noise {"a":1} tail'), '{"a":1}');
check('extract array', extractJsonText('[1,2,3]'), '[1,2,3]');
check('extract respects braces inside strings', extractJsonText('{"k":"a}b{c"}'), '{"k":"a}b{c"}');
check('extract nested', extractJsonText('x {"a":{"b":[1]}} y'), '{"a":{"b":[1]}}');
check('extract none -> null', extractJsonText('no json here'), null);

// ---- tolerantParse: RC#3 parse-failure half. Strict-first, then SAFE syntax repairs. ----
{
  // valid JSON is returned UNTOUCHED and never marked repaired (the core safety guarantee)
  const v = tolerantParse('{"a":1,"b":[2,3],"s":"x, y, True // not a comment"}');
  ok('tolerant: valid JSON parses with repaired=false', v.value && v.repaired === false && v.repairs.length === 0);
  ok('tolerant: valid JSON string values are untouched', v.value.s === 'x, y, True // not a comment');
  // trailing commas
  const tc = tolerantParse('{"a":1,"b":[2,3,],}');
  ok('tolerant: removes trailing commas', tc.value && tc.value.a === 1 && tc.value.b.length === 2 && tc.repaired === true);
  // // and /* */ comments
  const cm = tolerantParse('{\n  "a":1, // one\n  /* block */ "b":2\n}');
  ok('tolerant: strips // and /* */ comments', cm.value && cm.value.a === 1 && cm.value.b === 2);
  // single-quoted strings + apostrophe preserved inside a double-quoted value
  const sq = tolerantParse("{'k': 'v', \"q\": \"can't stop\"}");
  ok('tolerant: converts single-quoted strings, keeps real apostrophes', sq.value && sq.value.k === 'v' && sq.value.q === "can't stop");
  // smart / curly quotes
  const smart = tolerantParse('{“k”: “v”}');
  ok('tolerant: normalizes smart quotes', smart.value && smart.value.k === 'v');
  // Python literals as bare words
  const py = tolerantParse('{"a": True, "b": False, "c": None}');
  ok('tolerant: maps True/False/None -> true/false/null', py.value && py.value.a === true && py.value.b === false && py.value.c === null);
  ok('tolerant: does NOT touch True/None inside a string value', tolerantParse('{"s":"True None"}').value.s === 'True None');
  // unquoted keys
  const uk = tolerantParse('{name: "x", rung: "new"}');
  ok('tolerant: quotes unquoted keys', uk.value && uk.value.name === 'x' && uk.value.rung === 'new');
  // combined defects (the realistic kiro-cli-2.5.0 case)
  const combo = tolerantParse("{name: 'BetaCalibration', is_new: True, axes: ['a','b',], /* c */}");
  ok('tolerant: repairs a combined-defect blob', combo.value && combo.value.name === 'BetaCalibration' && combo.value.is_new === true && combo.value.axes.length === 2);
  ok('tolerant: reports the repairs it made', combo.repaired === true && combo.repairs.length >= 3);
  // unrepairable -> value undefined, error surfaced
  const bad = tolerantParse('{"a": @@@ }');
  ok('tolerant: truly-broken JSON returns undefined value + error', bad.value === undefined && typeof bad.error === 'string');
}

const schema = {
  type: 'object',
  required: ['name', 'rung'],
  properties: { name: { type: 'string' }, rung: { enum: ['reuse', 'new'] }, n: { type: 'integer' } },
};
check('schema valid -> []', validateAgainstSchema({ name: 'x', rung: 'new', n: 3 }, schema), []);
ok('schema missing required -> error', validateAgainstSchema({ rung: 'new' }, schema).length > 0);
ok('schema bad enum -> error', validateAgainstSchema({ name: 'x', rung: 'bad' }, schema).some((e) => e.includes('enum')));
ok('schema bad type -> error', validateAgainstSchema({ name: 5, rung: 'new' }, schema).some((e) => e.includes('type')));
ok('schema array valid item -> []', validateAgainstSchema([{ name: 'a', rung: 'reuse' }], { type: 'array', items: schema }).length === 0);
ok('schema array invalid item -> error', validateAgainstSchema([{ name: 'a', rung: 'bad' }], { type: 'array', items: schema }).length > 0);

// ---- coerceToSchema: the Opus-4.8 [{...}]-instead-of-{...} fix ----
{
  const c1 = coerceToSchema([{ name: 'x', rung: 'new' }], schema); // object wanted, single-elem array given
  ok('coerce: unwraps single-element array -> object', c1.coerced === true && !Array.isArray(c1.value) && c1.value.name === 'x');
  check('coerce: unwrapped value validates clean', validateAgainstSchema(c1.value, schema), []);
  const c2 = coerceToSchema({ name: 'x', rung: 'new' }, schema); // already correct
  ok('coerce: leaves a correct object untouched', c2.coerced === false && c2.value.name === 'x');
  const c3 = coerceToSchema([{ a: 1 }, { a: 2 }], schema); // 2-elem array: NOT unwrapped (ambiguous)
  ok('coerce: does NOT unwrap a multi-element array', c3.coerced === false && Array.isArray(c3.value));
  const arrSchema = { type: 'array', items: schema };
  const c4 = coerceToSchema({ name: 'x', rung: 'new' }, arrSchema); // array wanted, bare object given
  ok('coerce: wraps bare object -> single-element array', c4.coerced === true && Array.isArray(c4.value) && c4.value.length === 1);
  const c5 = coerceToSchema('plain', { type: 'string' }); // no container mismatch
  ok('coerce: leaves a scalar untouched', c5.coerced === false && c5.value === 'plain');
  // multi-element array of IDENTICAL objects -> unwrap to the first (no data lost); this is the
  // Opus-4.8-on-2.5.0 "same answer N times in an array" case reported on the small `locate` schema.
  const dup = { name: 'x', rung: 'new' };
  const c6 = coerceToSchema([dup, { ...dup }, { ...dup }], schema);
  ok('coerce: unwraps a multi-element array of IDENTICAL objects', c6.coerced === true && !Array.isArray(c6.value) && c6.value.name === 'x');
  check('coerce: the unwrapped identical-array value validates clean', validateAgainstSchema(c6.value, schema), []);
  // multi-element array of DIFFERENT objects -> ambiguous, NOT unwrapped (left for the re-prompt).
  const c7 = coerceToSchema([{ name: 'a', rung: 'new' }, { name: 'b', rung: 'reuse' }], schema);
  ok('coerce: does NOT unwrap a multi-element array of DIFFERENT objects', c7.coerced === false && Array.isArray(c7.value));
}

// ---- shapeDirective: explicit "one object, not an array" instruction (RC#1a) ----
{
  const objDir = shapeDirective(schema);
  ok('shapeDirective(object) says EXACTLY ONE object', /EXACTLY ONE JSON object/.test(objDir));
  ok('shapeDirective(object) forbids arrays/alternatives', /do NOT return a JSON array/i.test(objDir) && /alternatives/.test(objDir));
  ok('shapeDirective(object) shows a { } skeleton with the keys', objDir.includes('"name"') && objDir.includes('"rung"') && /\{[\s\S]*\}/.test(objDir));
  const arrDir = shapeDirective({ type: 'array', items: schema });
  ok('shapeDirective(array) asks for a JSON array', /JSON array/.test(arrDir) && !/EXACTLY ONE JSON object/.test(arrDir));
}

// ---- runPool concurrency cap ----
(async () => {
  let inFlight = 0;
  let peak = 0;
  const thunks = Array.from({ length: 10 }, () => async () => {
    inFlight++;
    peak = Math.max(peak, inFlight);
    await new Promise((r) => setTimeout(r, 10));
    inFlight--;
    return 1;
  });
  const res = await runPool(thunks, 3);
  ok('runPool ran all tasks', res.length === 10 && res.every((r) => r === 1));
  ok('runPool respected concurrency cap', peak <= 3);

  // ---- end-to-end via run-workflow.js + a mock kiro-cli ----
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiro-wf-test-'));
  const mock = path.join(dir, 'mock-kiro.js');
  fs.writeFileSync(
    mock,
    `#!/usr/bin/env node
const argv = process.argv.slice(2);
const prompt = argv[argv.length-1] || "";
process.stderr.write("WARNING mock\\n");
let reply;
const lab = (prompt.match(/LABEL=(\\w+)/)||[])[1] || "none";
if (/Respond AGAIN|was rejected/i.test(prompt)) reply = '{"label":"recovered","ok":true}';
else if (/FORCE_REPROMPT/.test(prompt)) reply = "prose only, no json";
else if (/ARRAYWRAP/.test(prompt)) reply = '[{"label":"'+lab+'","ok":true},{"label":"'+lab+'","ok":true}]';   // Opus-4.8-on-2.5.0: same object repeated in a MULTI-element array -> coercion unwraps
else if (/MULTIALT/.test(prompt)) reply = '[{"label":"aa","ok":true},{"label":"bb","ok":false}]';   // multi-element DIFFERENT objects -> coercion can't unwrap -> sharper re-prompt recovers
else if (/SCHEMA/i.test(prompt)) reply = '{"label":"'+lab+'","ok":true}';
else reply = "plain reply " + prompt.slice(0,10);
process.stdout.write("\\x1b[38;5;141m> \\x1b[0m"+reply+"\\x1b[0m\\n");
process.exit(0);
`
  );
  fs.chmodSync(mock, 0o755);

  const wf = path.join(dir, 'wf.js');
  fs.writeFileSync(
    wf,
    `export const meta = { name: 'e2e', description: 'x', phases: [{title:'Fan'},{title:'Pipe'},{title:'Reprompt'}] };
const S = { type:'object', required:['label','ok'], properties:{ label:{type:'string'}, ok:{type:'boolean'} } };
const names = (args && args.names) || ['a','b'];
phase('Fan');
const fan = await parallel(names.map(n => () => agent('SCHEMA LABEL='+n, { label:'fan:'+n, schema:S })));
phase('Pipe');
const piped = await pipeline(names,
  (n) => agent('plain '+n, { label:'p1:'+n }),
  (prev, orig) => agent('SCHEMA LABEL='+orig, { label:'p2:'+orig, schema:S }));
phase('Reprompt');
const rec = await agent('SCHEMA FORCE_REPROMPT LABEL=z', { label:'rp', schema:S });
const wrapped = await agent('SCHEMA ARRAYWRAP LABEL=w', { label:'wrap', schema:S });   // multi-element identical array coerced -> {...}
const multialt = await agent('SCHEMA MULTIALT LABEL=m', { label:'malt', schema:S });   // multi-element DIFFERENT array -> re-prompt recovers
return { fan, piped, rec, wrapped, multialt, argsSeen: names, fanOk: fan.filter(Boolean).length };
`
  );

  const runner = path.join(__dirname, 'run-workflow.js');
  let out;
  try {
    out = execFileSync(
      'node',
      [runner, wf, '--kiro-bin', mock, '--args', '{"names":["x","y"]}', '--concurrency', '4'],
      { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }
    );
  } catch (e) {
    out = e.stdout || '';
    fail++;
    console.error('FAIL e2e runner threw: ' + e.message);
  }
  let result = null;
  try {
    result = JSON.parse(out);
  } catch (e) {
    fail++;
    console.error('FAIL e2e result not JSON: ' + out.slice(0, 200));
  }
  if (result) {
    ok('e2e args threaded', JSON.stringify(result.argsSeen) === JSON.stringify(['x', 'y']));
    ok('e2e parallel schema agents ok', result.fanOk === 2 && result.fan.every((f) => f && f.ok === true));
    ok('e2e pipeline produced 2', result.piped.filter(Boolean).length === 2);
    ok('e2e last pipeline stage validated', result.piped.every((p) => p && p.ok === true));
    ok('e2e re-prompt recovered from prose', result.rec && result.rec.label === 'recovered');
    ok('e2e multi-element identical array coerced to object (Opus-4.8-on-2.5.0 fix)', result.wrapped && !Array.isArray(result.wrapped) && result.wrapped.label === 'w' && result.wrapped.ok === true);
    ok('e2e multi-element DIFFERENT array recovers via sharper re-prompt', result.multialt && !Array.isArray(result.multialt) && result.multialt.label === 'recovered');
  }

  // ---- ACP transport via run-workflow.js + a mock `kiro-cli acp` server ----
  // The mock speaks the real wire contract: initialize -> session/new -> session/prompt, streaming a
  // session/update agent_message_chunk then returning {stopReason:'end_turn'}. It echoes a JSON reply
  // keyed off LABEL= in the prompt so the schema gate passes, and emits a _kiro.dev/* notification the
  // client must ignore.
  const acpMock = path.join(dir, 'mock-acp.js');
  const mcpSeenFile = path.join(dir, 'mcp-seen.json'); // the mock records session/new mcpServers here
  fs.writeFileSync(
    acpMock,
    `#!/usr/bin/env node
const fs=require('node:fs');
let buf='';
const send=o=>process.stdout.write(JSON.stringify(o)+'\\n');
process.stdin.on('data',d=>{ buf+=d.toString(); let i;
  while((i=buf.indexOf('\\n'))>=0){ const line=buf.slice(0,i); buf=buf.slice(i+1); if(!line.trim())continue;
    let m; try{m=JSON.parse(line)}catch{continue}
    if(m.method==='initialize'){ send({jsonrpc:'2.0',id:m.id,result:{protocolVersion:1,agentInfo:{name:'MockAgent',version:'0.0.0'},agentCapabilities:{}}}); }
    else if(m.method==='session/new'){ try{ fs.writeFileSync(${JSON.stringify(mcpSeenFile)}, JSON.stringify(m.params.mcpServers||null)); }catch(e){}
      send({jsonrpc:'2.0',method:'_kiro.dev/noise',params:{sessionId:'s1'}}); send({jsonrpc:'2.0',id:m.id,result:{sessionId:'s1',modes:{}}}); }
    else if(m.method==='session/prompt'){ const text=(m.params.prompt&&m.params.prompt[0]&&m.params.prompt[0].text)||'';
      const lab=(text.match(/LABEL=(\\w+)/)||[])[1]; const reply=/SCHEMA/i.test(text)?JSON.stringify({label:lab||'none',ok:true}):('MOCK '+lab);
      send({jsonrpc:'2.0',method:'session/update',params:{sessionId:'s1',update:{sessionUpdate:'agent_message_chunk',content:{type:'text',text:reply}}}});
      send({jsonrpc:'2.0',id:m.id,result:{stopReason:'end_turn'}}); }
  }
});
`
  );
  fs.chmodSync(acpMock, 0o755);

  const acpWf = path.join(dir, 'acp-wf.js');
  fs.writeFileSync(
    acpWf,
    `export const meta = { name:'acp-e2e', description:'x', phases:[{title:'T'}] };
const S = { type:'object', required:['label','ok'], properties:{ label:{type:'string'}, ok:{type:'boolean'} } };
phase('T');
const one = await agent('SCHEMA LABEL=uno', { label:'one', schema:S });
const fan = await parallel([ () => agent('plain LABEL=aa', {label:'aa'}), () => agent('plain LABEL=bb', {label:'bb'}) ]);
return { one, fan };
`
  );

  let acpOut = '';
  try {
    acpOut = execFileSync(
      'node',
      // --mcp-cursus proves the RC#3/RC#5 wiring reaches session/new (mock records it to mcpSeenFile)
      [runner, acpWf, '--kiro-bin', acpMock, '--transport', 'acp', '--mcp-cursus'],
      { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }
    );
  } catch (e) {
    acpOut = e.stdout || '';
    fail++;
    console.error('FAIL acp runner threw: ' + e.message);
  }
  // The cursus MCP server must have reached the mock's session/new call (RC#3/RC#5 end-to-end).
  try {
    const seen = JSON.parse(fs.readFileSync(mcpSeenFile, 'utf8'));
    ok('acp e2e: --mcp-cursus registered the cursus server in session/new',
      Array.isArray(seen) && seen.length === 1 && seen[0].name === 'cursus' && seen[0].command === 'cursus' && JSON.stringify(seen[0].args) === JSON.stringify(['mcp', 'serve']));
  } catch (e) {
    fail++;
    console.error('FAIL acp e2e: mcpServers not recorded by mock: ' + e.message);
  }
  let acpResult = null;
  try {
    acpResult = JSON.parse(acpOut);
  } catch (e) {
    fail++;
    console.error('FAIL acp result not JSON: ' + acpOut.slice(0, 200));
  }
  if (acpResult) {
    ok('acp schema turn ok', acpResult.one && acpResult.one.label === 'uno' && acpResult.one.ok === true);
    ok('acp parallel reused session', Array.isArray(acpResult.fan) && acpResult.fan.length === 2 && acpResult.fan.every((f) => typeof f === 'string' && f.startsWith('MOCK')));
  }

  fs.rmSync(dir, { recursive: true, force: true });

  // ---- version-skew arg-shape (kiro-cli 2.5.0 vs 2.10.0) — no spawn, just inspect the arg builder ----
  // Default (2.10.0-style): subcommand entry, new flags allowed.
  const acpDefault = new KiroAcpClient({ kiroBin: 'kiro-cli', trustTools: 'fs_read', effort: 'low' });
  const tgtDefault = acpDefault._spawnTarget();
  ok('acp default entry = kiro-cli', tgtDefault.bin === 'kiro-cli');
  ok('acp default leads with the acp subcommand', tgtDefault.args[0] === 'acp');
  ok('acp default emits granular --trust-tools when allowed', tgtDefault.args.some((a) => a.startsWith('--trust-tools=')));

  // 2.5.0-safe: chat-binary entry + legacy flags — no `acp` arg, no --effort, no granular --trust-tools.
  const acp25 = new KiroAcpClient({
    acpEntry: 'chat-binary',
    kiroChatBin: 'kiro-cli-chat',
    allowNewFlags: false,
    trustTools: 'fs_read',
    effort: 'low',
    agent: 'my-agent',
  });
  const tgt25 = acp25._spawnTarget();
  ok('acp 2.5.0 entry = kiro-cli-chat binary', tgt25.bin === 'kiro-cli-chat');
  ok('acp 2.5.0 has NO acp subcommand arg', !tgt25.args.includes('acp'));
  ok('acp 2.5.0 drops granular --trust-tools', !tgt25.args.some((a) => a.startsWith('--trust-tools=')));
  ok('acp 2.5.0 drops --effort', !tgt25.args.includes('--effort'));
  ok('acp 2.5.0 keeps --trust-all-tools (safe)', tgt25.args.includes('--trust-all-tools'));
  ok('acp 2.5.0 keeps --agent (safe)', tgt25.args.includes('--agent') && tgt25.args.includes('my-agent'));

  // ---- ACP mcpServers wiring (RC#3/RC#5): session/new params shape ----
  const acpMcp = new KiroAcpClient({
    mcpServers: [
      { name: 'cursus', command: 'cursus', args: ['mcp', 'serve'] },
      { name: 'py', command: 'python3', args: ['-m', 'cursus.mcp.server'], env: { PYTHONNOUSERSITE: '1' }, cwd: '/w' },
    ],
  });
  const mp = acpMcp._mcpServerParams();
  ok('acp mcp: two servers mapped', mp.length === 2 && mp[0].name === 'cursus' && mp[0].command === 'cursus');
  ok('acp mcp: cursus args preserved', JSON.stringify(mp[0].args) === JSON.stringify(['mcp', 'serve']));
  ok('acp mcp: env encoded as [{name,value}] array', Array.isArray(mp[1].env) && mp[1].env[0].name === 'PYTHONNOUSERSITE' && mp[1].env[0].value === '1');
  ok('acp mcp: cwd carried through', mp[1].cwd === '/w');
  ok('acp mcp: none configured -> empty array (prior default)', new KiroAcpClient({})._mcpServerParams().length === 0);

  // ---- runtime.mcpCursus auto-registers the cursus server ----
  const { KiroWorkflowRuntime } = require('./kiro-workflow-runtime');
  const rtCursus = new KiroWorkflowRuntime({ mcpCursus: true });
  ok('runtime mcpCursus:true -> cursus mcp serve entry', rtCursus.mcpServers.length === 1 && rtCursus.mcpServers[0].command === 'cursus' && JSON.stringify(rtCursus.mcpServers[0].args) === JSON.stringify(['mcp', 'serve']));
  const rtPy = new KiroWorkflowRuntime({ mcpCursus: 'python3' });
  ok('runtime mcpCursus:"python3" -> python -m cursus.mcp.server', rtPy.mcpServers[0].command === 'python3' && JSON.stringify(rtPy.mcpServers[0].args) === JSON.stringify(['-m', 'cursus.mcp.server']));
  ok('runtime no mcp config -> empty', new KiroWorkflowRuntime({}).mcpServers.length === 0);

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
})();
