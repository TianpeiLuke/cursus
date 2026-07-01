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
  validateAgainstSchema,
  runPool,
} = require('./kiro-workflow-runtime');

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
if (/Respond AGAIN|was rejected/i.test(prompt)) reply = '{"label":"recovered","ok":true}';
else if (/FORCE_REPROMPT/.test(prompt)) reply = "prose only, no json";
else if (/SCHEMA/i.test(prompt)) reply = '{"label":"'+((prompt.match(/LABEL=(\\w+)/)||[])[1]||"none")+'","ok":true}';
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
return { fan, piped, rec, argsSeen: names, fanOk: fan.filter(Boolean).length };
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
  }

  fs.rmSync(dir, { recursive: true, force: true });

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
})();
