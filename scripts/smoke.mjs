// Launches the app in self-test mode (fake camera, hidden windows), then checks the report.
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const electron = require('electron');
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const out = join(tmpdir(), `vcam-smoke-${process.pid}.json`);
rmSync(out, { force: true });

const args = [root];
if (process.platform === 'linux' && process.getuid && process.getuid() === 0) args.push('--no-sandbox');

let cmd = electron;
let cmdArgs = args;
if (process.platform === 'linux' && !process.env.DISPLAY && existsSync('/usr/bin/xvfb-run')) {
  cmd = 'xvfb-run';
  cmdArgs = ['-a', '-s', '-screen 0 1600x1000x24', electron, ...args];
}

const run = spawnSync(cmd, cmdArgs, {
  env: { ...process.env, VCAM_SMOKE: '1', VCAM_SMOKE_OUT: out },
  encoding: 'utf8',
  timeout: 90000,
});

if (!existsSync(out)) {
  console.error('No smoke report was written.');
  console.error(run.stdout, run.stderr);
  process.exit(1);
}

const result = JSON.parse(readFileSync(out, 'utf8'));
const errors = result.console.filter((m) => m.level === 'error');
const warnings = result.console.filter((m) => m.level === 'warning');

console.log(JSON.stringify(result.report, null, 2));
const switcherRows = result.console.find((m) => m.message.startsWith('[switcher] rows='));
console.log('Quick switcher:', switcherRows ? switcherRows.message : 'did not report');
if (!switcherRows || switcherRows.message.endsWith('=0')) result.report.ok = false;
if (warnings.length) console.log('Console warnings:', warnings.map((w) => w.message));
if (result.hotkeyFailures.length) console.log('Hotkeys not registered here:', result.hotkeyFailures);

if (!result.report.ok || errors.length) {
  console.error('SMOKE FAILED');
  if (errors.length) console.error('Console errors:', errors.map((e) => e.message));
  process.exit(1);
}
console.log('SMOKE PASSED');
