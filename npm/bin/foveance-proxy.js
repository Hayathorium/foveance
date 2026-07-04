#!/usr/bin/env node
'use strict';

// Thin cross-platform launcher: forwards all CLI args to `python -m foveance.cli proxy`.
// This lets users of Node-based agents (Claude Code, Codex, opencode, ...) start the Foveance
// proxy with `npx foveance-proxy ...` without a manual Python invocation.
//
// Requirements: Python 3.10+ on PATH and `pip install foveance` (one time).

const { spawnSync, spawn } = require('child_process');

function findPython() {
  const candidates = process.platform === 'win32'
    ? ['python', 'py', 'python3']
    : ['python3', 'python'];
  for (const cmd of candidates) {
    const r = spawnSync(cmd, ['-c', 'import sys; print(sys.version_info[0])'], { encoding: 'utf8' });
    if (r.status === 0 && (r.stdout || '').trim() === '3') return cmd;
  }
  return null;
}

const py = findPython();
if (!py) {
  console.error(
    'foveance-proxy: Python 3 not found on PATH.\n' +
    'Install Python 3.10+ and then:  pip install foveance'
  );
  process.exit(127);
}

// Confirm the Python package is importable, with a friendly hint if not.
const check = spawnSync(py, ['-c', 'import foveance.proxy'], { encoding: 'utf8' });
if (check.status !== 0) {
  console.error(
    'foveance-proxy: the Foveance Python package is not installed.\n' +
    'Run:  pip install foveance'
  );
  process.exit(127);
}

const child = spawn(py, ['-m', 'foveance.cli', 'proxy', ...process.argv.slice(2)], { stdio: 'inherit' });
child.on('exit', (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  else process.exit(code === null ? 1 : code);
});
child.on('error', (err) => {
  console.error('foveance-proxy: failed to launch the proxy:', err.message);
  process.exit(1);
});
