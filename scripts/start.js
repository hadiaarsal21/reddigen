#!/usr/bin/env node
/**
 * ReddiGen single-command launcher.
 *
 *   npm start          production build, both services
 *   npm run dev        dev server with hot reload, both services
 *
 * Starts the FastAPI ML server and the Next.js app together, waits until each
 * is actually answering before declaring it ready, and shuts both down on
 * Ctrl+C. Preflight checks run first so a missing dependency produces one
 * clear message instead of a half-started stack.
 *
 * Deliberately dependency-free: it runs before `npm install` has necessarily
 * been verified, so it cannot rely on anything from node_modules.
 */

'use strict';

const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const http = require('http');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const DEV = process.argv.includes('--dev');
const ML_PORT = Number(process.env.ML_PORT || 8000);
const WEB_PORT = Number(process.env.PORT || 3000);
const IS_WIN = process.platform === 'win32';

// ── Output ──────────────────────────────────────────────────────────────

const useColour = process.stdout.isTTY && !process.env.NO_COLOR;
const c = (code, s) => (useColour ? `\x1b[${code}m${s}\x1b[0m` : s);
const dim = (s) => c('2', s);
const bold = (s) => c('1', s);
const red = (s) => c('31', s);
const green = (s) => c('32', s);
const yellow = (s) => c('33', s);
const cyan = (s) => c('36', s);

const TAG = { ml: c('35', '[ml] '), web: c('36', '[web]') };

// Lines that say nothing useful to someone watching the app start.
const NOISE = [
  /Loading weights:/,           // tqdm progress frames
  /^\s*\d+%\|/,                 // bare progress bars
  /it\/s\]|s\/it\]/,            // tqdm rate suffixes
  /^INFO: {5}127\.0\.0\.1/,     // uvicorn per-request access logs
  /^\[?A+\]?$/,                 // stray ANSI cursor-up artefacts
];

function log(tag, line) {
  process.stdout.write(`${TAG[tag]} ${line}\n`);
}

function fail(message, hint) {
  process.stderr.write(`\n${red('x')} ${bold(message)}\n`);
  if (hint) process.stderr.write(`  ${dim(hint)}\n`);
  process.stderr.write('\n');
  process.exit(1);
}

// ── Preflight ───────────────────────────────────────────────────────────

/** Path to the virtualenv interpreter, or null when the venv is missing. */
function venvPython() {
  const p = IS_WIN
    ? path.join(ROOT, '.venv', 'Scripts', 'python.exe')
    : path.join(ROOT, '.venv', 'bin', 'python');
  return fs.existsSync(p) ? p : null;
}

/** Resolve the Next.js CLI without relying on shell PATH or npx. */
function nextBin() {
  const p = path.join(ROOT, 'node_modules', 'next', 'dist', 'bin', 'next');
  return fs.existsSync(p) ? p : null;
}

function portInUse(port) {
  return new Promise((resolve) => {
    const server = require('net').createServer();
    server.once('error', (err) => resolve(err.code === 'EADDRINUSE'));
    server.once('listening', () => server.close(() => resolve(false)));
    server.listen(port, '127.0.0.1');
  });
}

async function preflight() {
  const problems = [];

  if (!fs.existsSync(path.join(ROOT, 'node_modules'))) {
    problems.push(['Node dependencies are not installed', 'Run:  npm install']);
  }

  const py = venvPython();
  if (!py) {
    problems.push([
      'Python virtualenv not found at .venv',
      IS_WIN
        ? 'Run:  python -m venv .venv  then  .venv\\Scripts\\python.exe -m pip install -r ml/requirements.txt'
        : 'Run:  python3 -m venv .venv  then  .venv/bin/python -m pip install -r ml/requirements.txt',
    ]);
  } else {
    const check = spawnSync(py, ['-c', 'import fastapi, uvicorn'], { cwd: ROOT });
    if (check.status !== 0) {
      problems.push([
        'The virtualenv is missing the ML dependencies',
        IS_WIN
          ? 'Run:  .venv\\Scripts\\python.exe -m pip install -r ml/requirements.txt'
          : 'Run:  .venv/bin/python -m pip install -r ml/requirements.txt',
      ]);
    }
  }

  if (problems.length) {
    process.stderr.write(`\n${red('Cannot start.')}\n\n`);
    for (const [what, how] of problems) {
      process.stderr.write(`  ${red('x')} ${what}\n     ${dim(how)}\n\n`);
    }
    process.exit(1);
  }

  // .env is required by Prisma. The example has working defaults, so create
  // it rather than stopping for something we can fix.
  const env = path.join(ROOT, '.env');
  if (!fs.existsSync(env) && fs.existsSync(path.join(ROOT, '.env.example'))) {
    fs.copyFileSync(path.join(ROOT, '.env.example'), env);
    log('web', yellow('created .env from .env.example'));
  }

  // A production start needs a build. Do it rather than failing, since the
  // alternative is an error the user has to decode.
  if (!DEV && !fs.existsSync(path.join(ROOT, '.next', 'BUILD_ID'))) {
    log('web', yellow('no production build found, running next build (one time)'));
    const build = spawnSync(process.execPath, [nextBin(), 'build'], {
      cwd: ROOT,
      stdio: 'inherit',
    });
    if (build.status !== 0) fail('next build failed', 'Fix the errors above and try again.');
  }

  for (const [port, name] of [[ML_PORT, 'ML server'], [WEB_PORT, 'web app']]) {
    if (await portInUse(port)) {
      fail(
        `Port ${port} is already in use (${name})`,
        IS_WIN
          ? `Find it with:  netstat -ano | findstr :${port}   then stop that process.`
          : `Find it with:  lsof -i :${port}   then stop that process.`,
      );
    }
  }
}

// ── Health polling ──────────────────────────────────────────────────────

function probe(port, timeoutMs = 2000, probePath = '/') {
  return new Promise((resolve) => {
    const req = http.get(
      { host: '127.0.0.1', port, path: probePath, timeout: timeoutMs },
      (res) => {
        res.resume();
        resolve(res.statusCode > 0);
      },
    );
    req.on('error', () => resolve(false));
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function waitFor(port, label, timeoutMs, probePath = '/') {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if (shuttingDown) return false;
    if (await probe(port, 2000, probePath)) return true;
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

// ── Process management ──────────────────────────────────────────────────

const children = [];
let shuttingDown = false;

function start(tag, command, args, extraEnv) {
  const child = spawn(command, args, {
    cwd: ROOT,
    env: { ...process.env, ...extraEnv },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  children.push({ tag, child });

  const relay = (stream, isError) => {
    let buffer = '';
    stream.on('data', (chunk) => {
      buffer += chunk.toString();
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const raw of lines) {
        // tqdm redraws with carriage returns, so one "line" can hold many
        // frames. Keep only the last, then drop it if it is a progress bar.
        const text = raw.split('\r').pop().replace(/\[[0-9;]*m/g, '');
        if (!text.trim()) continue;
        if (NOISE.some((re) => re.test(text))) continue;
        log(tag, isError ? dim(text) : text);
      }
    });
  };
  relay(child.stdout, false);
  relay(child.stderr, true);

  child.on('exit', (code, signal) => {
    if (shuttingDown) return;
    const why = signal ? `signal ${signal}` : `code ${code}`;
    log(tag, red(`exited with ${why}`));
    // One half of the app is useless alone: stop the other rather than
    // leaving a half-running stack that looks healthy.
    shutdown(code === 0 ? 0 : 1);
  });

  return child;
}

function shutdown(code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  process.stdout.write(`\n${dim('Shutting down...')}\n`);

  for (const { tag, child } of children) {
    if (child.exitCode !== null || child.signalCode) continue;
    try {
      if (IS_WIN) {
        // child.kill() does not reap grandchildren on Windows, and Next
        // spawns workers. taskkill /T ends the whole tree.
        spawnSync('taskkill', ['/pid', String(child.pid), '/T', '/F'], {
          stdio: 'ignore',
        });
      } else {
        child.kill('SIGTERM');
      }
    } catch {
      /* already gone */
    }
    log(tag, dim('stopped'));
  }

  setTimeout(() => process.exit(code), 300);
}

for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
  process.on(sig, () => shutdown(0));
}

// ── Main ────────────────────────────────────────────────────────────────

async function main() {
  process.stdout.write(
    `\n${bold('ReddiGen')} ${dim(DEV ? '(development)' : '(production)')}\n\n`,
  );

  await preflight();

  const py = venvPython();
  const next = nextBin();

  log('ml', `starting FastAPI on port ${ML_PORT}`);
  start('ml', py, [path.join('ml', 'server.py')], {
    PORT: String(ML_PORT),
    PYTHONUNBUFFERED: '1',
  });

  // /healthz reports liveness without touching the checkpoints. Polling / here
  // would make every probe attempt to load all five models.
  const mlUp = await waitFor(ML_PORT, 'ML server', 120_000, '/healthz');
  if (shuttingDown) return;
  if (!mlUp) {
    fail(
      `The ML server did not respond on port ${ML_PORT} within 120s`,
      'Check the [ml] output above. Loading five checkpoints on CPU is slow the first time.',
    );
  }
  log('ml', green(`ready on http://localhost:${ML_PORT}`));

  log('web', `starting Next.js on port ${WEB_PORT}`);
  start('web', process.execPath, [next, DEV ? 'dev' : 'start', '-p', String(WEB_PORT)], {
    ML_SERVICE_URL: `http://localhost:${ML_PORT}`,
  });

  const webUp = await waitFor(WEB_PORT, 'web app', 180_000);
  if (shuttingDown) return;
  if (!webUp) {
    fail(
      `The web app did not respond on port ${WEB_PORT} within 180s`,
      'Check the [web] output above.',
    );
  }

  process.stdout.write(
    `\n  ${green('ReddiGen is running')}\n` +
      `  ${bold(cyan(`http://localhost:${WEB_PORT}`))}\n\n` +
      `  ${dim(`ML API   http://localhost:${ML_PORT}`)}\n` +
      `  ${dim('Stop     Ctrl+C')}\n\n`,
  );
}

main().catch((err) => {
  process.stderr.write(`\n${red('Launcher error:')} ${err && err.stack ? err.stack : err}\n`);
  shutdown(1);
});
