// Runs inside the hidden benchmark window: renders each pair with the real Lite engine.
// Node integration is on here (test tool only), so results are written straight to disk.
import { LiteFaceAI, liteTuning } from '../../src/renderer/engine/ai-lite.js';
import { output } from '../../src/renderer/engine/state.js';

const fs = require('node:fs');
const path = require('node:path');

const config = JSON.parse(fs.readFileSync(process.env.BENCH_CONFIG, 'utf8'));
const DIST = config.assetsDir;
const W = output.width;
const H = output.height;

const api = {
  readAsset: async (name) => new Uint8Array(fs.readFileSync(path.join(DIST, name))),
};

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`cannot load ${file}`));
    img.src = `file://${file}`;
  });
}

// The "camera" picture: the target face placed like a webcam frame, slightly tilted.
function makeFrame(img, frame) {
  const c = document.createElement('canvas');
  c.width = W;
  c.height = H;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = 'rgb(40,44,52)';
  ctx.fillRect(0, 0, W, H);
  ctx.save();
  ctx.translate(frame.x + frame.size / 2, frame.y + frame.size / 2);
  ctx.rotate((frame.tilt * Math.PI) / 180);
  ctx.drawImage(img, -frame.size / 2, -frame.size / 2, frame.size, frame.size);
  ctx.restore();
  return c;
}

function rgbaOf(canvas) {
  const c = document.createElement('canvas');
  c.width = W;
  c.height = H;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(canvas, 0, 0);
  return ctx.getImageData(0, 0, W, H).data;
}

// The photo as the scorer sees it: 360 px face centred in 512 px (close-ups need a border to be detected).
async function paddedRef(file) {
  const img = await loadImage(file);
  const c = document.createElement('canvas');
  c.width = 512;
  c.height = 512;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, 512, 512);
  ctx.drawImage(img, 76, 76, 360, 360);
  return ctx.getImageData(0, 0, 512, 512).data.buffer;
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------- motion mode: a moving "camera" with a known head transform ----------
const FPS = 30;
const ease = (u) => u * u * (3 - 2 * u);

// Head transform at motion frame k (k = 0 is the still start). The head moves for the first 70% of
// the frames (sideways and back, rolling +8° to -8°, scale 0.9 to 1.1, auto-exposure dimming to 75%),
// easing in and out like a real head, then holds still. It also narrows to 85% width and back, like
// the foreshortening of a ~30° head turn: that changes the face's shape, not just its position, so a
// smoother can't pass the test by only following a rigid move. With a dropout, a few black frames
// follow (a camera hiccup), and the face comes back slightly to the side, as if the head moved meanwhile.
function motionAt(k, m, size, sign) {
  const n = m.frames;
  const moveEnd = Math.round(n * 0.7);
  const v = ease(Math.min(1, k / moveEnd));
  const tr = {
    dx: sign * 0.05 * size * Math.sin(2 * Math.PI * v),
    dy: 0.04 * size * v,
    rot: sign * 8 * Math.sin(1.5 * Math.PI * v),
    scale: 1 - 0.1 * Math.sin(1.5 * Math.PI * v),
    squash: 1 - 0.15 * Math.sin(Math.PI * v),
    bright: 1 - 0.25 * v,
    phase: k === 0 ? 'still' : k < moveEnd ? 'move' : 'hold',
  };
  if (k >= n) {
    tr.phase = k < n + m.dropout ? 'dropout' : 'back';
    tr.dx += sign * 0.04 * size;
  }
  return tr;
}

// Seeded noise so runs repeat exactly: sensor noise of about `sigma` levels on every frame.
function makeNoise(sigma, length) {
  let s = 0x9e3779b9;
  const rand = () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const noise = new Int8Array(length);
  for (let i = 0; i < length; i += 1) {
    // Sum of 4 uniforms: close to Gaussian, variance 4/12.
    const g = (rand() + rand() + rand() + rand() - 2) * Math.sqrt(3);
    noise[i] = Math.max(-127, Math.min(127, Math.round(g * sigma)));
  }
  return noise;
}

function makeMotionFrame(img, frame, tr, noise, k) {
  const c = document.createElement('canvas');
  c.width = W;
  c.height = H;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  if (tr.phase === 'dropout') {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, W, H);
    return c;
  }
  ctx.fillStyle = 'rgb(40,44,52)';
  ctx.fillRect(0, 0, W, H);
  ctx.save();
  ctx.translate(frame.x + frame.size / 2 + tr.dx, frame.y + frame.size / 2 + tr.dy);
  ctx.rotate(((frame.tilt + tr.rot) * Math.PI) / 180);
  ctx.scale(tr.scale * tr.squash * config.motion.zoom, tr.scale * config.motion.zoom);
  ctx.drawImage(img, -frame.size / 2, -frame.size / 2, frame.size, frame.size);
  ctx.restore();
  if (tr.bright !== 1) {
    const v = Math.round(255 * tr.bright);
    ctx.globalCompositeOperation = 'multiply';
    ctx.fillStyle = `rgb(${v},${v},${v})`;
    ctx.fillRect(0, 0, W, H);
    ctx.globalCompositeOperation = 'source-over';
  }
  if (noise) {
    const im = ctx.getImageData(0, 0, W, H);
    const d = im.data;
    const off = ((k * 7919) % 65536) * 4;
    for (let i = 0; i < d.length; i += 4) {
      d[i] += noise[off + i];
      d[i + 1] += noise[off + i + 1];
      d[i + 2] += noise[off + i + 2];
    }
    ctx.putImageData(im, 0, 0);
  }
  return c;
}

// Ground truth: a point of the still start frame moved by the head transform.
function transformPoints(p0, frame, tr) {
  const cx = frame.x + frame.size / 2;
  const cy = frame.y + frame.size / 2;
  const a0 = (frame.tilt * Math.PI) / 180;
  const a1 = ((frame.tilt + tr.rot) * Math.PI) / 180;
  const [c0, s0, c1, s1] = [Math.cos(a0), Math.sin(a0), Math.cos(a1), Math.sin(a1)];
  const out = new Float32Array(p0.length);
  for (let i = 0; i < p0.length; i += 2) {
    const x = p0[i] - cx;
    const y = p0[i + 1] - cy;
    // Into the face's own axes (undo its tilt), narrow and scale, then turn to the new angle.
    const u = (c0 * x + s0 * y) * tr.scale * tr.squash;
    const v = (-s0 * x + c0 * y) * tr.scale;
    out[i] = cx + tr.dx + c1 * u - s1 * v;
    out[i + 1] = cy + tr.dy + s1 * u + c1 * v;
  }
  return out;
}

const rounded = (a) => (a ? Array.from(a, (v) => Math.round(v * 100) / 100) : null);

// Feeds one pair through lite.tick like a moving camera at 30 fps (simulated timestamps, so the
// smoothing sees real camera timing even though software WebGL renders far slower than that).
async function runMotion(lite, pair, img, index, noise, r) {
  const m = config.motion;
  const size = pair.frame.size * m.zoom;
  const sign = index % 2 ? -1 : 1;
  const total = m.frames + (m.dropout > 0 ? m.dropout + 8 : 0);
  const scoreAt = new Set([0]);
  for (let j = 1; j <= m.score; j += 1) scoreAt.add(Math.round(((m.frames - 1) * j) / m.score));
  const step = 1000 / FPS;
  let t = Math.max(lite.lastTs, performance.now()) + 1000;
  const ms = [];
  // One camera frame at the next 30 fps timestamp; returns what the engine tracked and drew.
  const tick = (canvas) => {
    const t0 = performance.now();
    lite.tick((ctx, w, h) => ctx.drawImage(canvas, 0, 0, w, h), t);
    ms.push(performance.now() - t0);
    const snap = {
      t,
      found: lite.faceFound,
      raw: lite.faceFound ? Float32Array.from(lite.raw) : null,
      pts: lite.points ? Float32Array.from(lite.points) : null,
      strength: lite.points && lite.dstStats ? (lite.smoother.strength ?? 1) : 0,
    };
    t += step;
    return snap;
  };
  const save = (canvas, k, kind) =>
    fs.writeFileSync(path.join(config.outDir, `${pair.target}.m${k}.${kind}.rgba`), Buffer.from(rgbaOf(canvas).buffer));

  // Still start: let tracking settle; the ground-truth points are the mean raw points of the last few.
  const still = motionAt(0, m, size, sign);
  const warm = [];
  const frames = [];
  let f0 = null;
  for (let i = 0; i < m.warmup; i += 1) {
    f0 = makeMotionFrame(img, pair.frame, still, noise, 1000 + i);
    const snap = tick(f0);
    if (snap.found && i >= m.warmup - 4) warm.push(snap.raw);
    frames.push({ k: i - m.warmup + 1, phase: i < m.warmup - 1 ? 'warmup' : 'still', gt: null, ...snap });
  }
  if (!warm.length) {
    r.motion = { error: 'no face at the start' };
    return;
  }
  const p0 = new Float32Array(warm[0].length);
  for (const w of warm) for (let i = 0; i < p0.length; i += 1) p0[i] += w[i] / warm.length;
  frames[frames.length - 1].gt = p0;
  save(f0, 0, 'orig');
  save(lite.canvas, 0, 'out');

  for (let k = 1; k < total; k += 1) {
    const tr = motionAt(k, m, size, sign);
    const f = makeMotionFrame(img, pair.frame, tr, noise, k);
    frames.push({ k, phase: tr.phase, gt: tr.phase === 'dropout' ? null : transformPoints(p0, pair.frame, tr), ...tick(f) });
    if (scoreAt.has(k)) {
      save(f, k, 'orig');
      save(lite.canvas, k, 'out');
    }
  }
  fs.writeFileSync(
    path.join(config.outDir, `${pair.target}.motion.json`),
    JSON.stringify(frames.map((f) => ({ ...f, gt: rounded(f.gt), raw: rounded(f.raw), pts: rounded(f.pts) }))),
  );
  ms.sort((a, b) => a - b);
  r.faceFound = lite.faceFound;
  r.renderMs = Math.round(ms[Math.floor(ms.length / 2)]);
  r.motion = { scored: [...scoreAt].sort((a, b) => a - b), frames: total };
}

async function run() {
  Object.assign(liteTuning, config.params || {});
  const lite = new LiteFaceAI({ api, onState: () => {}, onError: (m) => console.warn(`lite: ${m}`) });
  const results = [];
  const noise = config.motion && config.motion.noise > 0 ? makeNoise(config.motion.noise, W * H * 4 + 65536 * 4) : null;
  for (const [index, pair] of config.pairs.entries()) {
    const started = performance.now();
    const refBytes = fs.readFileSync(path.join(config.facesDir, pair.ref));
    const refBlob = new Blob([refBytes], { type: 'image/jpeg' });
    const targetImg = await loadImage(path.join(config.facesDir, pair.target));
    const frame = makeFrame(targetImg, pair.frame);
    // Inputs for scoring (only when the scorer has not cached them yet).
    if (pair.writeOrig) fs.writeFileSync(path.join(config.outDir, `${pair.target}.orig.rgba`), Buffer.from(rgbaOf(frame).buffer));
    if (pair.writeRef) fs.writeFileSync(path.join(config.outDir, `${pair.ref}.ref.rgba`), Buffer.from(await paddedRef(path.join(config.facesDir, pair.ref))));
    if (config.mode === 'inputs') {
      results.push({ id: `${pair.split}:${pair.target}`, state: 'inputs' });
      continue;
    }
    lite.setTarget({ refKey: pair.ref, refBlob });
    const until = performance.now() + 60000;
    while (performance.now() < until && !['generating', 'no-face', 'error'].includes(lite.state)) await wait(20);
    while (lite.running) await wait(20);
    const r = { id: `${pair.split}:${pair.target}`, state: lite.state };
    if (lite.state === 'generating') {
      // Each pair is a new camera: no smoothing carried over from the previous one.
      lite.smoother.reset();
      lite.dstStats = null;
      if (config.motion) {
        await runMotion(lite, pair, targetImg, index, noise, r);
        r.totalMs = Math.round(performance.now() - started);
        results.push(r);
        lite.setTarget(null);
        while (lite.running) await wait(5);
        continue;
      }
      // A few frames so tracking and smoothing settle, as they would on a live camera.
      const times = [];
      for (let i = 0; i < (config.frames || 4); i += 1) {
        const t0 = performance.now();
        lite.tick((ctx, w, h) => ctx.drawImage(frame, 0, 0, w, h));
        times.push(performance.now() - t0);
        await wait(5);
      }
      r.faceFound = lite.faceFound;
      r.renderMs = Math.round(times[times.length - 1]);
      fs.writeFileSync(path.join(config.outDir, `${pair.target}.out.rgba`), Buffer.from(rgbaOf(lite.canvas).buffer));
    }
    r.totalMs = Math.round(performance.now() - started);
    results.push(r);
    // Next pair: forget this reference.
    lite.setTarget(null);
    while (lite.running) await wait(5);
  }
  fs.writeFileSync(path.join(config.outDir, 'render.json'), JSON.stringify({ delegate: lite.delegate, results }, null, 2));
}

run()
  .catch((err) => {
    fs.writeFileSync(path.join(config.outDir, 'render.json'), JSON.stringify({ error: String(err && err.stack) }));
  })
  .finally(() => window.close());
