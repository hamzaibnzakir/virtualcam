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

async function run() {
  Object.assign(liteTuning, config.params || {});
  const lite = new LiteFaceAI({ api, onState: () => {}, onError: (m) => console.warn(`lite: ${m}`) });
  const results = [];
  for (const pair of config.pairs) {
    const started = performance.now();
    const refBytes = fs.readFileSync(path.join(config.facesDir, pair.ref));
    const refBlob = new Blob([refBytes], { type: 'image/jpeg' });
    const frame = makeFrame(await loadImage(path.join(config.facesDir, pair.target)), pair.frame);
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
