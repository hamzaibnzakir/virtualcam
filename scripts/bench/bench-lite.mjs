// Lite engine benchmark. Renders every pair with the real Lite engine, then scores it.
//
//   node scripts/bench/bench-lite.mjs [--split train|holdout|all] [--limit N] [--engine lite|local|original]
//        [--params '{"shading":0.3}'] [--out DIR] [--images N] [--frames N] [--only i,j,...] [--quiet]
//        [--motion [--motion-frames 30] [--motion-score 4] [--motion-noise 3] [--motion-dropout 2] [--motion-zoom 1]]
//        [--sweep '[{"shading":0.1},{"shading":0.3}]' | --sweep FILE.json]
//   --sweep (tuning aid, lite only): after the normal render, re-renders each pair's last tracked frame
//   with every parameter set (on top of --params) and prints one extra summary line per set.
//
// Shared data (set these to reuse downloads between runs and worktrees):
//   BENCH_FACES   folder with the SFHQ sample jpgs   (downloaded if missing)
//   BENCH_MODELS  folder with the InsightFace models (downloaded if missing)
//   BENCH_CACHE   folder for cached scores of the unchanging inputs
// Prints one summary JSON line at the end; full per-pair results go to <out>/results.json.
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..', '..');

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : fallback;
}
const flag = (name) => process.argv.includes(`--${name}`);

const split = arg('split', 'train');
const limit = Number(arg('limit', '0'));
const engine = arg('engine', 'lite');
const params = JSON.parse(arg('params', '{}'));
const frames = Number(arg('frames', '4'));
const imageCount = Number(arg('images', '0'));
const quiet = flag('quiet');
// Motion mode (off by default; lite only): each pair moves over ~30 frames like a live camera. See motion.cjs.
const motion = flag('motion')
  ? {
      frames: Number(arg('motion-frames', '30')),
      score: Number(arg('motion-score', '4')), // frames scored for identity, besides the still start
      noise: Number(arg('motion-noise', '3')), // camera sensor noise, in 0-255 levels
      dropout: Number(arg('motion-dropout', '2')), // black frames after the move (0 = none)
      zoom: Number(arg('motion-zoom', '1')), // face size vs the still benchmark (0.5 = a face further away)
      warmup: 8,
    }
  : null;
if (motion && engine !== 'lite') throw new Error('--motion needs --engine lite');
const sweepArg = arg('sweep', '');
const sweep = sweepArg ? JSON.parse(existsSync(sweepArg) ? readFileSync(sweepArg, 'utf8') : sweepArg) : null;
const out = resolve(arg('out', join(tmpdir(), `lite-bench-${Date.now()}`)));
const cacheRoot = resolve(process.env.BENCH_CACHE || join(root, '.bench-cache'));
const facesDir = resolve(process.env.BENCH_FACES || join(cacheRoot, 'faces'));
const modelsDir = resolve(process.env.BENCH_MODELS || join(cacheRoot, 'models'));
const log = (...a) => {
  if (!quiet) console.error(...a);
};

mkdirSync(out, { recursive: true });

// ---------- data ----------
const dataset = JSON.parse(readFileSync(join(here, 'dataset.json'), 'utf8'));
let pairs = dataset.pairs.filter((p) => split === 'all' || p.split === split);
// --only 3,17,40: just these pairs (indices within the split), e.g. to look at particular cases.
const only = arg('only', '');
if (only) pairs = only.split(',').map(Number).map((i) => pairs[i]).filter(Boolean);
if (limit) pairs = pairs.slice(0, limit);

if (!existsSync(join(facesDir, pairs[0].target))) {
  log('Downloading the SFHQ face sample…');
  mkdirSync(facesDir, { recursive: true });
  const zip = join(cacheRoot, 'faces.zip');
  const res = await fetch('https://huggingface.co/datasets/bitmind/SyntheticFacesHQ/resolve/main/tiny-sample.zip');
  writeFileSync(zip, Buffer.from(await res.arrayBuffer()));
  spawnSync('python3', ['-c', 'import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])', zip, facesDir], { stdio: 'inherit' });
}
if (!existsSync(join(modelsDir, 'w600k_r50.onnx'))) {
  log('Downloading the scoring models…');
  const { Downloader } = require('../../src/main/models.js');
  await new Downloader(modelsDir).download('core', () => {});
}

const { Scorer, cos, poseError, summarise, W, H } = require('./score.cjs');
const scorer = new Scorer(modelsDir, join(cacheRoot, 'scores'));
const key = (kind, p) => `${kind}:${kind === 'ref' ? p.ref : `${p.target}:${JSON.stringify(p.frame)}`}`;
const cacheFile = (k) => join(cacheRoot, 'scores', `${createHash('sha1').update(k).digest('hex')}.json`);
for (const p of pairs) {
  p.writeOrig = !existsSync(cacheFile(key('orig', p)));
  p.writeRef = !existsSync(cacheFile(key('ref', p)));
}
// Comparison images need the original frames.
pairs.slice(0, imageCount).forEach((p) => {
  p.writeOrig = true;
});

// ---------- render ----------
async function render(mode, renderPairs) {
  const dist = join(out, 'dist');
  rmSync(dist, { recursive: true, force: true });
  mkdirSync(join(dist, 'mediapipe'), { recursive: true });
  await build({
    entryPoints: [join(here, 'bench-renderer.js')],
    bundle: true,
    format: 'iife',
    platform: 'browser',
    external: ['node:*'],
    outfile: join(dist, 'bench.js'),
    logLevel: 'error',
  });
  cpSync(join(here, 'bench.html'), join(dist, 'bench.html'));
  const wasm = join(root, 'node_modules/@mediapipe/tasks-vision/wasm');
  for (const f of ['vision_wasm_internal.js', 'vision_wasm_internal.wasm']) symlinkSync(join(wasm, f), join(dist, 'mediapipe', f));
  symlinkSync(join(root, 'assets/mediapipe/face_landmarker.task'), join(dist, 'mediapipe', 'face_landmarker.task'));

  const configFile = join(out, 'config.json');
  writeFileSync(configFile, JSON.stringify({ mode, pairs: renderPairs, params, sweep, frames, facesDir, outDir: out, assetsDir: dist, ...(motion ? { motion } : {}) }));
  const electron = require('electron');
  let cmd = electron;
  let args = [join(here, 'electron-main.cjs')];
  if (process.platform === 'linux' && process.getuid && process.getuid() === 0) args.push('--no-sandbox');
  if (process.platform === 'linux' && !process.env.DISPLAY) {
    cmd = 'xvfb-run';
    args = ['-a', '-s', '-screen 0 1600x1000x24', electron, ...args];
  }
  const t0 = Date.now();
  const res = spawnSync(cmd, args, {
    env: { ...process.env, BENCH_CONFIG: configFile, BENCH_DIST: dist },
    stdio: ['ignore', 'ignore', quiet ? 'ignore' : 'inherit'],
    timeout: 60 * 60 * 1000,
  });
  if (res.error) throw res.error;
  const rendered = JSON.parse(readFileSync(join(out, 'render.json'), 'utf8'));
  if (rendered.error) throw new Error(`render failed: ${rendered.error}`);
  log(`rendered ${renderPairs.length} pairs in ${Math.round((Date.now() - t0) / 1000)} s (tracking on ${rendered.delegate || '-'})`);
  return rendered;
}

const readRgba = (file) => new Uint8Array(readFileSync(file));

const needInputs = pairs.some((p) => p.writeOrig || p.writeRef);
let rendered = { results: [] };
if (engine === 'lite') rendered = await render('lite', pairs);
else if (needInputs || engine === 'local') {
  // Other engines only need the inputs; local also needs the raw frames every time.
  await render('inputs', pairs.map((p) => ({ ...p, writeOrig: p.writeOrig || engine === 'local', writeRef: p.writeRef || engine === 'local' })));
}

// ---------- motion mode: landmark accuracy over the moving frames, identity on a few of them ----------
if (motion) {
  const { pairMetrics, summariseMotion } = require('./motion.cjs');
  const avg = (v) => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : undefined);
  const rows = [];
  for (const [index, p] of pairs.entries()) {
    const id = `${p.split}:${p.target}`;
    const r = rendered.results.find((x) => x.id === id) || {};
    const row = { id, ref: p.ref, state: r.state, renderMs: r.renderMs };
    const trackFile = join(out, `${p.target}.motion.json`);
    if (!r.motion || r.motion.error || !existsSync(trackFile)) {
      row.error = (r.motion && r.motion.error) || `no output (${r.state || 'not rendered'})`;
      rows.push(row);
      continue;
    }
    Object.assign(row, pairMetrics(JSON.parse(readFileSync(trackFile, 'utf8'))));
    const ref = await scorer.cached(key('ref', p), () => scorer.analyse(readRgba(join(out, `${p.ref}.ref.rgba`)), 512, 512));
    row.frames = [];
    for (const k of r.motion.scored) {
      const base = join(out, `${p.target}.m${k}`);
      const o = await scorer.analyse(readRgba(`${base}.out.rgba`), W, H);
      const g = await scorer.analyse(readRgba(`${base}.orig.rgba`), W, H);
      const fr = { k };
      if (o && g && ref) Object.assign(fr, { idRef: cos(o.emb, ref.emb), idOrig: cos(o.emb, g.emb), pose: poseError(o.kps, g.kps) });
      else fr.error = 'no face';
      row.frames.push(fr);
      if (index >= imageCount) for (const kind of ['out', 'orig']) rmSync(`${base}.${kind}.rgba`, { force: true });
    }
    const still = row.frames.find((f) => f.k === 0 && f.idRef !== undefined);
    const moving = row.frames.filter((f) => f.k > 0 && f.idRef !== undefined);
    if (still) row.idRef0 = still.idRef;
    if (moving.length) {
      row.idRefFrames = avg(moving.map((f) => f.idRef));
      row.idRefFramesMin = Math.min(...moving.map((f) => f.idRef));
      row.idOrigFrames = avg(moving.map((f) => f.idOrig));
      row.poseFrames = avg(moving.map((f) => f.pose));
    }
    rows.push(row);
  }
  const summary = { engine, split, params, motion, ...summariseMotion(rows) };
  const ms = rows.map((r) => r.renderMs).filter((v) => v !== undefined);
  if (ms.length) summary.renderMsMedian = ms.sort((a, b) => a - b)[Math.floor(ms.length / 2)];
  writeFileSync(join(out, 'results.json'), JSON.stringify({ summary, rows }, null, 2));
  if (imageCount > 0) {
    // One row per pair: the photo, then camera and result side by side at each scored frame.
    const pick = rows.filter((r) => r.frames).slice(0, imageCount);
    const script = `
import sys, json, os
from PIL import Image, ImageDraw
out, faces, W, H = sys.argv[1], sys.argv[2], 1280, 720
rows = json.loads(sys.argv[3]); pairs = {p['split']+':'+p['target']: p for p in json.loads(sys.argv[4])}
tile = 200
cols = 1 + 2 * max(len(r['frames']) for r in rows)
sheet = Image.new('RGB', (tile * cols, tile * len(rows)), (0, 0, 0))
d = ImageDraw.Draw(sheet)
for i, r in enumerate(rows):
    p = pairs[r['id']]; f = p['frame']; m = f['size'] * 0.18
    box = (int(f['x'] - m), int(f['y'] - m), int(f['x'] + f['size'] + m), int(f['y'] + f['size'] + m))
    sheet.paste(Image.open(faces + '/' + p['ref']).convert('RGB').resize((tile, tile)), (0, i * tile))
    for j, fr in enumerate(r['frames']):
        for c, kind in enumerate(['orig', 'out']):
            fn = out + '/' + p['target'] + '.m%d.%s.rgba' % (fr['k'], kind)
            if not os.path.exists(fn): continue
            im = Image.frombytes('RGBA', (W, H), open(fn, 'rb').read()).convert('RGB')
            x = tile * (1 + 2 * j + c)
            sheet.paste(im.crop(box).resize((tile, tile)), (x, i * tile))
            if kind == 'out':
                d.rectangle([x, i * tile, x + 120, i * tile + 16], fill=(0, 0, 0))
                d.text((x + 3, i * tile + 3), 'k%d id %s' % (fr['k'], ('%.2f' % fr['idRef']) if 'idRef' in fr else '-'), fill=(255, 255, 0))
sheet.save(out + '/motion.jpg', quality=88)
`;
    spawnSync('python3', ['-c', script, out, facesDir, JSON.stringify(pick), JSON.stringify(pairs)], { stdio: 'inherit' });
    summary.images = join(out, 'motion.jpg');
  }
  console.log(JSON.stringify(summary));
  process.exit(0);
}

// ---------- score ----------
// Models load only now, so they don't hold memory while Electron renders.
await scorer.load();
let local = null;
if (engine === 'local') {
  const ort = require('onnxruntime-node');
  const { FaceSwapper } = require('../../src/main/faceswap/swapper.js');
  local = new FaceSwapper(ort, modelsDir);
  await local.load({ preferred: 'cpu' });
}
const { pasteFaces } = require('../../src/main/faceswap/paste.js');

const rows = [];
for (const p of pairs) {
  const id = `${p.split}:${p.target}`;
  const r = (rendered.results || []).find((x) => x.id === id) || {};
  const row = { id, ref: p.ref, state: r.state, faceFound: r.faceFound, renderMs: r.renderMs };
  const ref = await scorer.cached(key('ref', p), () => scorer.analyse(readRgba(join(out, `${p.ref}.ref.rgba`)), 512, 512));
  const orig = await scorer.cached(key('orig', p), () => scorer.analyse(readRgba(join(out, `${p.target}.orig.rgba`)), W, H));
  if (!ref || !orig) {
    row.error = 'no face in input';
    rows.push(row);
    continue;
  }
  row.idBase = cos(orig.emb, ref.emb);
  let outFile = join(out, `${p.target}.out.rgba`);
  if (engine === 'original') outFile = join(out, `${p.target}.orig.rgba`);
  if (engine === 'local') {
    const frame = readRgba(join(out, `${p.target}.orig.rgba`));
    await local.setReference(readRgba(join(out, `${p.ref}.ref.rgba`)), 512, 512);
    const faces = await local.process(frame, W, H);
    pasteFaces(frame, W, H, faces);
    writeFileSync(outFile, Buffer.from(frame.buffer));
  }
  if (engine === 'original') {
    row.idRef = row.idBase;
    row.idOrig = 1;
    row.pose = 0;
    rows.push(row);
    continue;
  }
  if (!existsSync(outFile)) {
    row.error = `no output (${r.state || 'not rendered'})`;
    rows.push(row);
    continue;
  }
  const res = await scorer.analyse(readRgba(outFile), W, H);
  if (!res) {
    row.error = 'no face in output';
    rows.push(row);
    continue;
  }
  row.idRef = cos(res.emb, ref.emb);
  row.idOrig = cos(res.emb, orig.emb);
  row.pose = poseError(res.kps, orig.kps);
  row.detScore = res.score;
  rows.push(row);
}

const summary = { engine, split, params, ...summarise(rows) };
const ms = rows.map((r) => r.renderMs).filter((v) => v !== undefined);
if (ms.length) summary.renderMsMedian = ms.sort((a, b) => a - b)[Math.floor(ms.length / 2)];
writeFileSync(join(out, 'results.json'), JSON.stringify({ summary, rows }, null, 2));

// ---------- comparison images ----------
if (imageCount > 0) {
  const pick = rows.filter((r) => r.idRef !== undefined).slice(0, imageCount);
  const script = `
import sys, json
from PIL import Image, ImageDraw
out, faces, W, H = sys.argv[1], sys.argv[2], 1280, 720
rows = json.loads(sys.argv[3]); pairs = {p['split']+':'+p['target']: p for p in json.loads(sys.argv[4])}
tile = 240
sheet = Image.new('RGB', (tile * 3, tile * len(rows)), (0, 0, 0))
d = ImageDraw.Draw(sheet)
for i, r in enumerate(rows):
    p = pairs[r['id']]; f = p['frame']
    box = (f['x'] - 20, f['y'] - 20, f['x'] + f['size'] + 20, f['y'] + f['size'] + 20)
    ref = Image.open(faces + '/' + p['ref']).convert('RGB').resize((tile, tile))
    orig = Image.frombytes('RGBA', (W, H), open(out + '/' + p['target'] + '.orig.rgba', 'rb').read()).convert('RGB') if __import__('os').path.exists(out + '/' + p['target'] + '.orig.rgba') else None
    res = Image.frombytes('RGBA', (W, H), open(out + '/' + p['target'] + '.out.rgba', 'rb').read()).convert('RGB')
    sheet.paste(ref, (0, i * tile))
    if orig: sheet.paste(orig.crop(box).resize((tile, tile)), (tile, i * tile))
    sheet.paste(res.crop(box).resize((tile, tile)), (tile * 2, i * tile))
    d.rectangle([tile * 2, i * tile, tile * 2 + 92, i * tile + 16], fill=(0, 0, 0))
    d.text((tile * 2 + 3, i * tile + 3), 'id %.2f' % r['idRef'], fill=(255, 255, 0))
sheet.save(out + '/compare.jpg', quality=88)
`;
  spawnSync('python3', ['-c', script, out, facesDir, JSON.stringify(pick), JSON.stringify(pairs)], { stdio: 'inherit' });
  summary.images = join(out, 'compare.jpg');
}

console.log(JSON.stringify(summary));

// ---------- sweep: one summary per parameter set ----------
if (sweep && engine === 'lite') {
  const sweepRows = sweep.map(() => []);
  for (const p of pairs) {
    const id = `${p.split}:${p.target}`;
    const ref = await scorer.cached(key('ref', p), () => null);
    const orig = await scorer.cached(key('orig', p), () => null);
    for (let k = 0; k < sweep.length; k += 1) {
      const row = { id };
      const outFile = join(out, `${p.target}.out.${k}.rgba`);
      const res = ref && orig && existsSync(outFile) ? await scorer.analyse(readRgba(outFile), W, H) : null;
      if (!res) row.error = 'no face in output';
      else {
        row.idRef = cos(res.emb, ref.emb);
        row.idOrig = cos(res.emb, orig.emb);
        row.idBase = cos(orig.emb, ref.emb);
        row.pose = poseError(res.kps, orig.kps);
      }
      sweepRows[k].push(row);
    }
  }
  const all = sweep.map((set, k) => ({ sweep: k, params: { ...params, ...set }, ...summarise(sweepRows[k]) }));
  writeFileSync(join(out, 'sweep.json'), JSON.stringify({ all, rows: sweepRows }, null, 2));
  for (const s of all) console.log(JSON.stringify(s));
}
