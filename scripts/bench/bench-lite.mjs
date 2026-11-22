// Lite engine benchmark. Renders every pair with the real Lite engine, then scores it.
//
//   node scripts/bench/bench-lite.mjs [--split train|holdout|all] [--limit N] [--engine lite|local|original]
//        [--params '{"shading":0.3}'] [--out DIR] [--images N] [--frames N] [--only i,j,...] [--quiet]
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
await scorer.load();
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
  writeFileSync(configFile, JSON.stringify({ mode, pairs: renderPairs, params, frames, facesDir, outDir: out, assetsDir: dist }));
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

// ---------- score ----------
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
