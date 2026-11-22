// Re-scores the point smoothing offline on tracks recorded by `bench-lite.mjs --motion`, in about a
// second instead of re-rendering: MediaPipe's raw points don't depend on the smoothing, so the
// engine's own PointSmoother is replayed over them with the frames' recorded timestamps.
//
//   node scripts/bench/motion-replay.mjs <out dir> [more dirs] [--params '{"minCutoff":2}']
//        [--grid '[{"beta":6},{"beta":12}]']     one summary line per settings
import { readdirSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));
const { pairMetrics, summariseMotion } = require('./motion.cjs');
const { PointSmoother, liteTuning } = await import('../../src/renderer/engine/ai-lite.js');

const argv = process.argv.slice(2);
const opt = (name, fallback) => (argv.includes(`--${name}`) ? argv[argv.indexOf(`--${name}`) + 1] : fallback);
const dirs = argv.filter((a, i) => !a.startsWith('--') && !(i > 0 && argv[i - 1].startsWith('--')));
const base = JSON.parse(opt('params', '{}'));
const grid = JSON.parse(opt('grid', '[{}]'));
const defaults = { ...liteTuning };

const tracks = dirs.flatMap((d) =>
  readdirSync(d)
    .filter((f) => f.endsWith('.motion.json'))
    .map((f) => JSON.parse(readFileSync(join(d, f), 'utf8'))),
);
if (!tracks.length) throw new Error(`no *.motion.json in ${dirs.join(', ') || '(no dir given)'} (see ${here}/README.md)`);

for (const settings of grid) {
  Object.assign(liteTuning, defaults, base, settings);
  const rows = tracks.map((frames) => {
    const smoother = new PointSmoother();
    return pairMetrics(
      frames.map((f) => {
        const pts = smoother.update(f.found ? Float32Array.from(f.raw) : null, f.t);
        return { ...f, pts: pts && Array.from(pts), strength: pts ? smoother.strength : 0 };
      }),
    );
  });
  const { pairs, lag, jitter, err, jitterHold, rawLag, rawJitter, popMax, backErr, backFade } = summariseMotion(rows);
  console.log(JSON.stringify({ params: { ...base, ...settings }, pairs, lag, jitter, err, jitterHold, rawLag, rawJitter, popMax, backErr, backFade }));
}
