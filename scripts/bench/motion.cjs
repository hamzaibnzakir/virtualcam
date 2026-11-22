// Motion metrics for the Lite benchmark (bench-lite.mjs --motion). Each pair is animated with a known
// head transform; the tracked points the engine drew are compared with the still start's points moved
// by that transform. All distances are a share of the eye distance, over all 468 mesh points.
//   err     - mean distance from the true points
//   lag     - the smooth part of the error (a 7-frame Savitzky-Golay fit of it): smoothing delay plus
//             the tracker's own delay and bias, so compare with rawLag
//   jitter  - the frame-to-frame noise: RMS of what that fit leaves over, scaled so it equals the RMS
//             noise for white noise; smooth drift (even a changing lag) hardly counts
//   raw*    - the same for MediaPipe's points before smoothing (the floor the smoothing starts from)
//   *Hold   - only the frames where the head holds still after moving
//   lost    - frames (not counting the black dropout frames) where tracking found no face
//   popMax  - largest one-frame change in face opacity (1 = the face pops in or out)
//   backErr - error over the first 4 frames after the face comes back from the dropout
//   backFade- frames from the face coming back until it is fully shown
const EYE = [33, 133, 362, 263];

function eyeDist(p) {
  const rx = (p[EYE[0] * 2] + p[EYE[1] * 2]) / 2;
  const ry = (p[EYE[0] * 2 + 1] + p[EYE[1] * 2 + 1]) / 2;
  const lx = (p[EYE[2] * 2] + p[EYE[3] * 2]) / 2;
  const ly = (p[EYE[2] * 2 + 1] + p[EYE[3] * 2 + 1]) / 2;
  return Math.hypot(lx - rx, ly - ry) || 1;
}

function errors(frames, key) {
  return frames.map((f) => {
    if (!f.gt || !f[key]) return null;
    const eye = eyeDist(f.gt);
    const e = new Float64Array(f.gt.length);
    for (let i = 0; i < e.length; i += 1) e[i] = (f[key][i] - f.gt[i]) / eye;
    return e;
  });
}

// 7-frame quadratic Savitzky-Golay smoothing weights; the centre weight is 7/21.
const SG = [-2, 3, 6, 7, 6, 3, -2].map((w) => w / 21);
const SG_NOISE = 1 - 7 / 21; // share of white-noise power left in the residual

// Error, lag and jitter over frames [from, to).
function trackStats(frames, key, from, to) {
  const e = errors(frames, key);
  const ok = (k) => k >= from && k < to && e[k];
  let errSum = 0;
  let errN = 0;
  let lagSum = 0;
  let jitSq = 0;
  let fitN = 0;
  for (let k = from; k < to; k += 1) {
    if (!e[k]) continue;
    const n = e[k].length / 2;
    for (let i = 0; i < n; i += 1) errSum += Math.hypot(e[k][2 * i], e[k][2 * i + 1]);
    errN += n;
    let full = true;
    for (let j = -3; j <= 3; j += 1) full = full && ok(k + j);
    if (!full) continue;
    for (let i = 0; i < n; i += 1) {
      let fx = 0;
      let fy = 0;
      for (let j = -3; j <= 3; j += 1) {
        fx += SG[j + 3] * e[k + j][2 * i];
        fy += SG[j + 3] * e[k + j][2 * i + 1];
      }
      lagSum += Math.hypot(fx, fy);
      const rx = e[k][2 * i] - fx;
      const ry = e[k][2 * i + 1] - fy;
      jitSq += rx * rx + ry * ry;
    }
    fitN += n;
  }
  return {
    err: errN ? errSum / errN : null,
    lag: fitN ? lagSum / fitN : null,
    jitter: fitN ? Math.sqrt(jitSq / fitN / SG_NOISE) : null,
  };
}

// Landmark metrics for one pair's frames (as written by the renderer).
function pairMetrics(allFrames) {
  const frames = allFrames.filter((f) => f.phase !== 'warmup');
  const n = frames.findIndex((f) => f.phase === 'dropout' || f.phase === 'back');
  const end = n < 0 ? frames.length : n;
  const holdFrom = frames.findIndex((f) => f.phase === 'hold');
  const s = trackStats(frames, 'pts', 0, end);
  const raw = trackStats(frames, 'raw', 0, end);
  const hold = trackStats(frames, 'pts', holdFrom, end);
  const rawHold = trackStats(frames, 'raw', holdFrom, end);
  const m = {
    err: s.err,
    lag: s.lag,
    jitter: s.jitter,
    jitterHold: hold.jitter,
    rawErr: raw.err,
    rawLag: raw.lag,
    rawJitter: raw.jitter,
    rawJitterHold: rawHold.jitter,
    lost: frames.filter((f) => f.phase !== 'dropout' && !f.found).length,
  };
  let pop = 0;
  for (let k = 1; k < frames.length; k += 1) pop = Math.max(pop, Math.abs(frames[k].strength - frames[k - 1].strength));
  m.popMax = pop;
  const back = frames.findIndex((f) => f.phase === 'back');
  if (back > 0) {
    m.backErr = trackStats(frames, 'pts', back, Math.min(frames.length, back + 4)).err;
    let k = back;
    while (k < frames.length && frames[k].strength < 0.99) k += 1;
    m.backFade = k - back;
  }
  return m;
}

function mean(rows, key) {
  const v = rows.map((r) => r[key]).filter((x) => typeof x === 'number' && Number.isFinite(x));
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}

function summariseMotion(rows) {
  const ok = rows.filter((r) => !r.error);
  const out = { pairs: rows.length, scored: ok.length, failures: rows.length - ok.length };
  const keys = ['lag', 'jitter', 'err', 'jitterHold', 'rawLag', 'rawJitter', 'rawErr', 'rawJitterHold', 'lost', 'popMax', 'backErr', 'backFade', 'idRef0', 'idRefFrames', 'idRefFramesMin', 'poseFrames', 'idOrigFrames'];
  for (const k of keys) {
    const v = mean(ok, k);
    if (v !== null) out[k] = Number(v.toFixed(4));
  }
  return out;
}

module.exports = { eyeDist, trackStats, pairMetrics, summariseMotion };
