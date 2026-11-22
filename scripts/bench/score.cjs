// Scoring for the Lite benchmark, using InsightFace models:
//   idRef   - identity match between the output face and the photo (ArcFace cosine; higher is better)
//   idOrig  - identity match between the output face and the original camera face (lower = more changed)
//   pose    - how far the output's eyes/nose/mouth points moved from the camera's, as a share of the
//             eye distance. Small = head pose and expression still follow the camera (no pasting the photo).
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const ort = require('onnxruntime-node');
const { FaceSwapper } = require('../../src/main/faceswap/swapper');
const { alignmentTarget, similarity, warpToTensor, l2normalize } = require('../../src/main/faceswap/face-math');

const W = 1280;
const H = 720;

class Scorer {
  constructor(modelsDir, cacheDir) {
    this.modelsDir = modelsDir;
    this.cacheDir = cacheDir;
    fs.mkdirSync(cacheDir, { recursive: true });
  }

  async load() {
    this.sw = new FaceSwapper(ort, this.modelsDir);
    await this.sw.load({ preferred: 'cpu' });
  }

  async analyse(rgba, w, h) {
    const faces = await this.sw.detector.detect(rgba, w, h);
    if (!faces.length) return null;
    const face = faces[0];
    const m = similarity(face.kps, alignmentTarget(112));
    const t = warpToTensor(rgba, w, h, m, 112, 127.5, 127.5);
    const rec = this.sw.recognizer;
    const out = await rec.run({ [rec.inputNames[0]]: new ort.Tensor('float32', t, [1, 3, 112, 112]) });
    return { emb: Array.from(l2normalize(out[rec.outputNames[0]].data)), kps: face.kps, score: face.score };
  }

  // Cached: references and original frames never change between runs.
  async cached(key, make) {
    const file = path.join(this.cacheDir, `${crypto.createHash('sha1').update(key).digest('hex')}.json`);
    if (fs.existsSync(file)) return JSON.parse(fs.readFileSync(file, 'utf8'));
    const value = await make();
    fs.writeFileSync(file, JSON.stringify(value));
    return value;
  }
}

const cos = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

function poseError(a, b) {
  const eye = Math.hypot(b[1][0] - b[0][0], b[1][1] - b[0][1]) || 1;
  let sum = 0;
  for (let i = 0; i < 5; i += 1) sum += Math.hypot(a[i][0] - b[i][0], a[i][1] - b[i][1]);
  return sum / 5 / eye;
}

function summarise(rows) {
  const ok = rows.filter((r) => r.idRef !== undefined);
  const mean = (k) => (ok.length ? ok.reduce((s, r) => s + r[k], 0) / ok.length : 0);
  const sorted = ok.map((r) => r.idRef).sort((a, b) => a - b);
  return {
    pairs: rows.length,
    scored: ok.length,
    failures: rows.length - ok.length,
    idRef: Number(mean('idRef').toFixed(4)),
    idRefMedian: Number((sorted[Math.floor(sorted.length / 2)] || 0).toFixed(4)),
    idOrig: Number(mean('idOrig').toFixed(4)),
    idBase: Number(mean('idBase').toFixed(4)),
    pose: Number(mean('pose').toFixed(4)),
    matchRate: Number((ok.filter((r) => r.idRef >= 0.4).length / Math.max(1, rows.length)).toFixed(3)),
  };
}

module.exports = { Scorer, cos, poseError, summarise, W, H };
