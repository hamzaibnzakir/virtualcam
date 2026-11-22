// Scores how much a face in an image looks like a reference person (cosine similarity of ArcFace identities).
// Usage: node scripts/identity-score.cjs <modelsDir> <ref.rgba> <w> <h> <img.rgba> <w> <h>
const fs = require('node:fs');
const ort = require('onnxruntime-node');
const { FaceSwapper } = require('../src/main/faceswap/swapper');
const { alignmentTarget, similarity, warpToTensor, l2normalize } = require('../src/main/faceswap/face-math');

async function embed(sw, rgba, w, h) {
  const faces = await sw.detector.detect(rgba, w, h);
  if (!faces.length) return null;
  const m = similarity(faces[0].kps, alignmentTarget(112));
  const t = warpToTensor(rgba, w, h, m, 112, 127.5, 127.5);
  const out = await sw.recognizer.run({ [sw.recognizer.inputNames[0]]: new ort.Tensor('float32', t, [1, 3, 112, 112]) });
  return l2normalize(out[sw.recognizer.outputNames[0]].data);
}

(async () => {
  const [dir, refPath, rw, rh, imgPath, iw, ih] = process.argv.slice(2);
  const sw = new FaceSwapper(ort, dir);
  await sw.load({ preferred: 'cpu' });
  const a = await embed(sw, new Uint8Array(fs.readFileSync(refPath)), Number(rw), Number(rh));
  const b = await embed(sw, new Uint8Array(fs.readFileSync(imgPath)), Number(iw), Number(ih));
  if (!a || !b) {
    console.log('no face found');
    process.exit(1);
  }
  console.log((a.reduce((s, v, i) => s + v * b[i], 0)).toFixed(3));
})();
