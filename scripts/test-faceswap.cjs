// Offline check of the local face swap on still images (needs the models downloaded).
// Usage: node scripts/test-faceswap.cjs <modelsDir> <ref.rgba> <refW> <refH> <frame.rgba> <frameW> <frameH> <out.rgba> [other.rgba]
// Prints how much the swapped face now matches the reference identity (cosine similarity).
const fs = require('node:fs');
const ort = require('onnxruntime-node');
const { FaceSwapper } = require('../src/main/faceswap/swapper');
const { alignmentTarget, similarity, warpToTensor, l2normalize } = require('../src/main/faceswap/face-math');
const { pasteFaces } = require('../src/main/faceswap/paste');

async function embed(sw, rgba, w, h) {
  const faces = await sw.detector.detect(rgba, w, h);
  if (!faces.length) return null;
  const m = similarity(faces[0].kps, alignmentTarget(112));
  const t = warpToTensor(rgba, w, h, m, 112, 127.5, 127.5);
  const out = await sw.recognizer.run({ [sw.recognizer.inputNames[0]]: new ort.Tensor('float32', t, [1, 3, 112, 112]) });
  return l2normalize(out[sw.recognizer.outputNames[0]].data);
}

const cos = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

function padForTest(rgba, w, h, pad) {
  const W = w + pad * 2;
  const out = new Uint8Array(W * (h + pad * 2) * 4).fill(0);
  for (let y = 0; y < h; y += 1) out.set(rgba.subarray(y * w * 4, (y + 1) * w * 4), ((y + pad) * W + pad) * 4);
  return out;
}

(async () => {
  const [modelsDir, refPath, rw, rh, framePath, fw, fh, outPath, otherPath] = process.argv.slice(2);
  const ref = new Uint8Array(fs.readFileSync(refPath));
  const frame = new Uint8Array(fs.readFileSync(framePath));
  const W = Number(fw);
  const H = Number(fh);

  const sw = new FaceSwapper(ort, modelsDir);
  let t = Date.now();
  const { provider } = await sw.load({ preferred: 'cpu' });
  console.log(`models loaded on ${provider} in ${Date.now() - t} ms`);
  if (process.env.ENHANCE === '1') {
    await sw.setEnhance(true);
    console.log('face detail boost: on');
  }

  t = Date.now();
  const found = await sw.setReference(ref, Number(rw), Number(rh));
  console.log(`reference face found: ${found} (${Date.now() - t} ms)`);
  if (!found) process.exit(1);

  const times = [];
  let results = [];
  for (let i = 0; i < 3; i += 1) {
    t = Date.now();
    results = await sw.process(frame, W, H);
    times.push(Date.now() - t);
  }
  console.log(`faces swapped: ${results.length}, per frame (cpu): ${times.join(' / ')} ms`);
  if (!results.length) process.exit(1);

  const out = new Uint8ClampedArray(frame);
  pasteFaces(out, W, H, results);
  fs.writeFileSync(outPath, Buffer.from(out.buffer));

  const eRef = await embed(sw, ref, Number(rw), Number(rh));
  const eBefore = await embed(sw, frame, W, H);
  const eAfter = await embed(sw, new Uint8Array(out.buffer), W, H);
  console.log(`identity match with reference: before ${cos(eRef, eBefore).toFixed(3)}, after ${cos(eRef, eAfter).toFixed(3)}`);
  if (otherPath) {
    // Control: the swapper's reference path (with its padding retry) on a tight close-up of someone else.
    const other = new Uint8Array(fs.readFileSync(otherPath));
    const found = await sw.setReference(other, 512, 512);
    console.log(`control close-up detected: ${found}`);
    const eOther = found ? await embed(sw, padForTest(other, 512, 512, 128), 768, 768) : null;
    if (eOther) console.log(`control (unrelated person vs swapped): ${cos(eOther, eAfter).toFixed(3)}`);
  }
  const ok = cos(eRef, eAfter) > 0.4 && cos(eRef, eAfter) > cos(eRef, eBefore) + 0.2;
  console.log(ok ? 'FACESWAP TEST PASSED' : 'FACESWAP TEST FAILED');
  process.exit(ok ? 0 : 1);
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
