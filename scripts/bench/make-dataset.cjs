// Builds the Lite benchmark dataset from the SFHQ sample (AI-generated faces, no real people).
// Keeps only faces the age model reads as clearly adult, then makes reference/target pairs
// with a fixed train / holdout split. Usage:
//   node scripts/bench/make-dataset.cjs <sfhqDir> <modelsDir> <out.json>
// modelsDir needs det_10g.onnx, w600k_r50.onnx, inswapper_128_fp16.onnx (to load the detector) and genderage.onnx.
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const ort = require('onnxruntime-node');
const { FaceSwapper } = require('../../src/main/faceswap/swapper');
const { warpToTensor } = require('../../src/main/faceswap/face-math');

const MIN_AGE = 25; // generous margin above 18
// Excluded after a manual look at every face: anyone who could read as under 18 is left out,
// even when the age model says otherwise (it let two children through).
const EXCLUDE = new Set(["SFHQ_pt3_00000399.jpg", "SFHQ_pt3_00000598.jpg", "SFHQ_pt3_00000741.jpg", "SFHQ_pt3_00000801.jpg", "SFHQ_pt3_00001098.jpg", "SFHQ_pt3_00001157.jpg", "SFHQ_pt3_00001415.jpg", "SFHQ_pt3_00001601.jpg", "SFHQ_pt3_00001621.jpg", "SFHQ_pt3_00001629.jpg", "SFHQ_pt3_00002309.jpg", "SFHQ_pt3_00002432.jpg", "SFHQ_pt3_00004727.jpg"]);

function decode(file) {
  // RGBA via Python/Pillow (dev-time tool only).
  const out = execFileSync('python3', ['-c', `
import sys
from PIL import Image
# Close-ups detect better with a border, so the face sits at 360 px inside a 512 px canvas.
face = Image.open(sys.argv[1]).convert('RGBA').resize((360, 360))
im = Image.new('RGBA', (512, 512), (0, 0, 0, 255))
im.paste(face, (76, 76))
sys.stdout.buffer.write(im.tobytes())`, file], { maxBuffer: 64 * 1024 * 1024 });
  return new Uint8Array(out);
}

(async () => {
  const [dir, modelsDir, outFile] = process.argv.slice(2);
  const sw = new FaceSwapper(ort, modelsDir);
  await sw.load({ preferred: 'cpu' });
  const ageModel = await ort.InferenceSession.create(path.join(modelsDir, 'genderage.onnx'), { logSeverityLevel: 3 });

  const files = fs.readdirSync(dir).filter((f) => /\.jpe?g$/i.test(f)).sort();
  const faces = [];
  for (const file of files) {
    const rgba = decode(path.join(dir, file));
    const det = await sw.detector.detect(rgba, 512, 512);
    if (!det.length || det[0].score < 0.7) continue;
    const [x1, y1, x2, y2] = det[0].box;
    const size = 96;
    const s = size / (Math.max(x2 - x1, y2 - y1) * 1.5);
    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;
    const m = [s, 0, size / 2 - s * cx, 0, s, size / 2 - s * cy];
    const t = warpToTensor(rgba, 512, 512, m, size, 0, 1);
    const out = await ageModel.run({ [ageModel.inputNames[0]]: new ort.Tensor('float32', t, [1, 3, size, size]) });
    const pred = out[ageModel.outputNames[0]].data;
    const age = Math.round(pred[2] * 100);
    faces.push({ file, age, det: Number(det[0].score.toFixed(3)) });
    process.stdout.write(`${file} age~${age}\n`);
  }
  const adults = faces.filter((f) => f.age >= MIN_AGE && !EXCLUDE.has(f.file));

  // Deterministic shuffle, then split identities so holdout faces are never seen in training.
  let seed = 12345;
  const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
  const pool = adults.slice().sort(() => rand() - 0.5);
  const cut = Math.round(pool.length * 0.65);
  const split = { train: pool.slice(0, cut), holdout: pool.slice(cut) };

  // Pairs: each face is a "camera" target once, wearing another face from the same split.
  // Webcam-like placement: face size, position and a little head tilt vary.
  const pairs = [];
  for (const name of ['train', 'holdout']) {
    const list = split[name];
    list.forEach((target, i) => {
      const ref = list[(i + 1 + (i % 3)) % list.length];
      if (ref.file === target.file) return;
      const size = 420 + Math.round(rand() * 260);
      pairs.push({
        split: name,
        ref: ref.file,
        target: target.file,
        frame: { size, x: Math.round(640 - size / 2 + (rand() - 0.5) * 300), y: Math.round(360 - size / 2 + (rand() - 0.5) * 120), tilt: Math.round((rand() - 0.5) * 16) },
      });
    });
  }
  fs.writeFileSync(outFile, JSON.stringify({ source: 'bitmind/SyntheticFacesHQ tiny-sample.zip', minAge: MIN_AGE, excluded: [...EXCLUDE], faces, pairs }, null, 2));
  console.log(`faces ${faces.length}, adults ${adults.length}, pairs train ${pairs.filter((p) => p.split === 'train').length}, holdout ${pairs.filter((p) => p.split === 'holdout').length}`);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
