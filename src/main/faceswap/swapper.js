'use strict';

// Local face swap: detect your face, then redraw it with the identity from a reference photo.
// Runs on the GPU through DirectML on Windows (any DirectX 12 GPU), otherwise on the CPU.
const fs = require('node:fs');
const path = require('node:path');
const { FaceDetector } = require('./detector');
const { extractEmap } = require('./onnx-initializer');
const {
  alignmentTarget,
  similarity,
  invertAffine,
  compose,
  warpToTensor,
  warpRgba,
  l2normalize,
} = require('./face-math');
const { featherMask } = require('./paste');

const MODEL_FILES = {
  detector: 'det_10g.onnx',
  recognizer: 'w600k_r50.onnx',
  swapper: 'inswapper_128_fp16.onnx',
};
// Optional sharpening pass for the swapped face.
const ENHANCER_FILE = 'GFPGANv1.4.onnx';
const SWAP_SIZE = 128;
const REC_SIZE = 112;
const ENH_SIZE = 512;
// Where the enhancer expects eyes, nose and mouth corners in its 512x512 input.
const FFHQ_512 = [
  [192.98138, 239.94708],
  [318.90277, 240.1936],
  [256.63416, 314.01935],
  [201.26117, 371.41043],
  [313.08905, 371.15118],
];

function padImage(rgba, width, height, pad) {
  const w = width + pad * 2;
  const h = height + pad * 2;
  const out = new Uint8Array(w * h * 4);
  for (let i = 3; i < out.length; i += 4) out[i] = 255;
  for (let y = 0; y < height; y += 1) {
    out.set(rgba.subarray(y * width * 4, (y + 1) * width * 4), ((y + pad) * w + pad) * 4);
  }
  return { rgba: out, width: w, height: h };
}

function providerList(preferred) {
  if (preferred === 'cpu') return [['cpu']];
  if (process.platform === 'win32') return [['dml'], ['cpu']];
  if (process.platform === 'darwin') return [['coreml'], ['cpu']];
  return [['cpu']];
}

class FaceSwapper {
  constructor(ort, modelsDir) {
    this.ort = ort;
    this.modelsDir = modelsDir;
    this.provider = null;
    this.detector = null;
    this.recognizer = null;
    this.swapper = null;
    this.emap = null;
    this.latent = null; // identity of the reference photo, ready for the swapper
    this.enhancer = null;
    this.enhance = false;
    this.swapBuffer = new Float32Array(3 * SWAP_SIZE * SWAP_SIZE);
    this.recBuffer = new Float32Array(3 * REC_SIZE * REC_SIZE);
    this.mask128 = featherMask(SWAP_SIZE);
    // The swapped 128 crop -> enhancer 512 alignment never changes, so work it out once.
    this.toEnh = similarity(alignmentTarget(SWAP_SIZE), FFHQ_512);
    this.fromEnh = invertAffine(this.toEnh);
    this.mask512 = null;
  }

  hasEnhancerFile() {
    return fs.existsSync(path.join(this.modelsDir, ENHANCER_FILE));
  }

  async load({ preferred = 'auto', detSize = 640 } = {}) {
    const file = (key) => path.join(this.modelsDir, MODEL_FILES[key]);
    for (const key of Object.keys(MODEL_FILES)) {
      if (!fs.existsSync(file(key))) throw new Error(`Missing model file ${MODEL_FILES[key]}`);
    }

    let lastError = null;
    this.fallbackReason = '';
    for (const providers of providerList(preferred)) {
      try {
        const options = {
          executionProviders: providers,
          graphOptimizationLevel: 'all',
          logSeverityLevel: 3,
        };
        if (providers[0] === 'dml') {
          // DirectML requires these two settings.
          options.enableMemPattern = false;
          options.executionMode = 'sequential';
        }
        const [det, rec, swap] = await Promise.all([
          this.ort.InferenceSession.create(file('detector'), options),
          this.ort.InferenceSession.create(file('recognizer'), options),
          this.ort.InferenceSession.create(file('swapper'), options),
        ]);
        this.detector = new FaceDetector(this.ort, det, { size: detSize });
        this.recognizer = rec;
        this.swapper = swap;
        this.provider = providers[0];
        this.sessionOptions = options;
        break;
      } catch (err) {
        lastError = err;
        this.fallbackReason = `${providers[0]}: ${(err && err.message) || err}`.slice(0, 300);
      }
    }
    if (!this.provider) throw lastError || new Error('Could not load the face models');

    this.emap = this.loadEmap(file('swapper'));
    return { provider: this.provider, fallbackReason: this.fallbackReason };
  }

  // Turns the sharpening pass on or off; loads its model the first time.
  async setEnhance(on) {
    if (on && !this.enhancer) {
      if (!this.hasEnhancerFile()) throw new Error('The face detail model is not downloaded');
      this.enhancer = await this.ort.InferenceSession.create(path.join(this.modelsDir, ENHANCER_FILE), this.sessionOptions);
      // Soft-edge mask in enhancer space: the 128 mask carried through the same alignment.
      const m = new Uint8ClampedArray(SWAP_SIZE * SWAP_SIZE * 4);
      for (let i = 0; i < this.mask128.length; i += 1) m[i * 4 + 3] = this.mask128[i] * 255;
      const warped = warpRgba(m, SWAP_SIZE, SWAP_SIZE, this.toEnh, ENH_SIZE);
      this.mask512 = new Float32Array(ENH_SIZE * ENH_SIZE);
      for (let i = 0; i < this.mask512.length; i += 1) this.mask512[i] = warped[i * 4 + 3] / 255;
    }
    this.enhance = Boolean(on);
    return this.enhance;
  }

  // The identity matrix is cached next to the models after the first extraction.
  loadEmap(swapperPath) {
    const cache = path.join(this.modelsDir, 'inswapper_emap.f32');
    const expected = 512 * 512 * 4;
    try {
      const data = fs.readFileSync(cache);
      if (data.length === expected) {
        return new Float32Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.length));
      }
    } catch {
      // not cached yet
    }
    const emap = extractEmap(fs.readFileSync(swapperPath));
    fs.writeFileSync(cache, Buffer.from(emap.buffer));
    return emap;
  }

  // Sets the person to become. Returns false when no face is found in the photo.
  async setReference(rgba, width, height) {
    let image = { rgba, width, height };
    let faces = await this.detector.detect(rgba, width, height);
    if (!faces.length) {
      // Tight close-ups (face touching the edges) detect better with a border around them.
      image = padImage(rgba, width, height, Math.round(Math.max(width, height) * 0.25));
      faces = await this.detector.detect(image.rgba, image.width, image.height);
    }
    if (!faces.length) {
      this.latent = null;
      return false;
    }
    ({ rgba, width, height } = image);
    const m = similarity(faces[0].kps, alignmentTarget(REC_SIZE));
    const tensor = warpToTensor(rgba, width, height, m, REC_SIZE, 127.5, 127.5, this.recBuffer);
    const input = new this.ort.Tensor('float32', tensor, [1, 3, REC_SIZE, REC_SIZE]);
    const out = await this.recognizer.run({ [this.recognizer.inputNames[0]]: input });
    const embedding = l2normalize(out[this.recognizer.outputNames[0]].data);

    // latent = normalize(embedding x emap)
    const latent = new Float32Array(512);
    for (let j = 0; j < 512; j += 1) {
      let sum = 0;
      for (let i = 0; i < 512; i += 1) sum += embedding[i] * this.emap[i * 512 + j];
      latent[j] = sum;
    }
    this.latent = l2normalize(latent);
    return true;
  }

  clearReference() {
    this.latent = null;
  }

  // Swaps the largest face (or up to maxFaces) in a frame.
  // Returns [{ rgba, size, toFrame: [a,c,e,b,d,f] }]: a square RGBA face whose alpha already holds
  // the soft edge, plus the transform that places it on the frame.
  async process(rgba, width, height, { maxFaces = 1 } = {}) {
    if (!this.latent) return [];
    const faces = (await this.detector.detect(rgba, width, height)).slice(0, maxFaces);
    const results = [];
    for (const face of faces) {
      const m = similarity(face.kps, alignmentTarget(SWAP_SIZE));
      const target = warpToTensor(rgba, width, height, m, SWAP_SIZE, 0, 255, this.swapBuffer);
      const out = await this.swapper.run({
        target: new this.ort.Tensor('float32', target, [1, 3, SWAP_SIZE, SWAP_SIZE]),
        source: new this.ort.Tensor('float32', this.latent, [1, 512]),
      });
      const pred = out[this.swapper.outputNames[0]].data;
      const plane = SWAP_SIZE * SWAP_SIZE;
      const face128 = new Uint8ClampedArray(plane * 4);
      for (let i = 0; i < plane; i += 1) {
        face128[i * 4] = pred[i] * 255;
        face128[i * 4 + 1] = pred[plane + i] * 255;
        face128[i * 4 + 2] = pred[2 * plane + i] * 255;
        face128[i * 4 + 3] = this.mask128[i] * 255;
      }
      const toFrame = invertAffine(m);
      if (this.enhance && this.enhancer) {
        results.push(await this.enhanceFace(face128, toFrame));
      } else {
        results.push({ rgba: face128, size: SWAP_SIZE, toFrame });
      }
    }
    return results;
  }

  async enhanceFace(face128, toFrame) {
    const opaque = new Uint8ClampedArray(face128);
    for (let i = 3; i < opaque.length; i += 4) opaque[i] = 255;
    const input = warpToTensor(opaque, SWAP_SIZE, SWAP_SIZE, this.toEnh, ENH_SIZE, 127.5, 127.5);
    const out = await this.enhancer.run({
      [this.enhancer.inputNames[0]]: new this.ort.Tensor('float32', input, [1, 3, ENH_SIZE, ENH_SIZE]),
    });
    const pred = out[this.enhancer.outputNames[0]].data;
    const plane = ENH_SIZE * ENH_SIZE;
    const face = new Uint8ClampedArray(plane * 4);
    for (let i = 0; i < plane; i += 1) {
      face[i * 4] = (pred[i] + 1) * 127.5;
      face[i * 4 + 1] = (pred[plane + i] + 1) * 127.5;
      face[i * 4 + 2] = (pred[2 * plane + i] + 1) * 127.5;
      face[i * 4 + 3] = this.mask512[i] * 255;
    }
    return { rgba: face, size: ENH_SIZE, toFrame: compose(toFrame, this.fromEnh) };
  }
}

module.exports = { FaceSwapper, MODEL_FILES, ENHANCER_FILE };
