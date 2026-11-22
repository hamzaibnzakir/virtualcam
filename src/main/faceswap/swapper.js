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

// Where the models can run, in the order they are tried. On Windows a laptop often has two graphics
// chips (one built into the processor, one dedicated), and DirectML numbers them 0, 1 in an order
// that depends on which drives the screen, so both are tried and the faster one wins.
// gpuCount is how many graphics chips Windows reported (0 when unknown); with just one, chip 2 would be
// Windows' slow software stand-in, so it isn't tried.
function deviceCandidates(preferred, gpuCount = 0) {
  if (preferred === 'cpu') return [{ name: 'cpu' }];
  if (process.platform === 'win32') {
    const gpus = gpuCount === 1 ? [0] : [0, 1];
    return [...gpus.map((deviceId) => ({ name: 'dml', deviceId })), { name: 'cpu' }];
  }
  if (process.platform === 'darwin') return [{ name: 'coreml' }, { name: 'cpu' }];
  return [{ name: 'cpu' }];
}

function sessionOptionsFor(device) {
  const options = {
    executionProviders: [device.name === 'dml' ? { name: 'dml', deviceId: device.deviceId } : device.name],
    graphOptimizationLevel: 'all',
    logSeverityLevel: 3,
  };
  if (device.name === 'dml') {
    // DirectML requires these two settings.
    options.enableMemPattern = false;
    options.executionMode = 'sequential';
  }
  return options;
}

function deviceLabel(device) {
  if (!device) return 'none';
  if (device.name === 'dml') return `graphics chip ${device.deviceId + 1}`;
  if (device.name === 'coreml') return 'your Mac';
  return 'the processor';
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

  // Times one swap on a device: the first run sets the device up (it can take seconds on a graphics
  // chip), the second is the real speed.
  async timeSwap(session) {
    const feeds = {
      target: new this.ort.Tensor('float32', new Float32Array(3 * SWAP_SIZE * SWAP_SIZE).fill(0.5), [1, 3, SWAP_SIZE, SWAP_SIZE]),
      source: new this.ort.Tensor('float32', new Float32Array(512).fill(0.04), [1, 512]),
    };
    let t = Date.now();
    await session.run(feeds);
    const first = Date.now() - t;
    if (first > 15000) return first; // hopeless for live video; no need for a second run
    t = Date.now();
    await session.run(feeds);
    return Date.now() - t;
  }

  async load({ preferred = 'auto', gpuCount = 0, detSize = 640, onStep = () => {}, log = () => {} } = {}) {
    const file = (key) => path.join(this.modelsDir, MODEL_FILES[key]);
    for (const key of Object.keys(MODEL_FILES)) {
      if (!fs.existsSync(file(key))) throw new Error(`Missing model file ${MODEL_FILES[key]}`);
    }

    // Try each device with the swap model (the heavy one) and keep the fastest.
    const candidates = deviceCandidates(preferred, gpuCount);
    this.tried = [];
    let best = null;
    for (const device of candidates) {
      const label = deviceLabel(device);
      // A graphics chip that swaps this fast beats any processor, so skip timing the processor.
      if (device.name === 'cpu' && best && best.ms < 60) break;
      onStep(`testing ${label}`);
      const started = Date.now();
      let session = null;
      try {
        session = await this.ort.InferenceSession.create(file('swapper'), sessionOptionsFor(device));
        const loadMs = Date.now() - started;
        const ms = await this.timeSwap(session);
        this.tried.push({ device: label, loadMs, swapMs: ms });
        log(`${label}: loaded in ${loadMs} ms, one swap ${ms} ms`);
        if (!best || ms < best.ms) {
          if (best) await best.session.release().catch(() => {});
          best = { device, session, ms };
        } else {
          await session.release().catch(() => {});
        }
      } catch (err) {
        const message = `${(err && err.message) || err}`.replace(/\s+/g, ' ').slice(0, 200);
        this.tried.push({ device: label, error: message });
        log(`${label}: not usable (${message})`);
        if (session) await session.release().catch(() => {});
      }
    }
    if (!best) throw new Error(`The face models could not run here (${this.tried.map((t) => `${t.device}: ${t.error}`).join('; ')})`);

    const options = sessionOptionsFor(best.device);
    this.provider = best.device.name;
    this.deviceId = best.device.deviceId;
    this.deviceLabel = deviceLabel(best.device);
    this.swapMs = best.ms;
    this.sessionOptions = options;
    this.swapper = best.session;
    const failed = this.tried.filter((t) => t.error);
    this.fallbackReason = best.device.name === 'cpu' && failed.length ? failed.map((t) => `${t.device}: ${t.error}`).join('; ').slice(0, 300) : '';

    onStep(`loading the face finder on ${this.deviceLabel}`);
    const det = await this.ort.InferenceSession.create(file('detector'), options);
    this.detector = new FaceDetector(this.ort, det, { size: detSize });
    onStep(`loading the identity model on ${this.deviceLabel}`);
    this.recognizer = await this.ort.InferenceSession.create(file('recognizer'), options);
    log(`using ${this.deviceLabel}`);

    onStep('reading the swap model');
    this.emap = this.loadEmap(file('swapper'));
    return { provider: this.provider, deviceId: this.deviceId, deviceLabel: this.deviceLabel, swapMs: this.swapMs, tried: this.tried, fallbackReason: this.fallbackReason };
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
