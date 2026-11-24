// Local face swap adapter, joined with Lite's face tracker:
//   - every camera frame, the tracker finds your face (fast, on the graphics chip);
//   - when the AI is free, the studio cuts out your face, aligned the way the swap model wants,
//     and sends just that small crop to the face engine process;
//   - the swapped face that comes back is bent onto your face as it is *now*, every frame, so the
//     picture moves at full camera speed however long the AI takes per face.
// If the tracker can't start, whole frames go to the engine instead (its own face finder).
import { FaceLandmarker } from '@mediapipe/tasks-vision';
import { output } from './state.js';
import { PointSmoother } from './ai-lite.js';
import { FaceWarp } from './face-warp.js';
import { MESH_POINT_COUNT, alignmentTarget, composeAffine, fivePoints, invertAffine, similarity } from './face-points.js';

const CROP = 128; // the swap model's input size

const FRESH_MS = 1000; // older results than this are not shown (the real picture shows instead)

export class LocalFaceAI {
  constructor({ api, onState, onError, onProgress = () => {} }) {
    this.api = api;
    this.onState = onState;
    this.onError = onError;
    this.onProgress = onProgress;
    this.state = 'off';
    this.provider = '';
    this.deviceLabel = '';
    this.tried = [];
    this.step = ''; // what the engine is doing during a long start-up step
    this.gpuCount = 0; // graphics chips found by the hardware check (0 = unknown)
    this.swapFile = null; // the fast swap model's file, when chosen (null = the standard model)
    this.target = null;
    this.dirty = false;
    this.running = false;
    this.engineReady = false;
    this.applied = { refKey: null, enhance: false };
    this.waiters = [];
    this.inFlight = null; // { id, image }
    this.frameId = 0;
    this.lastResultAt = 0;
    this.stats = { ms: 0, fps: 0 };
    this.resultTimes = [];

    // What the engine is fed (mirrored like the output) and the finished picture.
    this.input = document.createElement('canvas');
    this.input.width = output.width;
    this.input.height = output.height;
    // Kept on the graphics chip: the face tracker and the face warp read it there (a processor-side
    // canvas cost a copy out of the chip and back every frame). Only the small face crop is read back.
    this.inputCtx = this.input.getContext('2d', { alpha: false });
    this.canvas = document.createElement('canvas');
    this.canvas.width = output.width;
    this.canvas.height = output.height;
    this.ctx = this.canvas.getContext('2d', { alpha: false });
    this.ctx.imageSmoothingQuality = 'high';
    this.faceCanvas = document.createElement('canvas');
    this.faceCtx = this.faceCanvas.getContext('2d');

    // Tracker mode (see the top of this file).
    this.tracker = null;
    this.trackerFailed = false;
    this.smoother = new PointSmoother();
    this.raw = new Float32Array(MESH_POINT_COUNT * 2);
    this.points = null;
    this.lastTs = 0;
    this.lastTickAt = 0;
    this.warp = null;
    this.crop = document.createElement('canvas');
    this.crop.width = CROP;
    this.crop.height = CROP;
    this.cropCtx = this.crop.getContext('2d', { willReadFrequently: true, alpha: false });
    this.cropCtx.imageSmoothingQuality = 'high';
    this.stats.stages = null;
    this.look = null; // hair and outfit from the photo (look-layers.js), set by the studio
    this.lookOpts = { hair: false, outfit: false };

    api.onEngineMessage((msg) => this.handleMessage(msg));
    api.onEngineExit(() => {
      this.engineReady = false;
      this.inFlight = null;
      this.applied = { refKey: null, enhance: false };
      this.rejectWaiters(new Error('The face engine stopped'));
      if (this.target && performance.now() - (this.reloadAt || -1e9) > 10000) {
        this.setState('error');
        this.onError('The local face engine stopped. Toggle AI to restart it.');
      }
    });
  }

  setState(state) {
    if (this.state === state) return;
    this.state = state;
    this.onState(state);
  }

  get live() {
    if (this.tracker) {
      // The picture is redrawn every frame (with the face whenever one is tracked).
      return this.state === 'generating' && this.lastResultAt > 0 && performance.now() - this.lastTickAt < FRESH_MS;
    }
    return this.state === 'generating' && performance.now() - this.lastResultAt < FRESH_MS;
  }

  // target: null (off) or { refKey, refBlob, enhance }
  // Restarts the engine (to load a different swap model).
  async reload() {
    this.reloadAt = performance.now();
    try {
      await this.api.engineStop();
    } catch {
      // already stopped
    }
    this.engineReady = false;
    this.inFlight = null;
    this.applied = { refKey: null, enhance: false };
    if (this.warp) this.warp.clear();
    // Through the same queue as every other change, so a reload never starts a second engine load
    // alongside one already running.
    this.dirty = true;
    this.pump();
  }

  setTarget(target) {
    this.target = target;
    this.dirty = true;
    this.pump();
  }

  async pump() {
    if (this.running) return;
    this.running = true;
    try {
      while (this.dirty) {
        this.dirty = false;
        try {
          await this.apply(this.target);
        } catch {
          // apply reports its own errors; keep going if something changed meanwhile
        }
      }
    } finally {
      this.running = false;
    }
  }

  async apply(target) {
    if (!target) {
      this.setState('off');
      return;
    }
    try {
      const models = await this.api.modelsStatus();
      if (!models.core) {
        this.setState('needs-models');
        return;
      }
      if (!this.engineReady) await this.startEngine();
      if (this.target !== target) return; // changed while starting; the next pass handles it

      if (target.enhance !== this.applied.enhance) {
        if (target.enhance && !models.boost) {
          this.onError('Download the face detail boost first (Settings in the AI panel).');
        } else {
          this.step = target.enhance ? 'loading the face detail model' : '';
          this.send({ type: 'enhance', on: target.enhance });
          const res = await this.waitFor('enhance', { idle: 90000, cap: 600000 });
          this.applied.enhance = Boolean(res.on);
          this.step = '';
        }
      }

      const alreadyTried = this.applied.refKey === `missing:${target.refKey}`;
      if (target.refKey !== this.applied.refKey && !alreadyTried) {
        if (!target.refBlob) {
          this.send({ type: 'clear-reference' });
          this.applied.refKey = target.refKey;
        } else {
          this.setState('loading');
          const ref = await blobToPixels(target.refBlob, 1024);
          this.step = 'reading the face in your photo';
          this.send({ type: 'reference', width: ref.width, height: ref.height, pixels: ref.data });
          const res = await this.waitFor('reference');
          this.applied.refKey = res.found ? target.refKey : `missing:${target.refKey}`;
          this.step = '';
          if (this.warp) this.warp.clear(); // don't show the previous person's face
        }
      }

      if (!this.tracker && !this.trackerFailed) await this.startTracker();

      if (!target.refBlob) this.setState('needs-reference');
      else if (this.applied.refKey !== target.refKey) this.setState('no-face');
      else this.setState('generating');
    } catch (err) {
      this.setState('error');
      this.onError(`Local AI: ${(err && err.message) || err}`);
    }
  }

  // The same face tracker Lite uses. Without it (no WebGL 2, say) the engine's own face finder is used.
  async startTracker() {
    try {
      this.step = 'starting the face tracker';
      this.onProgress(this.step);
      const wasm = await this.api.readAsset('mediapipe/vision_wasm_internal.wasm');
      const model = await this.api.readAsset('mediapipe/face_landmarker.task');
      const fileset = {
        wasmLoaderPath: new URL('mediapipe/vision_wasm_internal.js', document.baseURI).href,
        wasmBinaryPath: URL.createObjectURL(new Blob([wasm], { type: 'application/wasm' })),
      };
      const make = (delegate) =>
        FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetBuffer: new Uint8Array(model), delegate },
          runningMode: 'VIDEO',
          numFaces: 1,
          minFaceDetectionConfidence: 0.5,
          minFacePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
      try {
        this.tracker = await make('GPU');
      } catch {
        this.tracker = await make('CPU');
      }
      this.warp = new FaceWarp();
    } catch (err) {
      this.tracker = null;
      this.warp = null;
      this.trackerFailed = true;
      console.warn(`Local AI: tracker unavailable, using the engine's face finder: ${(err && err.message) || err}`);
    } finally {
      this.step = '';
    }
  }

  async startEngine() {
    this.setState('starting');
    this.step = 'starting the face engine';
    const res = await this.api.engineStart();
    if (!res || !res.ok) throw new Error('The face engine could not start');
    await this.waitFor('hello', { idle: 30000 });
    this.send({ type: 'load', modelsDir: res.modelsDir, gpuCount: this.gpuCount, swapFile: this.swapFile });
    // The first start can be slow: the AI runtime loads, then each graphics chip is tried.
    const loaded = await this.waitFor('loaded', { idle: 90000, cap: 600000 });
    this.provider = loaded.provider;
    this.deviceLabel = loaded.deviceLabel || '';
    this.tried = loaded.tried || [];
    this.fallbackReason = loaded.fallbackReason || '';
    this.step = '';
    this.engineReady = true;
    this.applied = { refKey: null, enhance: false };
  }

  send(message) {
    if (!this.api.enginePost(message)) throw new Error('The face engine is not connected');
  }

  // Waits for a reply. 'idle' is how long to wait without any sign of work: the engine reports
  // progress every few seconds during long steps, and each report restarts the clock. 'cap' is the
  // longest wait in total.
  waitFor(type, { idle = 60000, cap = 300000 } = {}) {
    return new Promise((resolve, reject) => {
      const started = performance.now();
      const waiter = { type, resolve, reject };
      const fail = () => {
        this.waiters = this.waiters.filter((w) => w !== waiter);
        clearTimeout(waiter.timer);
        const during = this.step ? ` while ${this.step}` : '';
        reject(new Error(`The face engine did not answer in time${during}. Its log (face-engine.log) has the details.`));
      };
      waiter.arm = () => {
        clearTimeout(waiter.timer);
        const left = cap - (performance.now() - started);
        waiter.timer = setTimeout(fail, Math.max(0, Math.min(idle, left)));
      };
      waiter.arm();
      this.waiters.push(waiter);
    });
  }

  rejectWaiters(err) {
    const pending = this.waiters;
    this.waiters = [];
    pending.forEach((w) => {
      clearTimeout(w.timer);
      w.reject(err);
    });
  }

  handleMessage(msg) {
    if (!msg || !msg.type) return;
    if (msg.type === 'result') {
      this.handleResult(msg);
      return;
    }
    if (msg.type === 'swapped') {
      this.handleSwapped(msg);
      return;
    }
    if (msg.type === 'progress') {
      const replyType = msg.request === 'load' ? 'loaded' : msg.request;
      this.waiters.filter((w) => w.type === replyType).forEach((w) => w.arm());
      if (msg.step && msg.step !== this.step) {
        this.step = msg.step;
        this.onProgress(this.step);
      }
      return;
    }
    if (msg.type === 'error') {
      if (msg.request === 'frame' || msg.request === 'swap') {
        this.inFlight = null;
        return;
      }
      const waiter = this.waiters.find((w) => w.type === msg.request || (msg.request === 'load' && w.type === 'loaded'));
      if (waiter) {
        this.waiters = this.waiters.filter((w) => w !== waiter);
        clearTimeout(waiter.timer);
        waiter.reject(new Error(msg.step && !String(msg.message).includes(msg.step) ? `${msg.message} (while ${msg.step})` : msg.message));
      } else {
        this.onError(`Local AI: ${msg.message}`);
      }
      return;
    }
    const waiter = this.waiters.find((w) => w.type === msg.type);
    if (waiter) {
      this.waiters = this.waiters.filter((w) => w !== waiter);
      clearTimeout(waiter.timer);
      waiter.resolve(msg);
    }
  }

  // Called by the compositor every frame with a function that draws the current source.
  tick(drawSource) {
    if (this.tracker) {
      this.tickTracked(drawSource);
      return;
    }
    if (this.state !== 'generating' || !this.engineReady || this.inFlight) return;
    if (this.input.width !== output.width || this.input.height !== output.height) {
      // Picture shape changed: follow it (canvas contexts keep their settings).
      this.input.width = output.width;
      this.input.height = output.height;
      this.canvas.width = output.width;
      this.canvas.height = output.height;
      this.ctx.imageSmoothingQuality = 'high';
    }
    drawSource(this.inputCtx, output.width, output.height);
    const image = this.inputCtx.getImageData(0, 0, output.width, output.height);
    const id = ++this.frameId;
    this.inFlight = { id, image, sentAt: performance.now() };
    try {
      this.send({ type: 'frame', id, width: image.width, height: image.height, pixels: image.data });
    } catch {
      this.inFlight = null;
    }
  }

  tickTracked(drawSource) {
    if (this.state !== 'generating' || !this.engineReady) return;
    const { width, height } = output;
    if (this.input.width !== width || this.input.height !== height) {
      this.input.width = width;
      this.input.height = height;
      this.canvas.width = width;
      this.canvas.height = height;
      this.smoother.reset();
      this.warp.clear();
    }
    const t0 = performance.now();
    drawSource(this.inputCtx, width, height);
    const t = performance.now();
    const ts = Math.max(this.lastTs + 1, Math.floor(t));
    this.lastTs = ts;
    let landmarks = null;
    try {
      const result = this.tracker.detectForVideo(this.input, ts);
      landmarks = result.faceLandmarks && result.faceLandmarks[0];
    } catch {
      landmarks = null;
    }
    if (landmarks) {
      for (let i = 0; i < MESH_POINT_COUNT; i += 1) {
        this.raw[i * 2] = landmarks[i].x * width;
        this.raw[i * 2 + 1] = landmarks[i].y * height;
      }
    }
    const points = this.smoother.update(landmarks ? this.raw : null, t);
    this.points = points;
    const t1 = performance.now();

    // The live picture, with the latest swapped face bent onto where your face is now.
    this.ctx.drawImage(this.input, 0, 0);
    if (this.look && this.lookOpts.outfit) this.look.drawOutfit(this.ctx, this.input, width, height, t, points, { hideHair: this.lookOpts.hideHair });
    const t2 = performance.now();
    if (points && this.warp.hasFace) {
      // Camera detail on the face: full without the detail boost; a little with it (already sharp).
      this.warp.detail = (this.sharpness ?? 0.7) * (this.applied && this.applied.enhance ? 0.3 : 1);
      this.warp.draw(width, height, points, t, this.input);
      this.ctx.globalAlpha = this.smoother.strength;
      this.ctx.drawImage(this.warp.canvas, 0, 0);
      this.ctx.globalAlpha = 1;
    }
    const t3 = performance.now();
    if (this.look && this.lookOpts.hideHair && points) this.look.hideOwnHair(this.ctx, this.input, width, height, t, points, { segmented: this.lookOpts.outfit });
    if (this.look && this.lookOpts.hair && points) this.look.drawHair(this.ctx, points, this.smoother.strength);
    const t4 = performance.now();
    this.lastTickAt = t;
    // Where each frame's time goes (smoothed), for the speed report.
    const f = this.frameMs || (this.frameMs = { source: 0, track: 0, outfit: 0, face: 0, hair: 0, total: 0 });
    const ema = (k, v) => (f[k] = f[k] ? f[k] * 0.95 + v * 0.05 : v);
    ema('source', t - t0);
    ema('track', t1 - t);
    ema('outfit', t2 - t1);
    ema('face', t3 - t2);
    ema('hair', t4 - t3);
    ema('total', t4 - t0);

    // Ask for a new face whenever the engine is free.
    if (!landmarks || !points || this.inFlight) return;
    const kps = fivePoints(points);
    const toCrop = kps && similarity(kps, alignmentTarget(CROP));
    if (!toCrop) return;
    const c = this.cropCtx;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.fillStyle = '#000';
    c.fillRect(0, 0, CROP, CROP);
    c.setTransform(toCrop[0], toCrop[3], toCrop[1], toCrop[4], toCrop[2], toCrop[5]);
    c.drawImage(this.input, 0, 0);
    const crop = c.getImageData(0, 0, CROP, CROP);
    const id = ++this.frameId;
    this.inFlight = { id, points: Float32Array.from(points), toCrop, sentAt: t };
    try {
      this.send({ type: 'swap', id, pixels: crop.data });
    } catch {
      this.inFlight = null;
    }
  }

  handleSwapped(msg) {
    const pending = this.inFlight;
    if (!pending || pending.id !== msg.id) return;
    this.inFlight = null;
    if (msg.skipped || !this.warp) return;
    const now = performance.now();
    // Output pixels -> crop -> the swapped face's own pixels.
    const toFace = composeAffine(invertAffine(msg.toCrop), pending.toCrop);
    const px = msg.pixels;
    this.warp.addFace(new Uint8Array(px.buffer, px.byteOffset, px.byteLength), msg.size, pending.points, toFace, now);
    this.lastResultAt = now;
    this.stats.ms = msg.ms || Math.round(now - pending.sentAt);
    this.stats.stages = msg.stages || null;
    this.resultTimes.push(now);
    while (this.resultTimes.length && now - this.resultTimes[0] > 1000) this.resultTimes.shift();
    this.stats.fps = this.resultTimes.length;
  }

  handleResult(msg) {
    const pending = this.inFlight;
    if (!pending || pending.id !== msg.id) return;
    this.inFlight = null;
    if (msg.skipped) return;

    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.putImageData(pending.image, 0, 0);
    for (const face of msg.faces || []) {
      const size = face.size;
      if (this.faceCanvas.width !== size) {
        this.faceCanvas.width = size;
        this.faceCanvas.height = size;
      }
      this.faceCtx.putImageData(new ImageData(new Uint8ClampedArray(face.pixels), size, size), 0, 0);
      const m = face.toFrame; // [a, c, e, b, d, f]
      this.ctx.setTransform(m[0], m[3], m[1], m[4], m[2], m[5]);
      this.ctx.drawImage(this.faceCanvas, 0, 0);
    }
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);

    const now = performance.now();
    this.lastResultAt = now;
    this.stats.ms = msg.ms || Math.round(now - pending.sentAt);
    this.resultTimes.push(now);
    while (this.resultTimes.length && now - this.resultTimes[0] > 1000) this.resultTimes.shift();
    this.stats.fps = this.resultTimes.length;
  }
}

async function blobToPixels(blob, maxSide) {
  const bitmap = await createImageBitmap(blob);
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  return ctx.getImageData(0, 0, width, height);
}
